"""Bounded event-time history with a finite lateness watermark."""

from __future__ import annotations

import bisect
import math
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from .config import timestamp_tolerance
from .protocol import Sample


@dataclass(frozen=True)
class Window:
    samples: tuple[Sample | None, ...]
    channel_ids: tuple[str, ...]
    interval: Fraction
    configured_tolerance: Fraction = Fraction(0)

    @property
    def tolerance(self):
        return timestamp_tolerance(self.configured_tolerance, self.interval, self.reference.time_base)

    @property
    def max_timestamp_deviation(self):
        return max(
            abs(s.timestamp - (self.reference.timestamp - (len(self.samples) - 1 - i) * self.interval))
            for i, s in enumerate(self.samples)
            if s is not None
        )

    @property
    def reference(self):
        return self.samples[-1]

    @property
    def start(self):
        return (
            self.samples[0].timestamp
            if self.samples[0]
            else self.reference.timestamp - (len(self.samples) - 1) * self.interval
        )

    @property
    def timestamps(self):
        return tuple(
            s.timestamp if s else self.reference.timestamp - (len(self.samples) - 1 - i) * self.interval
            for i, s in enumerate(self.samples)
        )


class History:
    def __init__(
        self,
        channels,
        interval,
        count,
        call_interval,
        reorder_seconds,
        capacity=None,
        max_missing=0,
        timestamp_tolerance_seconds=0.0,
    ):
        self.channels, self.interval, self.count = tuple(channels), interval, count
        self.call_interval = Fraction(str(call_interval))
        self.reorder = Fraction(str(reorder_seconds))
        self.capacity = capacity or count + math.ceil(self.reorder / interval) + 4
        self.stamps, self.samples = [], {}
        self.committed, self.next_ref = None, None
        self.latest_arrival = time.monotonic()
        self.counters = Counter()
        self.sources = {int(c.split("/")[0]) for c in channels}
        self.max_missing = min(max_missing, count // 4)  # never repair more than 25% of a window
        self.first_seen = None
        self.configured_tolerance = timestamp_tolerance(timestamp_tolerance_seconds, interval)

    def add(self, sample: Sample):
        t = sample.timestamp
        if t in self.samples:
            self.counters["duplicate"] += 1
            return
        if self.committed is not None and t <= self.committed:
            self.counters["late"] += 1
            return
        bisect.insort(self.stamps, t)
        # Retain selected values plus the same-reference FREQ/DFREQ needed by output.
        retain = set(self.channels) | {f"{p}/{f}" for p in self.sources for f in ("freq", "dfreq")}
        self.samples[t] = Sample(
            t,
            sample.received_ns,
            {k: v for k, v in sample.values.items() if k in retain},
            dict(sample.statuses),
            sample.time_quality,
            sample.soc,
            sample.ticks,
            sample.time_base,
            dict(sample.digital),
            sample.version,
        )
        self.first_seen = t if self.first_seen is None else min(t, self.first_seen)
        self.latest_arrival = time.monotonic()
        while len(self.stamps) > self.capacity:
            del self.samples[self.stamps.pop(0)]
            self.counters["evicted"] += 1

    def ready(self, mono=None) -> list[tuple[Window | None, str]]:
        if not self.stamps:
            return []
        mono = time.monotonic() if mono is None else mono
        watermark = self.stamps[-1] - self.reorder
        if mono - self.latest_arrival >= float(self.reorder):
            watermark = self.stamps[-1]
        ready = []
        begin = bisect.bisect_right(self.stamps, self.committed) if self.committed is not None else 0
        for t in self.stamps[begin : bisect.bisect_right(self.stamps, watermark)]:
            self.committed = t
            if self.next_ref is None:
                self.next_ref = t + (self.count - 1) * self.interval
            tolerance = timestamp_tolerance(
                self.configured_tolerance, self.interval, self.samples[t].time_base
            )
            if t + tolerance < self.next_ref:
                continue
            missed = max(0, int((t - self.next_ref + tolerance) // self.call_interval))
            self.counters["schedule_gap"] += missed
            self.next_ref += (missed + 1) * self.call_interval
            end = bisect.bisect_right(self.stamps, t)
            start = bisect.bisect_left(self.stamps, t - (self.count - 1) * self.interval - tolerance)
            stamps = self.stamps[start:end]
            reason = ""
            if t - (self.count - 1) * self.interval + tolerance < self.first_seen:
                reason = "warming_up"
            else:
                samples = []
                for i in range(self.count):
                    expected = t - (self.count - 1 - i) * self.interval
                    pos = bisect.bisect_left(stamps, expected - tolerance)
                    upper = bisect.bisect_right(stamps, expected + tolerance)
                    if upper - pos > 1:
                        reason = "ambiguous_timestamps"
                        break
                    actual = (
                        stamps[pos]
                        if pos < len(stamps) and abs(stamps[pos] - expected) <= tolerance
                        else None
                    )
                    samples.append(self.samples[actual] if actual is not None else None)
                samples = tuple(samples)
                absent = sum(s is None for s in samples)
                nonfinite = sum(
                    s is not None and any(not math.isfinite(s.values.get(c, math.nan)) for c in self.channels)
                    for s in samples
                )
                if reason:
                    pass
                elif len(stamps) != len(samples) - absent:
                    reason = "off_grid_samples"
                elif absent + nonfinite > self.max_missing:
                    reason = "missing_or_irregular" if absent else "nonfinite_or_missing"
                elif any(s is not None and not s.valid(self.sources) for s in samples):
                    reason = "invalid_quality"
            if reason:
                self.counters[reason] += 1
                ready.append((None, reason))
            else:
                ready.append((Window(samples, self.channels, self.interval, self.configured_tolerance), ""))
        return ready
