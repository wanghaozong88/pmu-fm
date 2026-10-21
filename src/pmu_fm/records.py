from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction

from .protocol import Sample, ns


def iso_time(value: Fraction | int) -> str:
    nano = ns(value) if isinstance(value, Fraction) else value
    secs, remainder = divmod(nano, 1_000_000_000)
    return datetime.fromtimestamp(secs, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{remainder:09d}Z"


@dataclass
class Forecast:
    run_id: str
    forecast_id: str
    model: dict
    reference: Sample
    window_start: Fraction
    interval: Fraction
    generated_ns: int
    channels: dict
    values: list[list[float]]  # horizon x channels
    quality: str
    elapsed_seconds: float
    trace: dict | None = None
    timestamp_tolerance: Fraction = Fraction(0)
    max_timestamp_deviation: Fraction = Fraction(0)

    def rows(self):
        for step, row in enumerate(self.values, 1):
            target = self.reference.timestamp + step * self.interval
            for (channel_id, ch), value in zip(self.channels.items(), row):
                yield {
                    "run_id": self.run_id,
                    "forecast_id": self.forecast_id,
                    "model": self.model["model_id"],
                    "model_version": self.model["package_version"],
                    "model_revision": self.model["revision"],
                    "weights_sha256": self.model["weights_sha256"],
                    "script_sha256": self.model.get("script_sha256", ""),
                    "source_pmu": ch["source_pmu"],
                    "source_frame_version": self.reference.version,
                    "source_timestamp_tolerance_seconds": str(self.timestamp_tolerance),
                    "source_max_timestamp_deviation_seconds": str(self.max_timestamp_deviation),
                    "channel": channel_id,
                    "channel_name": ch["name"],
                    "window_start": iso_time(self.window_start),
                    "t_ref": iso_time(self.reference.timestamp),
                    "t_ref_exact": str(self.reference.timestamp),
                    "t_ref_soc": self.reference.soc,
                    "t_ref_ticks": self.reference.ticks,
                    "time_base": self.reference.time_base,
                    "target_time": iso_time(target),
                    "target_time_exact": str(target),
                    "generated_at": iso_time(self.generated_ns),
                    "horizon_step": step,
                    "horizon_seconds": str(step * self.interval),
                    "input_interval_seconds": str(self.interval),
                    "value": value,
                    "unit": ch["unit"],
                    "valid": True,
                    "statistic": self.model["point_statistic"],
                    "quality": self.quality,
                }
