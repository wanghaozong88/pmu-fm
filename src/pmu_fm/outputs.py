"""One forecast object feeds network mapping and asynchronous disk records."""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
from pathlib import Path

from .config import ConfigError, report_rate
from .protocol import Configuration, PMU, data_frame
from .records import iso_time


class ForecastMapping:
    def __init__(self, cfg, source, result, plan):
        self.channels = result.channels
        self.interval = result.interval
        self.entries, pmus = [], []
        ids = {p.id_code for p in source.pmus} | {source.id_code}
        selected_sources = list(dict.fromkeys(c["source_pmu"] for c in result.channels.values()))
        source_pmus = {p.id_code: p for p in source.pmus}
        if cfg["network"]["id_code"] in ids:
            raise ConfigError("Output stream ID collides with an input PMU/stream")
        for k, source_id in enumerate(selected_sources):
            ident = cfg["network"]["pmu_id_start"] + k
            if ident in ids or ident == cfg["network"]["id_code"] or source_id not in source_pmus:
                raise ConfigError("Invalid or colliding virtual PMU identity")
            names = []
            for col, (channel, ch) in enumerate(result.channels.items()):
                if ch["source_pmu"] != source_id:
                    continue
                for lead in plan["leads"]:
                    name = f"F{col:04d}_H{lead:04d}"
                    names.append(name)
                    self.entries.append(
                        {
                            "virtual_pmu": ident,
                            "wire_name": name,
                            "source_pmu": source_id,
                            "channel": channel,
                            "column": col,
                            "name": ch["name"],
                            "unit": ch["unit"],
                            "horizon_step": lead,
                            "horizon_seconds": str(lead * result.interval),
                            "reference": "frame SOC + FRACSEC/TIME_BASE",
                            "target": "reference + horizon_seconds",
                        }
                    )
            pmus.append(
                PMU(
                    f"FM_{ident}",
                    ident,
                    12,
                    analog_names=tuple(names),
                    anunits=tuple(1 for _ in names),
                    nominal_hz=source_pmus[source_id].nominal_hz,
                )
            )
        self.cfg = Configuration(
            cfg["network"]["id_code"],
            source.time_base,
            tuple(pmus),
            report_rate(cfg["model"]["interval_seconds"]),
            cfg["network"]["frame_version"],
        )
        self.cfg.validate()

    def encode(self, result):
        if result.channels != self.channels or result.interval != self.interval:
            raise ConfigError("Preprocessing output structure changed; stop and reconfigure")
        blocks = []
        for p in self.cfg.pmus:
            entries = [e for e in self.entries if e["virtual_pmu"] == p.id_code]
            source_id = entries[0]["source_pmu"]
            f = result.reference.values.get(f"{source_id}/freq", math.nan)
            df = result.reference.values.get(f"{source_id}/dfreq", math.nan)
            if not result.reference.valid([source_id]) or not math.isfinite(f) or not math.isfinite(df):
                raise ValueError("No valid same-reference source FREQ/DFREQ; network publication omitted")
            analogs = [result.values[e["horizon_step"] - 1][e["column"]] for e in entries]
            if any(not math.isfinite(v) or abs(v) > 3.402823466e38 for v in analogs):
                raise ValueError("Forecast cannot be represented as C37.118 float32")
            blocks.append(
                {
                    "stat": (result.reference.statuses[source_id] & 0x01FF) | 0x0200,
                    "freq": f,
                    "dfreq": df,
                    "analog": analogs,
                }
            )
        return data_frame(self.cfg, result.reference.timestamp, blocks, result.reference.time_quality)


class FileWriter:
    def __init__(self, directory, settings, emit):
        self.directory = Path(directory)
        self.settings, self.emit = settings, emit
        self.queue = queue.Queue(maxsize=settings["queue_size"])
        self.stop_event = threading.Event()
        self.failed, self.rows_written, self.dropped = False, 0, 0
        self.thread = threading.Thread(target=self._run, name="pmu-files", daemon=True)
        self.thread.start()

    def offer(self, kind, value):
        if self.failed or self.stop_event.is_set():
            self.dropped += 1
            return False
        try:
            self.queue.put_nowait((kind, value))
            return True
        except queue.Full:
            self.dropped += 1
            # Do not recursively send a writer error back into the same full queue.
            self.emit("file_queue_overflow", "Disk queue full; record dropped", persist=False)
            return False

    def _run(self):
        prediction = events = writer = None
        segment, segment_rows = 0, 0
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            events = (self.directory / "events.jsonl").open("w", encoding="utf-8", newline="\n")
            last_flush = time.monotonic()
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    kind, value = self.queue.get(timeout=0.1)
                except queue.Empty:
                    kind = None
                if kind == "metadata":
                    name, data = value
                    (self.directory / name).write_text(
                        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
                    )
                elif kind == "text":
                    name, content = value
                    (self.directory / name).write_text(content, encoding="utf-8")
                elif kind == "event":
                    events.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
                elif kind == "forecast" and self.settings["enabled"]:
                    if value.trace:
                        (self.directory / f"window_{value.forecast_id}.json").write_text(
                            json.dumps(value.trace, default=str), encoding="utf-8"
                        )
                    for row in value.rows():
                        if prediction is None or segment_rows >= self.settings["rows_per_segment"]:
                            if prediction:
                                prediction.close()
                            segment += 1
                            prediction = (self.directory / f"predictions_{segment:05d}.csv").open(
                                "w", encoding="utf-8", newline=""
                            )
                            writer = csv.DictWriter(prediction, fieldnames=list(row))
                            writer.writeheader()
                            segment_rows = 0
                        writer.writerow(row)
                        segment_rows += 1
                        self.rows_written += 1
                if time.monotonic() - last_flush >= self.settings["flush_seconds"]:
                    events.flush()
                    if prediction:
                        prediction.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            self.failed = True
            self.emit("file_error", f"{type(exc).__name__}: {exc}", persist=False)
        finally:
            for handle in (prediction, events):
                if handle:
                    try:
                        handle.close()
                    except OSError as exc:
                        self.failed = True
                        self.emit("file_error", str(exc), persist=False)
            if not self.failed:
                try:
                    (self.directory / "writer_status.json").write_text(
                        json.dumps(
                            {
                                "closed_and_flushed": True,
                                "rows_written": self.rows_written,
                                "dropped": self.dropped,
                                "closed_at": iso_time(time.time_ns()),
                            }
                        ),
                        encoding="utf-8",
                    )
                except OSError as exc:
                    self.failed = True
                    self.emit("file_error", str(exc), persist=False)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.emit(
                "file_close_timeout", "OS file I/O did not finish within 5s; flush unconfirmed", persist=False
            )


def event(kind, message, **details):
    return {"event": kind, "message": message, "at": iso_time(time.time_ns()), **details}
