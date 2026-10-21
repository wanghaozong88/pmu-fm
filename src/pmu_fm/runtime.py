"""Shared GUI/CLI runtime: bounded I/O, latest pending job, one local model."""

from __future__ import annotations

import importlib.metadata
import queue
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from .buffer import History
from .config import ConfigError, dumps, normalize, sample_plan, timestamp_tolerance, validate
from .models import LocalModel
from .network import Publisher, Receiver
from .outputs import FileWriter, ForecastMapping, event
from .protocol import channels, ns
from .worker import Worker


class Runtime:
    def __init__(self, cfg, *, _model_factory=LocalModel, _test_config_frame=None):
        self.cfg = normalize(cfg)
        self.test_config_frame = _test_config_frame
        self.model_factory = _model_factory  # test dependency injection; no fake CLI/config option
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.state = {
            "phase": "stopped",
            "last_error": "",
            "worker": "stopped",
            "window_samples": 0,
            "last_inference_seconds": None,
            "source_age_seconds": None,
        }
        self.counters = Counter()
        self.thread = None
        self.writer = self.receiver = self.publisher = self.worker = None
        self.run_id = None
        self.run_directory = None
        self.latest_result = None

    def emit(self, kind, message, persist=True, **details):
        with self.lock:
            self.counters[kind] += 1
            self.state["last_event"] = f"{kind}: {message}"
            if "error" in kind or kind in ("fatal", "worker_terminated", "configuration_changed"):
                self.state["last_error"] = message
        if persist and self.writer:
            self.writer.offer("event", event(kind, message, run_id=self.run_id, **details))

    def snapshot(self):
        with self.lock:
            result = dict(self.state) | {
                "counters": dict(self.counters),
                "run_id": self.run_id,
                "run_directory": str(self.run_directory or ""),
            }
        result["running"] = self.thread is not None and self.thread.is_alive()
        if self.receiver and self.receiver.transport:
            result["input_state"] = self.receiver.transport.phase
            result["input_peer"] = self.receiver.transport.peer
            result["input_command_version"] = self.receiver.transport.command_version
        if self.receiver and self.receiver.last_data_mono is not None:
            result["input_freshness_seconds"] = time.monotonic() - self.receiver.last_data_mono
        if self.writer:
            result.update(
                file_rows=self.writer.rows_written,
                file_dropped=self.writer.dropped,
                file_failed=self.writer.failed,
            )
        return result

    def start(self):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Already running; stop before changing configuration")
        validate(self.cfg, require_weights=self.model_factory is LocalModel)
        self.stop_event.clear()
        self.counters.clear()
        self.run_id = uuid.uuid4().hex
        self.run_directory = Path(self.cfg["files"]["directory"]) / self.run_id
        self.state.update(
            phase="starting",
            last_error="",
            worker="loading",
            window_samples=0,
            source_age_seconds=None,
            source_frame_version=None,
        )
        self.latest_result = None
        self.writer = self.receiver = self.publisher = self.worker = None
        self.thread = threading.Thread(target=self._run, name="pmu-runtime", daemon=True)
        self.thread.start()

    def _run(self):
        source = history = plan = pending = mapping = channel_info = None
        epoch, active_epoch, worker_restarts = 0, -1, 0
        active_forecast_id = None
        mapped_channels = None
        try:
            self.writer = FileWriter(self.run_directory, self.cfg["files"], self.emit)
            self.writer.offer("text", ("config.toml", dumps(self.cfg)))
            versions = {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            }
            self.writer.offer("metadata", ("environment.json", {"packages": versions}))
            if self.cfg["network"]["enabled"]:
                self.publisher = Publisher(self.cfg["network"], self.emit)
                self.emit("listening", f"TCP {self.cfg['network']['host']}:{self.publisher.port}")
            incoming = queue.Queue(maxsize=self.cfg["source"]["queue_size"])
            self.worker = Worker(self.cfg, self.model_factory)
            self.worker.start()
            self.receiver = Receiver(
                self.cfg["source"], incoming, self.emit, _test_config_frame=self.test_config_frame
            )
            self.receiver.start()
            self.state["phase"] = "running"
            self.emit("started", "Online run started")
            while not self.stop_event.wait(0.02):
                # Bound each drain so high-rate input cannot starve supervision.
                for _ in range(256):
                    try:
                        kind, value = incoming.get_nowait()
                    except queue.Empty:
                        break
                    if kind in ("disconnected", "config_pending"):
                        history, pending = None, None
                        self.state["window_samples"] = 0
                        self.state["source_age_seconds"] = None
                        self.state["source_frame_version"] = None
                        epoch += 1
                        self.emit(kind, "History cleared; awaiting fresh configuration and window")
                    elif kind == "config":
                        if source is not None and source != value:
                            self.writer.offer("metadata", ("input_config_changed.json", value.to_dict()))
                            raise ConfigError(
                                "Upstream configuration changed. Mapping paused; stop/reconfigure/restart required"
                            )
                        source = value
                        # Retain the received configuration even if timing validation fails.
                        self.writer.offer("metadata", ("input_config.json", source.to_dict()))
                        channel_info = {c["id"]: c for c in channels(source)}
                        missing = set(self.cfg["source"]["channels"]) - set(channel_info)
                        if missing:
                            raise ConfigError(f"Selected channels missing from CFG-2: {sorted(missing)}")
                        if history is None:
                            plan = sample_plan(self.cfg, source.interval, source.time_base)
                            tolerance = timestamp_tolerance(
                                self.cfg["source"]["timestamp_tolerance_seconds"],
                                min(source.interval, plan["dt"]),
                                source.time_base,
                            )
                            self.writer.offer(
                                "metadata",
                                (
                                    "input_timing_policy.json",
                                    {
                                        "configured_tolerance_seconds": str(
                                            self.cfg["source"]["timestamp_tolerance_seconds"]
                                        ),
                                        "effective_tolerance_seconds": str(tolerance),
                                        "source_interval_seconds": str(source.interval),
                                        "policy": "Bounded nominal-grid matching; original timestamps retained; gaps and ambiguous slots rejected",
                                    },
                                ),
                            )
                            history = History(
                                self.cfg["source"]["channels"],
                                source.interval,
                                plan["source_context"],
                                self.cfg["model"]["interval_seconds"],
                                self.cfg["source"]["reorder_seconds"],
                                max_missing=self.cfg["script"]["max_missing_samples"],
                                timestamp_tolerance_seconds=self.cfg["source"]["timestamp_tolerance_seconds"],
                            )
                            epoch += 1
                            self.writer.offer("metadata", ("source_channels.json", channel_info))
                            self.writer.offer("metadata", ("sample_plan.json", plan))
                            self.emit(
                                "configured",
                                f"{len(source.pmus)} PMU blocks; input interval {source.interval}s; CFG-2 version {source.version}",
                            )
                    elif kind == "sample" and history is not None:
                        history.add(value)
                        self.state["source_frame_version"] = value.version
                        self.counters["samples"] += 1
                        self.state["source_age_seconds"] = (time.time_ns() - ns(value.timestamp)) / 1e9
                if history is not None:
                    for window, reason in history.ready():
                        if window is None:
                            self.emit("window_skipped", reason)
                        else:
                            if pending is not None:
                                self.emit("pending_replaced", "Kept only the latest eligible window")
                            pending = window
                    self.state["window_samples"] = min(len(history.stamps), history.count)
                    self.state["history_counters"] = dict(history.counters)
                for kind, value in self.worker.poll():
                    if kind == "ready":
                        self.writer.offer("metadata", ("model.json", value))
                        self.emit("model_ready", "Local model loaded once in worker")
                    elif kind == "fatal":
                        raise RuntimeError(value)
                    elif kind == "job_error":
                        self.emit("inference_error", value, forecast_id=active_forecast_id)
                    elif kind == "result":
                        if active_epoch != epoch:
                            self.emit(
                                "stale_result_discarded",
                                "Source reset while inference was running",
                                forecast_id=value.forecast_id,
                            )
                            continue
                        if mapped_channels is not None and mapped_channels != value.channels:
                            raise ConfigError("Script output channel structure changed; stop and reconfigure")
                        if mapped_channels is None:
                            self.writer.offer("metadata", ("forecast_channels.json", value.channels))
                        mapped_channels = value.channels
                        self.latest_result = value
                        self.state["last_inference_seconds"] = value.elapsed_seconds
                        self.emit(
                            "forecast_completed", "Local forecast completed", forecast_id=value.forecast_id
                        )
                        self.writer.offer("forecast", value)
                        if self.publisher:
                            if mapping is None:
                                mapping = ForecastMapping(self.cfg, source, value, plan)
                                self.publisher.configure(mapping.cfg)
                                self.writer.offer(
                                    "metadata",
                                    (
                                        "output_mapping.json",
                                        {
                                            "configuration": mapping.cfg.to_dict(),
                                            "channels": mapping.entries,
                                            "freq_dfreq": "Original same-reference source measurements; never predictions",
                                            "stat": "data-modified bit set; forecast semantics are application-level",
                                        },
                                    ),
                                )
                            try:
                                self.publisher.publish(mapping.encode(value), value.forecast_id)
                            except ValueError as exc:
                                self.emit("network_omitted", str(exc), forecast_id=value.forecast_id)
                violation = self.worker.violation()
                if violation:
                    failed_phase = self.worker.phase
                    self.worker.close()
                    self.emit("worker_terminated", violation, forecast_id=active_forecast_id)
                    worker_restarts += 1
                    if failed_phase in ("loading", "script_import") or worker_restarts > 3:
                        raise RuntimeError("Worker failed during initialization or restart limit (3) reached")
                    self.worker.start()
                if pending is not None and self.worker.ready and not self.worker.busy:
                    forecast_id = uuid.uuid4().hex
                    if self.worker.submit((pending, channel_info, plan, self.run_id, forecast_id)):
                        active_epoch, pending = epoch, None
                        active_forecast_id = forecast_id
                self.state["worker"] = self.worker.phase
        except Exception as exc:
            self.emit("fatal", f"{type(exc).__name__}: {exc}")
            self.state["phase"] = "failed"
        finally:
            if self.receiver:
                self.receiver.close()
            if self.worker:
                self.worker.close()
            if self.publisher:
                self.publisher.close()
            self.emit("stopped", "Sockets closed and worker stopped; upstream STOP was not sent")
            if self.state["phase"] != "failed":
                self.state["phase"] = "stopped"
            self.state["worker"] = "stopped"
            if self.writer:
                self.writer.offer(
                    "metadata",
                    (
                        "final_status.json",
                        self.snapshot()
                        | {
                            "running": False,
                            "file_flush": "See writer_status.json for final disk completion",
                        },
                    ),
                )
                self.writer.close()

    def request_stop(self):
        self.stop_event.set()

    def stop(self, timeout=12):
        self.request_stop()
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                raise TimeoutError("Runtime cleanup has not finished; inspect status before restarting")
