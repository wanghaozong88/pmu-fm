"""Long-lived model worker; supervisor can terminate actual execution."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from pathlib import Path

from .models import LocalModel, enforce_offline


def worker_main(settings, jobs, messages, factory, script_content):
    # Heavy imports happen here so that GUI and receiver stay independent.
    enforce_offline()
    model = factory(settings["model"])
    try:
        messages.put(("phase", "loading"))
        info = model.load()
        messages.put(("phase", "script_import"))
        from .preprocessing import inverse, load_script, preprocess
        from .protocol import ns
        from .records import Forecast
        import pandas as pd

        module, script_hash = load_script(settings["script"]["path"], script_content)
        info = info | {"script_sha256": script_hash}
        messages.put(("ready", info))
        while True:
            job = jobs.get()
            if job is None:
                break
            window, channel_info, plan, run_id, forecast_id = job
            start = time.monotonic()
            try:
                messages.put(("phase", "preprocessing"))
                frame, metadata = preprocess(
                    window, channel_info, module, plan, settings["model"]["max_channels"]
                )
                messages.put(("phase", "predicting"))
                points = model.predict(frame.to_numpy(), plan["horizon"])
                messages.put(("phase", "inverse_transform"))
                future = pd.to_datetime(
                    [ns(window.reference.timestamp + i * plan["dt"]) for i in range(1, plan["horizon"] + 1)],
                    unit="ns",
                    utc=True,
                )
                predicted, out_meta = inverse(
                    points, future, metadata, module, plan, settings["model"]["max_channels"]
                )
                trace = None
                if settings["files"]["debug_windows"]:
                    trace = {
                        "model_input": {
                            "timestamps_ns": frame.index.as_unit("ns").asi8.tolist(),
                            "columns": list(frame.columns),
                            "values": frame.to_numpy().tolist(),
                        },
                        "raw_selected": {
                            "timestamps_exact": [str(t) for t in window.timestamps],
                            "columns": window.channel_ids,
                            "values": [
                                [s.values[c] if s else None for c in window.channel_ids]
                                for s in window.samples
                            ],
                        },
                        "metadata": metadata,
                    }
                result = Forecast(
                    run_id,
                    forecast_id,
                    info,
                    window.reference,
                    window.start,
                    plan["dt"],
                    time.time_ns(),
                    out_meta["channels"],
                    predicted.to_numpy().tolist(),
                    str(out_meta["quality"]),
                    time.monotonic() - start,
                    trace,
                    window.tolerance,
                    window.max_timestamp_deviation,
                )
                messages.put(("result", result))
            except Exception as exc:
                messages.put(("job_error", f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        messages.put(("fatal", f"{type(exc).__name__}: {exc}"))
    finally:
        model.close()


class Worker:
    def __init__(self, settings, factory=LocalModel):
        self.settings, self.factory = settings, factory
        self.process = None
        self.ready, self.busy = False, False
        self.phase, self.phase_start = "stopped", time.monotonic()
        self.ctx = mp.get_context("spawn")
        self.jobs = self.messages = None
        path = settings["script"]["path"]
        self.script_content = Path(path).read_bytes() if path else None
        if self.script_content and len(self.script_content) > 1_000_000:
            raise ValueError("Trusted script exceeds 1 MB source limit")

    def start(self):
        self.jobs = self.ctx.Queue(maxsize=1)
        self.messages = self.ctx.Queue(maxsize=8)
        self.process = self.ctx.Process(
            target=worker_main,
            args=(self.settings, self.jobs, self.messages, self.factory, self.script_content),
            daemon=True,
        )
        self.process.start()
        self.ready, self.busy = False, False
        self.phase, self.phase_start = "loading", time.monotonic()

    def submit(self, job):
        if not self.ready or self.busy:
            return False
        self.jobs.put_nowait(job)
        self.busy = True
        self.phase, self.phase_start = "preprocessing", time.monotonic()
        return True

    def poll(self):
        result = []
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "phase":
                self.phase, self.phase_start = value, time.monotonic()
            elif kind == "ready":
                self.ready, self.phase = True, "idle"
            elif kind in ("result", "job_error"):
                self.busy, self.phase = False, "idle"
            result.append((kind, value))
        return result

    def violation(self):
        if self.process is None:
            return None
        if not self.process.is_alive():
            return f"Worker exited unexpectedly (exit code {self.process.exitcode})"
        if self.phase == "loading":
            timeout = self.settings["model"]["load_timeout_seconds"]
        elif self.phase in ("preprocessing", "inverse_transform", "script_import"):
            timeout = self.settings["script"]["timeout_seconds"]
        elif self.phase == "predicting":
            timeout = self.settings["model"]["timeout_seconds"]
        else:
            timeout = None
        if timeout is not None and time.monotonic() - self.phase_start > timeout:
            return f"{self.phase} exceeded {timeout}s; execution terminated"
        import psutil

        try:
            parent = psutil.Process(self.process.pid)
            rss = parent.memory_info().rss
            for child in parent.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            if rss > self.settings["model"]["memory_mb"] * 1024 * 1024:
                return "Worker RSS limit exceeded; execution terminated"
        except psutil.NoSuchProcess:
            return "Worker disappeared"
        return None

    def close(self):
        if self.process is None:
            return
        import psutil

        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        if self.ready and not self.busy:
            try:
                self.jobs.put_nowait(None)
                self.process.join(timeout=0.5)
            except queue.Full:
                pass
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except psutil.NoSuchProcess:
                pass
        self.process.close()
        self.process = None
        for q in (self.jobs, self.messages):
            q.cancel_join_thread()
            q.close()
        self.ready, self.busy, self.phase = False, False, "stopped"
