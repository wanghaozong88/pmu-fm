import csv
import socket
import time
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from pmu_fm.buffer import History
from pmu_fm.config import DEFAULTS, ConfigError, dumps, load, sample_plan, validate
from pmu_fm.network import inspect_source
from pmu_fm.protocol import Framer, Sample, START, STOP, command_frame
from pmu_fm.runtime import Runtime
from tests.helpers import FakeModel, HungModel, Simulator, SlowModel, wait_for


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def config(tmp_path, source):
    cfg = deepcopy(DEFAULTS)
    cfg["source"].update(
        port=source.port,
        channels=["10/freq"],
        reorder_seconds=0.1,
        reconnect_seconds=0.1,
        timeout_seconds=1.0,
    )
    cfg["network"].update(port=free_port(), leads_seconds=[0.1])
    cfg["model"].update(context_seconds=0.2, horizon_seconds=0.1, interval_seconds=0.1)
    cfg["files"].update(directory=str(tmp_path / "runs"), rows_per_segment=4)
    return cfg


def sample(t, value=60, status=0):
    t = Fraction(t)
    soc = t.numerator // t.denominator
    return Sample(
        t,
        time.time_ns(),
        {"10/freq": value, "10/dfreq": 0.1},
        {10: status},
        0,
        soc,
        round((t - soc) * 1_000_000),
        1_000_000,
        {},
    )


def test_history_lateness_duplicates_gaps_and_snapshot():
    h = History(["10/freq"], Fraction(1, 10), 3, 0.1, 0.1)
    for t in ("1", "1.2", "1.1", "1.1", "1.3"):
        h.add(sample(t))
    out = [w for w, _ in h.ready() if w]
    assert out[0].reference.timestamp == Fraction("1.2")
    h.add(sample("1.15"))
    assert h.counters["late"] == 1 and h.counters["duplicate"] == 1
    assert len(out[0].samples) == 3
    h.add(sample("1.5"))
    h.add(sample("1.6", status=0x2000))
    results = h.ready(h.latest_arrival + 1)
    assert any(reason == "missing_or_irregular" for _, reason in results)
    h.add(sample("1.7"))
    h.add(sample("1.8"))
    assert any(reason == "invalid_quality" for _, reason in h.ready(h.latest_arrival + 1))
    for n in range(100):
        h.add(sample(2 + Fraction(n, 10)))
    assert len(h.samples) <= h.capacity


def test_static_config_and_exact_steps(tmp_path, monkeypatch):
    cfg = deepcopy(DEFAULTS)
    cfg["source"]["channels"] = ["10/freq"]
    monkeypatch.setattr(
        socket, "create_connection", lambda *_a, **_k: pytest.fail("static validation connected")
    )
    validate(cfg, require_weights=False)
    path = tmp_path / "config.toml"
    path.write_text(dumps(cfg), encoding="utf-8")
    assert load(path)["schema_version"] == 1
    with pytest.raises(ConfigError):
        sample_plan(cfg, Fraction(3, 100))
    cfg["source"]["allow_start"] = "true"
    with pytest.raises(ConfigError, match="expected bool"):
        validate(cfg, require_weights=False)


def test_online_inspection_never_starts_or_stops():
    with Simulator() as src:
        cfg = inspect_source({"host": "127.0.0.1", "port": src.port, "id_code": 1, "timeout_seconds": 2})
        assert len(cfg.pmus) == 2
        assert START not in src.commands and STOP not in src.commands


def test_pipeline_csv_network_and_restart(tmp_path):
    with Simulator(reorder=True, duplicate=[12]) as src:
        cfg = config(tmp_path, src)
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.publisher)
            with socket.create_connection(("127.0.0.1", cfg["network"]["port"])) as client:
                client.settimeout(5)
                client.sendall(command_frame(cfg["network"]["id_code"], START))
                framer, frames = Framer(), []
                while not any(f.kind == 0 for f in frames):
                    frames.extend(framer.feed(client.recv(65536)))
                assert any(f.kind == 3 for f in frames)
                # Downstream STOP is isolated; files continue and input is unaffected.
                client.sendall(command_frame(cfg["network"]["id_code"], STOP))
                before = run.counters["forecast_completed"]
                wait_for(lambda: run.counters["forecast_completed"] > before)
            with pytest.raises(RuntimeError, match="Already"):
                run.start()
        finally:
            run.stop()
        assert STOP not in src.commands and START not in src.commands
        files = list(run.run_directory.glob("predictions_*.csv"))
        assert files
        rows = [r for path in files for r in csv.DictReader(path.open(encoding="utf-8"))]
        assert all(r["target_time_exact"] != r["t_ref_exact"] for r in rows)
        assert all(r["generated_at"] != r["t_ref"] for r in rows)
        assert (run.run_directory / "output_mapping.json").is_file()
        assert len({r["forecast_id"] for r in rows}) >= 2
        first_id = run.run_id
        run.start()
        wait_for(lambda: run.counters["forecast_completed"] >= 1)
        run.stop()
        assert first_id != run.run_id
        assert not run.snapshot()["running"]


def test_slow_model_latest_only_reconnect_and_configuration_change(tmp_path):
    with Simulator() as src:
        run = Runtime(config(tmp_path, src), _model_factory=SlowModel)
        try:
            run.start()
            wait_for(lambda: run.counters["forecast_completed"] >= 1)
            assert run.counters["pending_replaced"] > 0
            src.disconnect_clients()
            wait_for(lambda: src.connections >= 2)
            wait_for(lambda: run.counters["forecast_completed"] >= 2)
            src.cfg = replace(src.cfg, pmus=(replace(src.cfg.pmus[0], config_count=2), src.cfg.pmus[1]))
            wait_for(lambda: run.snapshot()["phase"] == "failed")
        finally:
            run.stop()
        assert "configuration changed" in run.state["last_error"]
        assert STOP not in src.commands


def test_hung_model_is_actually_terminated(tmp_path):
    with Simulator() as src:
        cfg = config(tmp_path, src)
        cfg["model"]["timeout_seconds"] = 0.2
        run = Runtime(cfg, _model_factory=HungModel)
        try:
            run.start()
            wait_for(lambda: run.counters["worker_terminated"] >= 1)
            assert run.counters["forecast_completed"] == 0
        finally:
            run.stop()
        assert run.worker.process is None


def test_file_failure_does_not_stop_forecasts(tmp_path):
    with Simulator() as src:
        cfg = config(tmp_path, src)
        blocked = tmp_path / "is_file"
        blocked.write_text("Not a directory", encoding="utf-8")
        cfg["files"]["directory"] = str(blocked)
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.counters["forecast_completed"] >= 2)
            assert run.writer.failed and run.counters["file_error"] >= 1
            assert run.publisher is not None
        finally:
            run.stop()
