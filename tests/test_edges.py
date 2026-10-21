from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import time

import pytest

from pmu_fm.config import DEFAULTS, ConfigError, report_rate, sample_plan
from pmu_fm.outputs import ForecastMapping
from pmu_fm.protocol import Configuration, PMU, data_frame, decode, parse_data
from pmu_fm.records import Forecast
from pmu_fm.runtime import Runtime
from tests.helpers import FakeModel, Simulator, wait_for
from tests.test_runtime import config, sample


def test_subnanosecond_wire_time_preserved_and_quality_rejected():
    cfg = Configuration(1, 16_000_001, (PMU("Clock", 10, 8),), 60)
    exact = Fraction(1_700_000_000) + Fraction(1, cfg.time_base)
    s = parse_data(decode(data_frame(cfg, exact, [{"stat": 0, "freq": 60, "dfreq": 0}])), cfg)
    assert s.timestamp == exact
    for q in (1, 15, 0x20, 0x10):
        assert not replace(s, time_quality=q).valid([10])
    for stat in (0x8000, 0x4000, 0x2000, 0x01C0):
        assert not replace(s, statuses={10: stat}).valid([10])


def test_rate_conversion_and_model_limits():
    assert report_rate(0.2) == 5 and report_rate(3) == -3
    with pytest.raises(ConfigError):
        report_rate(0.3)
    cfg = deepcopy(DEFAULTS)
    cfg["model"]["context_seconds"] = 500
    with pytest.raises(ConfigError, match="limits"):
        sample_plan(cfg, Fraction(1, 60))
    cfg["model"].update(backend="timesfm25", context_seconds=816, horizon_seconds=2)
    with pytest.raises(ConfigError, match="16384"):
        sample_plan(cfg, Fraction(1, 20))


def test_no_fabricated_frequency_and_no_mapping_mutation():
    cfg = deepcopy(DEFAULTS)
    cfg["network"]["leads_seconds"] = [1.0]
    source = Configuration(1, 1_000_000, (PMU("Source", 10, 8),), 1)
    result = Forecast(
        "run",
        "id",
        FakeModel({}).load(),
        sample(100),
        Fraction(98),
        Fraction(1),
        time.time_ns(),
        {"10/freq": {"name": "F", "unit": "Hz", "source_pmu": 10}},
        [[60.1]],
        "valid",
        0.1,
    )
    mapping = ForecastMapping(cfg, source, result, {"leads": [1]})
    s = parse_data(decode(mapping.encode(result)), mapping.cfg)
    assert s.timestamp == 100 and s.values["50001/freq"] == 60
    assert s.values["50001/analog/0"] == pytest.approx(60.1)
    result.reference = replace(result.reference, values={"10/freq": 60})
    with pytest.raises(ValueError, match="FREQ/DFREQ"):
        mapping.encode(result)
    result.channels = {"other": {"name": "X", "unit": "Hz", "source_pmu": 10}}
    with pytest.raises(ConfigError, match="structure changed"):
        mapping.encode(result)


def test_source_silence_timeout_and_stop_while_disconnected(tmp_path):
    with Simulator(drop=range(200)) as silent:
        cfg = config(tmp_path, silent)
        cfg["source"]["timeout_seconds"] = 0.2
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.counters["input_error"] >= 1)
            wait_for(lambda: silent.connections >= 2)
            assert run.counters["forecast_completed"] == 0
        finally:
            run.stop()
    with Simulator() as src:
        cfg = config(tmp_path, src)
    # Endpoint no longer listening: receiver must reconnect without blocking stop.
    run = Runtime(cfg, _model_factory=FakeModel)
    run.start()
    wait_for(lambda: run.counters["input_error"])
    started = time.monotonic()
    run.stop()
    assert time.monotonic() - started < 4
    assert not run.receiver.thread.is_alive()


def test_config_change_flag_and_resume_same_cfg(tmp_path):
    # An explicit source CFG-change bit requests fresh configuration and drops
    # in-flight history, rather than inferring through a pending mapping.
    with Simulator() as src:
        cfg = config(tmp_path, src)
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.counters["forecast_completed"])
            # Send a flagged frame on the actual input TCP connection.
            for conn, _ in src.clients:
                conn.sendall(
                    data_frame(
                        src.cfg,
                        Fraction(time.time_ns(), 10**9),
                        [
                            {"stat": 0x0400, "freq": 60, "dfreq": 0, "analog": [1]},
                            {"stat": 0, "freq": 50, "dfreq": 0},
                        ],
                    )
                )
            wait_for(lambda: run.counters["config_pending"])
            assert src.commands.count(5) >= 2
        finally:
            run.stop()
        assert 1 not in src.commands
