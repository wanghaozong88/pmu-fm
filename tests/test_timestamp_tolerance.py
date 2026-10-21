"""Rounded source time is explicit; true gaps and forecast-target shifts stay errors."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pmu_fm.buffer import History, Window
from pmu_fm.config import ConfigError, normalize, sample_plan, validate
from pmu_fm.preprocessing import input_frame, inverse, preprocess
from pmu_fm.protocol import Configuration, PMU, data_frame, decode, ns, parse_data


def rounded_samples(rate=30, count=120):
    cfg = Configuration(55, 1_000_000, (PMU("PMU55", 55, 8),), rate)
    samples = []
    for i in range(count):
        stamp = Fraction(1_700_000_000) + Fraction(round(Fraction(i * 1000, rate)), 1000)
        raw = data_frame(cfg, stamp, [{"stat": 0, "freq": 60 + i / 1000, "dfreq": 0.01}])
        samples.append(parse_data(decode(raw), cfg))
    return samples


def history(samples, count, rate=30, tolerance=0.001):
    h = History(["55/freq"], Fraction(1, rate), count, 1, 0, timestamp_tolerance_seconds=tolerance)
    for s in samples:
        h.add(s)
    return h, h.ready()


@pytest.mark.parametrize("rate", [30, 60])
def test_millisecond_rounding_preserves_wire_times_and_values(rate):
    samples = rounded_samples(rate, rate * 4)
    _, strict = history(samples, len(samples), rate, tolerance=0)
    assert not any(w for w, _ in strict)
    h, results = history(samples, len(samples), rate)
    windows = [w for w, _ in results if w]
    assert len(windows) == 1
    window = windows[0]
    assert window.timestamps == tuple(s.timestamp for s in samples)
    assert [(s.soc, s.ticks, s.time_base) for s in window.samples] == [
        (s.soc, s.ticks, s.time_base) for s in samples
    ]
    assert len({s.timestamp for s in window.samples}) == len(samples)
    assert window.max_timestamp_deviation == Fraction(1, 1500)
    info = {"55/freq": {"source_pmu": 55, "name": "Frequency", "unit": "Hz"}}
    plan = {"context": len(samples), "dt": Fraction(1, rate), "horizon": rate}
    frame, metadata = preprocess(window, info, None, plan, 32)
    assert frame.index.asi8.tolist() == [ns(s.timestamp) for s in samples]
    assert frame.iloc[:, 0].tolist() == [s.values["55/freq"] for s in samples]
    assert metadata["source_timestamp_tolerance_seconds"] == "1/1000"
    assert metadata["source_max_timestamp_deviation_seconds"] == "1/1500"
    assert h.counters["missing_or_irregular"] == 0


@pytest.mark.parametrize("damage", ["missing", "too_far", "quality", "ambiguous", "extra_off_grid"])
def test_tolerance_keeps_real_errors_visible(damage):
    samples = rounded_samples()
    if damage == "missing":
        del samples[60]
    elif damage == "too_far":
        s = samples[60]
        samples[60] = replace(s, timestamp=s.timestamp + Fraction(2, 1000), ticks=s.ticks + 2000)
    elif damage == "quality":
        samples[60] = replace(samples[60], statuses={55: 0x2000})
    elif damage == "ambiguous":
        s = samples[60]
        samples.insert(61, replace(s, timestamp=s.timestamp + Fraction(1, 2000), ticks=s.ticks + 500))
    else:
        s = samples[60]
        samples.insert(61, replace(s, timestamp=s.timestamp + Fraction(5, 1000), ticks=s.ticks + 5000))
    _, results = history(samples, 120)
    assert not any(w for w, _ in results)
    reason = {
        "quality": "invalid_quality",
        "ambiguous": "ambiguous_timestamps",
        "extra_off_grid": "off_grid_samples",
        "too_far": "off_grid_samples",
    }.get(damage, "missing_or_irregular")
    assert any(r == reason for _, r in results)


def test_tolerance_does_not_reuse_duplicates_or_change_reference():
    samples = rounded_samples()
    h, results = history(samples[:60] + [samples[59]] + samples[60:], 120)
    assert h.counters["duplicate"] == 1
    window = next(w for w, _ in results if w)
    assert window.reference == samples[-1]
    assert len(set(window.timestamps)) == 120
    h.add(samples[-2])
    assert not h.ready()  # no repeated forecast for the same reference


def test_zero_legacy_default_and_bounded_configuration():
    cfg = normalize({})
    assert cfg["source"]["timestamp_tolerance_seconds"] == 0
    validate(cfg, False, False)
    cfg["source"]["timestamp_tolerance_seconds"] = 0.001
    assert sample_plan(cfg, Fraction(1, 30))["context"] == 120
    for value in (-1, float("nan"), float("inf")):
        bad = deepcopy(cfg)
        bad["source"]["timestamp_tolerance_seconds"] = value
        with pytest.raises(ConfigError):
            validate(bad, False, False)
    cfg["source"]["timestamp_tolerance_seconds"] = 0.01
    with pytest.raises(ConfigError, match="1/4"):
        sample_plan(cfg, Fraction(1, 30))
    cfg["source"]["timestamp_tolerance_seconds"] = 0.001
    cfg["model"]["input_interval_seconds"] = 0.004
    with pytest.raises(ConfigError, match="1/4"):
        sample_plan(cfg, Fraction(1, 30))


def test_tolerance_error_reports_setting_and_numeric_limit():
    cfg = normalize({"source": {"timestamp_tolerance_seconds": 0.01}})
    with pytest.raises(ConfigError) as error:
        sample_plan(cfg, Fraction(1, 30), 1_000_000)
    assert "source.timestamp_tolerance_seconds=0.01" in str(error.value)
    assert "required tolerance < 0.00833333333 s" in str(error.value)
    cfg["source"]["timestamp_tolerance_seconds"] = 0.001
    assert sample_plan(cfg, Fraction(1, 30), 1_000_000)["context"] == 120
    # The same preflight must reject a wire tick that is too coarse.
    with pytest.raises(ConfigError, match="effective tolerance 0.01 s"):
        sample_plan(cfg, Fraction(1, 30), 100)


def test_script_cannot_increase_tolerance_or_move_future_targets():
    window = Window(tuple(rounded_samples()), ("55/freq",), Fraction(1, 30), Fraction(1, 1000))
    info = {"55/freq": {"source_pmu": 55, "name": "Frequency", "unit": "Hz"}}
    plan = {"context": 120, "dt": Fraction(1, 30), "horizon": 30}

    def bad_preprocess(data, meta):
        stamps = data.index.asi8.copy()
        stamps[60] += 2_000_000
        data.index = pd.to_datetime(stamps, unit="ns", utc=True)
        meta["timestamp_tolerance_ns"] = 1_000_000_000
        return data, meta

    with pytest.raises(ConfigError, match="irregular"):
        preprocess(window, info, SimpleNamespace(preprocess=bad_preprocess), plan, 32)
    _, metadata = preprocess(window, info, None, plan, 32)
    future = pd.to_datetime(
        [ns(window.reference.timestamp + i * plan["dt"]) for i in range(1, 31)], unit="ns", utc=True
    )

    def bad_inverse(predictions, meta):
        stamps = predictions.index.asi8.copy()
        stamps[10] += 500_000  # inside source tolerance, outside future target tolerance
        predictions.index = pd.to_datetime(stamps, unit="ns", utc=True)
        meta["timestamp_tolerance_ns"] = 1_000_000_000
        return predictions, meta

    with pytest.raises(ConfigError, match="irregular"):
        inverse(np.ones((30, 1)), future, metadata, SimpleNamespace(inverse_transform=bad_inverse), plan, 32)
    predicted, _ = inverse(np.ones((30, 1)), future, metadata, None, plan, 32)
    assert predicted.index.equals(future)
    original, _ = input_frame(window, info)
    assert original.index[-1] == pd.Timestamp(ns(window.reference.timestamp), tz="UTC")
