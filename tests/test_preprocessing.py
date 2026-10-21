from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from pmu_fm.buffer import History, Window
from pmu_fm.config import DEFAULTS, ConfigError, normalize, sample_plan
from pmu_fm.preprocessing import input_frame, inverse, load_script, preprocess
from pmu_fm.runtime import Runtime
from tests.helpers import FakeModel, Simulator, wait_for
from tests.test_runtime import config, sample


def context():
    w = Window(
        tuple(sample(Fraction(10) + Fraction(i, 10), 60 + i / 100) for i in range(4)),
        ("10/freq",),
        Fraction(1, 10),
    )
    info = {"10/freq": {"name": "Frequency", "unit": "Hz", "source_pmu": 10, "nominal_hz": 60}}
    plan = {"context": 4, "horizon": 2, "dt": Fraction(1, 10)}
    return w, info, plan


def test_identity_and_frequency_deviation():
    w, info, plan = context()
    for path in ("examples/identity.py", "examples/frequency_deviation.py"):
        module, digest = load_script(path)
        data, metadata = preprocess(w, info, module, plan, 32)
        assert len(digest) == 64 and len(data) == 4
        assert str(data.index.tz) == "UTC"
        if "deviation" in path:
            assert data.iloc[-1, 0] == pytest.approx(0.03)
            assert metadata["channels"][data.columns[0]]["unit"] == "Hz deviation"
    assert w.reference.values["10/freq"] == 60.03


@pytest.mark.parametrize(
    "body,match",
    [
        ("data.iloc[0, 0] = float('nan')", "NaN"),
        ("data.index = data.index.tz_localize(None)", "timezone"),
        ("data = data.iloc[::-1]", "increasing"),
        ("data.index = data.index + __import__('pandas').Timedelta(seconds=1)", "t_ref"),
        ("metadata['channels'] = {}", "match"),
        ("data.iloc[:, 0] *= 2", "transform"),
    ],
)
def test_bad_script_contract(tmp_path, body, match):
    path = tmp_path / "bad.py"
    path.write_text(
        f"def preprocess(data, metadata):\n    {body}\n    return data, metadata\n", encoding="utf-8"
    )
    module, _ = load_script(path)
    w, info, plan = context()
    with pytest.raises(ConfigError, match=match):
        preprocess(w, info, module, plan, 32)


def test_resampling_and_inverse(tmp_path):
    w, info, _ = context()
    # End-aligned decimation preserves the reference sample and changes interval explicitly.
    script = tmp_path / "resample.py"
    script.write_text(
        """def preprocess(data, metadata):
    metadata['interval_seconds'] = '1/5'
    metadata['channels']['10/freq']['transform'] = 'End-aligned decimation by 2'
    return data.iloc[1::2], metadata

def inverse_transform(predictions, metadata):
    return predictions, metadata
""",
        encoding="utf-8",
    )
    module, _ = load_script(script)
    plan = {"context": 2, "horizon": 2, "dt": Fraction(1, 5)}
    data, meta = preprocess(w, info, module, plan, 32)
    assert len(data) == 2
    future = pd.to_datetime([10_500_000_000, 10_700_000_000], unit="ns", utc=True)
    pred, _ = inverse(np.array([[1.0], [2.0]]), future, meta, module, plan, 32)
    assert pred.iloc[-1, 0] == 2


@pytest.mark.parametrize(
    "body,expected",
    [
        ("raise RuntimeError('script test error')", "inference_error"),
        ("while True: pass", "worker_terminated"),
    ],
)
def test_script_error_and_killable_timeout(tmp_path, body, expected):
    script = tmp_path / "script.py"
    script.write_text(f"def preprocess(data, metadata):\n    {body}\n", encoding="utf-8")
    with Simulator() as src:
        cfg = config(tmp_path, src)
        # This deadline also covers cold pandas imports in the spawned worker.
        # Keep it bounded, but allow initialization so we test the script itself.
        cfg["script"].update(path=str(script), timeout_seconds=2.0)
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.counters[expected] >= 1)
            assert run.counters["model_ready"] >= 1
            if expected == "worker_terminated":
                assert "preprocessing exceeded" in run.snapshot()["last_error"]
            else:
                assert "script test error" in run.snapshot()["last_error"]
            assert run.counters["forecast_completed"] == 0
            assert run.counters["samples"] > 0
        finally:
            run.stop()


def test_dataframe_copy_cannot_modify_history():
    w, info, _ = context()
    data, meta = input_frame(w, deepcopy(info))
    data.iloc[0, 0] = -1
    meta["channels"]["10/freq"]["unit"] = "other"
    assert w.samples[0].values["10/freq"] == 60
    assert info["10/freq"]["unit"] == "Hz"


def test_explicit_bounded_missing_repair(tmp_path):
    h = History(["10/freq"], Fraction(1, 10), 4, 0.1, 0, max_missing=1)
    for t in ("10", "10.1", "10.3"):
        h.add(sample(t))
    windows = [w for w, _ in h.ready() if w]
    assert windows and windows[0].samples[2] is None
    script = tmp_path / "repair.py"
    script.write_text(
        """def preprocess(data, metadata):
    metadata['channels']['10/freq']['transform'] = 'Linear repair of at most one historical gap'
    metadata['quality'] = 'One historical missing value interpolated within already available data'
    return data.interpolate(limit=1), metadata
""",
        encoding="utf-8",
    )
    module, _ = load_script(script)
    _, info, plan = context()
    frame, meta = preprocess(windows[0], info, module, plan, 32)
    assert np.isfinite(frame.to_numpy()).all()
    assert meta["wire_time"][2] is None


def test_exact_60_to_30_hz_resampling_configuration():
    cfg = deepcopy(DEFAULTS)
    cfg["model"]["input_interval_seconds"] = "1/30"
    cfg["script"]["path"] = "trusted_script.py"
    plan = sample_plan(normalize(cfg), Fraction(1, 60))
    assert plan["dt"] == Fraction(1, 30)
    assert plan["context"] == 120 and plan["source_context"] == 240
    assert plan["horizon"] == 30
    cfg["model"]["input_interval_seconds"] = "1/0"
    with pytest.raises(ConfigError):
        normalize(cfg)
