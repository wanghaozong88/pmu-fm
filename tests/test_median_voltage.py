"""Median example contract and real online inference with an independent receiver."""

import csv
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import pytest

from pmu_fm.buffer import Window
from pmu_fm.preprocessing import input_frame, load_script, preprocess
from pmu_fm.runtime import Runtime
from tests.helpers import REAL_MODELS, wait_for
from tests.test_runtime import config, free_port, sample


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "median_voltage.py"
VOLTAGE = "55/phasor/0/magnitude"
FILTERED = VOLTAGE + "_median5"
SELECTED = ["55/dfreq", "55/freq", "55/phasor/0/angle", "55/phasor/0/imag", VOLTAGE, "55/phasor/0/real"]


def window_context(values):
    info = {
        VOLTAGE: {"id": VOLTAGE, "name": "Phasor_V magnitude", "unit": "V", "source_pmu": 55},
        "55/freq": {"name": "Frequency", "unit": "Hz", "source_pmu": 55},
        "55/phasor/1/magnitude": {"name": "Current magnitude", "unit": "A", "source_pmu": 55},
    }
    rows = tuple(
        replace(
            sample(10 + Fraction(i, 10)),
            values={VOLTAGE: v, "55/freq": 60.0, "55/phasor/1/magnitude": 2.0},
            statuses={55: 0},
        )
        for i, v in enumerate(values)
    )
    return Window(rows, tuple(info), Fraction(1, 10)), info


def test_median_spike_causality_and_preserved_originals():
    values = [120, 122, 900, 121, 119, 123, 121, 120]
    window, info = window_context(values)
    original, source_meta = input_frame(window, info)
    module, _ = load_script(SCRIPT)
    plan = {"context": len(values), "horizon": 2, "dt": window.interval}
    data, metadata = preprocess(window, info, module, plan, 32)
    assert list(data.columns) == [*original.columns, FILTERED]
    pd.testing.assert_frame_equal(data[original.columns], original)
    assert data[FILTERED].tolist() == [120, 121, 122, 121.5, 121, 122, 121, 121]
    assert metadata["wire_time"] == source_meta["wire_time"]
    assert {c: metadata["channels"][c] for c in original.columns} == info
    assert metadata["channels"][FILTERED]["unit"] == "V"
    assert metadata["channels"][FILTERED]["source_channel"] == VOLTAGE
    assert metadata["channels"][FILTERED]["source_pmu"] == 55
    assert metadata["channels"][FILTERED]["transform"]
    # Changing later samples must not affect earlier filtered values.
    changed = original.copy()
    changed.loc[changed.index[4]:, VOLTAGE] = 9999.0
    other, _ = module.preprocess(changed, deepcopy(source_meta))
    pd.testing.assert_series_equal(data[FILTERED].iloc[:4], other[FILTERED].iloc[:4])
    # Repeated, overlapping jobs cannot contaminate a window with mutable filter state.
    again, _ = preprocess(window, info, module, plan, 32)
    pd.testing.assert_frame_equal(data, again)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_median_does_not_hide_missing_or_nonfinite(bad_value):
    window, info = window_context([120, bad_value, 121])
    data, metadata = input_frame(window, info)
    module, _ = load_script(SCRIPT)
    with pytest.raises(ValueError, match="finite input"):
        module.preprocess(data, metadata)


def test_median_requires_selected_voltage():
    window, info = window_context([120, 121])
    data, metadata = input_frame(window, info)
    data = data.drop(columns=VOLTAGE)
    del metadata["channels"][VOLTAGE]
    module, _ = load_script(SCRIPT)
    with pytest.raises(ValueError, match="Select at least one voltage"):
        module.preprocess(data, metadata)


@pytest.mark.external
@pytest.mark.real_model
@pytest.mark.parametrize(
    "backend,variable", REAL_MODELS
)
def test_real_median_model_input_csv_and_external_receiver(tmp_path, backend, variable):
    path = os.environ.get(variable)
    if not path or not (Path(path) / "model.safetensors").is_file():
        pytest.skip(f"{variable} missing: real median-filter E2E NOT verified")
    pytest.importorskip("pySynphasor", reason="Independent receiver missing; median E2E NOT verified")
    from tests.external_receiver import ExternalReceiver
    from tests.external_source import ExternalSource

    class SpikyVoltageSource(ExternalSource):
        def data(self, number, start):
            import struct

            # Independent wire encoder; fresh online synthetic spikes, only in this test.
            magnitude = 900.0 if number % 11 == 5 else 120.0 + number % 3
            payload = struct.pack(">Hffff", 0, magnitude, 0.1, 60.0, 0.01)
            return self.frame(0, payload, start + Fraction(number, self.rate))

    with SpikyVoltageSource(target_port=free_port(), rate=30, pmu55=True, millisecond_time=True) as src:
        cfg = config(tmp_path, src)
        cfg["source"].update(
            mode="listen", id_code=55, channels=SELECTED, timestamp_tolerance_seconds=0.001
        )
        cfg["model"].update(
            backend=backend, local_dir=path, revision=Path(path).name,
            context_seconds=4.0, horizon_seconds=1.0, interval_seconds=1.0,
        )
        cfg["script"]["path"] = str(SCRIPT)
        cfg["network"]["leads_seconds"] = [1.0]
        cfg["files"].update(rows_per_segment=1000, debug_windows=True)
        run, receiver = Runtime(cfg), None
        try:
            run.start()
            wait_for(lambda: run.counters["model_ready"] or run.state["phase"] == "failed", timeout=90)
            assert run.counters["model_ready"], run.snapshot()
            receiver = ExternalReceiver("127.0.0.1", cfg["network"]["port"], cfg["network"]["id_code"])
            receiver.command(5)
            receiver.command(2)
            stamp, blocks = receiver.forecast()
            assert len(blocks) == 1 and len(blocks[0].analogs) == 7
            assert blocks[0].data_modified == 1 and blocks[0].freq == 60.0
        finally:
            if receiver:
                receiver.close()
            run.stop()
        assert 1 not in src.commands and 2 not in src.commands
        assert not run.writer.failed and run.writer.dropped == 0
        rows = [
            row for file in run.run_directory.glob("predictions_*.csv")
            for row in csv.DictReader(file.open(encoding="utf-8"))
            if Fraction(row["t_ref_exact"]) == stamp
        ]
        assert len(rows) == 210 and len({r["forecast_id"] for r in rows}) == 1
        trace = json.loads((run.run_directory / f"window_{rows[0]['forecast_id']}.json").read_text())
        inputs, raw = trace["model_input"], trace["raw_selected"]
        assert inputs["columns"] == [*SELECTED, FILTERED] and raw["columns"] == SELECTED
        values = np.array(inputs["values"])
        assert values.shape == (120, 7)
        np.testing.assert_array_equal(values[:, :6], raw["values"])
        assert inputs["timestamps_ns"] == [round(Fraction(t) * 10**9) for t in raw["timestamps_exact"]]
        voltage = values[:, 4].tolist()
        expected = [median(voltage[max(0, i - 4):i + 1]) for i in range(len(voltage))]
        np.testing.assert_array_equal(values[:, 6], expected)
        spikes = values[4:, 4] > 500
        assert spikes.any() and (values[4:, 6][spikes] < 130).all()
        mapping = json.loads((run.run_directory / "output_mapping.json").read_text())
        assert [entry["channel"] for entry in mapping["channels"]] == [*SELECTED, FILTERED]
        for index, entry in enumerate(mapping["channels"]):
            row = next(r for r in rows if r["channel"] == entry["channel"] and int(r["horizon_step"]) == 30)
            assert Fraction(row["target_time_exact"]) == stamp + 1
            assert blocks[0].analogs[index] == pytest.approx(float(row["value"]), rel=1e-6, abs=1e-6)
        model = json.loads((run.run_directory / "model.json").read_text())
        assert model["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        destination = os.environ.get("PMU_FM_EVIDENCE_DIR")
        if destination:
            import shutil

            shutil.copytree(run.run_directory, Path(destination) / f"median-{backend}" / run.run_id)
