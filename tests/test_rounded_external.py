"""Real model and independent wire acceptance with millisecond-rounded online time."""

import csv
import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from pmu_fm.runtime import Runtime
from tests.helpers import REAL_MODELS, wait_for
from tests.test_runtime import config, free_port


@pytest.mark.external
@pytest.mark.real_model
@pytest.mark.parametrize(
    "backend,variable", REAL_MODELS
)
def test_real_rounded_pmu55_csv_and_receiver(tmp_path, backend, variable):
    path = os.environ.get(variable)
    if not path or not (Path(path) / "model.safetensors").is_file():
        pytest.skip(f"{variable} missing: rounded-source real E2E NOT verified")
    pytest.importorskip("pySynphasor", reason="Independent receiver missing; rounded-source E2E NOT verified")
    from tests.external_receiver import ExternalReceiver
    from tests.external_source import ExternalSource

    with ExternalSource(target_port=free_port(), rate=30, pmu55=True, millisecond_time=True) as src:
        cfg = config(tmp_path, src)
        cfg["source"].update(
            mode="listen",
            id_code=55,
            channels=["55/freq", "55/phasor/0/magnitude"],
            timestamp_tolerance_seconds=0.001,
        )
        cfg["model"].update(
            backend=backend,
            local_dir=path,
            revision=Path(path).name,
            context_seconds=4.0,
            horizon_seconds=1.0,
            interval_seconds=1.0,
        )
        cfg["network"]["leads_seconds"] = [1.0]
        cfg["files"].update(rows_per_segment=1000, debug_windows=True)
        run = Runtime(cfg)
        receiver = None
        try:
            run.start()
            wait_for(lambda: run.counters["model_ready"] or run.state["phase"] == "failed", timeout=90)
            assert run.counters["model_ready"], run.snapshot()
            receiver = ExternalReceiver("127.0.0.1", cfg["network"]["port"], cfg["network"]["id_code"])
            receiver.command(5)
            receiver.command(2)
            stamp, blocks = receiver.forecast()
            assert len(blocks) == 1 and len(blocks[0].analogs) == 2
            assert blocks[0].data_modified == 1
            assert 59.96 < blocks[0].freq < 60.04
            assert run.latest_result.timestamp_tolerance == Fraction(1, 1000)
            assert run.latest_result.max_timestamp_deviation <= Fraction(1, 1000)
        finally:
            if receiver:
                receiver.close()
            run.stop()
        assert 1 not in src.commands and 2 not in src.commands
        assert not run.writer.failed and run.writer.dropped == 0
        rows = [
            row
            for path in run.run_directory.glob("predictions_*.csv")
            for row in csv.DictReader(path.open(encoding="utf-8"))
        ]
        matching = [r for r in rows if Fraction(r["t_ref_exact"]) == stamp]
        assert len(matching) == 60
        assert len({r["forecast_id"] for r in matching}) == 1
        assert {r["source_timestamp_tolerance_seconds"] for r in matching} == {"1/1000"}
        assert all(
            Fraction(r["source_max_timestamp_deviation_seconds"]) <= Fraction(1, 1000) for r in matching
        )
        mapping = json.loads((run.run_directory / "output_mapping.json").read_text())
        for index, entry in enumerate(mapping["channels"]):
            row = next(
                r for r in matching if r["channel"] == entry["channel"] and int(r["horizon_step"]) == 30
            )
            assert Fraction(row["target_time_exact"]) == stamp + 1
            assert blocks[0].analogs[index] == pytest.approx(float(row["value"]), rel=1e-6, abs=1e-6)
        destination = os.environ.get("PMU_FM_EVIDENCE_DIR")
        if destination:
            import shutil

            dest = Path(destination) / f"rounded-{backend}" / run.run_id
            shutil.copytree(run.run_directory, dest)
