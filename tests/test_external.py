import csv
import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from pmu_fm.models import LocalModel
from pmu_fm.runtime import Runtime
from tests.helpers import REAL_MODELS, FakeModel, wait_for
from tests.test_runtime import config, free_port


@pytest.mark.external
@pytest.mark.parametrize("mode,version", [("connect", 1), ("connect", 2), ("listen", 1), ("listen", 2)])
def test_independent_receiver_multiblock_csv(tmp_path, mode, version):
    pytest.importorskip(
        "pySynphasor", reason="Independent receiver unavailable; interoperability NOT verified"
    )
    verify_pipeline(tmp_path, "fake", None, mode, version)


@pytest.mark.external
@pytest.mark.real_model
@pytest.mark.parametrize("mode,version", [("connect", 1), ("connect", 2), ("listen", 1), ("listen", 2)])
@pytest.mark.parametrize(
    "backend,variable", REAL_MODELS
)
def test_real_model_external_receiver_csv(tmp_path, backend, variable, mode, version):
    path = os.environ.get(variable)
    if not path or not (Path(path) / "model.safetensors").is_file():
        pytest.skip(f"{variable} missing: real model + external receiver E2E NOT verified")
    pytest.importorskip("pySynphasor", reason="Independent receiver unavailable; real E2E NOT verified")
    verify_pipeline(tmp_path, backend, path, mode, version)


def verify_pipeline(tmp_path, backend, model_path, mode, version):
    from tests.external_receiver import ExternalReceiver
    from tests.external_source import ExternalSource

    with ExternalSource(
        target_port=free_port() if mode == "listen" else None, version=version,
        accepted_command_version=version, data_before_config=not bool(model_path),
    ) as src:
        cfg = config(tmp_path, src)
        cfg["source"].update(mode=mode, command_version="auto" if model_path else version)
        cfg["network"]["frame_version"] = version
        cfg["source"]["channels"] = ["10/freq", "20/freq"]
        cfg["model"].update(context_seconds=1.6, horizon_seconds=0.2, interval_seconds=1.0)
        cfg["network"]["leads_seconds"] = [0.1, 0.2]
        cfg["files"]["rows_per_segment"] = 1000
        if model_path:
            cfg["model"].update(backend=backend, local_dir=model_path, revision=Path(model_path).name)
            cfg["script"]["path"] = str(Path("examples/frequency_deviation.py").resolve())
        run = Runtime(cfg, _model_factory=LocalModel if model_path else FakeModel)
        receiver = None
        try:
            run.start()
            wait_for(lambda: run.publisher or run.snapshot()["phase"] == "failed")
            assert run.publisher, run.snapshot()
            # Wait until real model loaded; no need to let receiver's 5s timeout cover model startup.
            wait_for(lambda: run.counters["model_ready"] or run.snapshot()["phase"] == "failed", timeout=90)
            assert run.counters["model_ready"], run.snapshot()
            # Opposite command version also accepted; output has an explicit stable version.
            receiver = ExternalReceiver(
                "127.0.0.1",
                cfg["network"]["port"],
                cfg["network"]["id_code"],
                version=version,
                command_version=3 - version,
            )
            receiver.command(3)
            header = receiver.receive()
            assert header.type == 1
            receiver.command(5)
            receiver.command(2)
            stamp, blocks = receiver.forecast()
            assert len(blocks) == 2
            assert receiver.cfg.data_rate == 1 and receiver.cfg.num_pmu == 2
            assert [p.idcode for p in receiver.cfg.pmu] == [50001, 50002]
            assert all(p.phnmr == 0 and p.annmr == 2 and p.dgnmr == 0 for p in receiver.cfg.pmu)
            assert blocks[0].freq == pytest.approx(60, abs=0.031)
            assert blocks[1].freq == pytest.approx(50.01, abs=0.00001)
            assert blocks[1].dfreq == pytest.approx(-0.02)
            assert all(b.data_modified == 1 and b.data_error == 0 for b in blocks)
            receiver.command(1)
        finally:
            if receiver:
                receiver.close()
            run.stop()
        assert 1 not in src.commands and 2 not in src.commands
        assert src.command_versions == ([1, 2] if model_path and version == 2 else [version])
        assert src.accepted_commands == [(version, 5)]
        rows = [
            r
            for path in run.run_directory.glob("predictions_*.csv")
            for r in csv.DictReader(path.open(encoding="utf-8"))
        ]
        matching = [r for r in rows if Fraction(r["t_ref_exact"]) == stamp]
        assert len(matching) == 8
        assert {int(r["source_frame_version"]) for r in matching} == {version}
        forecast_ids = {r["forecast_id"] for r in matching}
        assert len(forecast_ids) == 1
        mapping = json.loads((run.run_directory / "output_mapping.json").read_text(encoding="utf-8"))
        for entry in mapping["channels"]:
            row = next(
                r
                for r in matching
                if r["channel"] == entry["channel"] and int(r["horizon_step"]) == entry["horizon_step"]
            )
            block = blocks[entry["virtual_pmu"] - 50001]
            ai = [e for e in mapping["channels"] if e["virtual_pmu"] == entry["virtual_pmu"]].index(entry)
            assert block.analogs[ai] == pytest.approx(float(row["value"]), rel=1e-6, abs=1e-6)
            assert Fraction(row["target_time_exact"]) == stamp + Fraction(entry["horizon_seconds"])
        evidence = {
            "backend": backend,
            "input_mode": mode,
            "input_frame_version": version,
            "input_command_mode": cfg["source"]["command_version"],
            "input_command_versions_tried": src.command_versions,
            "output_frame_version": version,
            "source": "Independent literal CFG-2/DATA encoder + pySynphasor CRC",
            "receiver": "pySynphasor 0.1.0 / Scapy 2.7.0",
            "model": run.latest_result.model,
            "preprocessing": "frequency_deviation.py" if model_path else "identity",
            "forecast_id": next(iter(forecast_ids)),
            "t_ref_exact": str(stamp),
            "pmu_blocks": len(blocks),
            "csv_rows_matched": len(matching),
            "runtime": run.snapshot(),
        }
        evidence_path = os.environ.get("PMU_FM_EVIDENCE_DIR")
        if evidence_path:
            dest = Path(evidence_path)
            dest.mkdir(parents=True, exist_ok=True)
            import shutil

            artifact = dest / f"{backend}-{mode}-v{version}"
            artifact.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run.run_directory, artifact / run.run_id)
            evidence["preserved_run"] = str((artifact / run.run_id).resolve())
            (artifact / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


@pytest.mark.external
@pytest.mark.parametrize("version", [1, 2])
def test_independent_pmu55_listener_30fps(tmp_path, version):
    pytest.importorskip("pySynphasor")
    from pmu_fm.network import inspect_source
    from tests.external_source import ExternalSource

    with ExternalSource(target_port=free_port(), version=version, rate=30, pmu55=True) as src:
        cfg = config(tmp_path, src)
        cfg["source"].update(mode="listen", id_code=55, channels=["55/freq", "55/phasor/0/magnitude"])
        cfg["model"].update(context_seconds=4.0, horizon_seconds=1.0, interval_seconds=1.0)
        cfg["network"].update(enabled=False, leads_seconds=[1.0])
        cfg["files"]["rows_per_segment"] = 1000
        source = inspect_source(cfg["source"])
        assert source.id_code == 55 and source.data_rate == 30 and source.version == version
        assert len(src.configuration()) == 74
        assert source.pmus[0].phasor_names == ("Phasor_V",)
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.counters["forecast_completed"], timeout=15)
            assert run.latest_result.reference.values["55/phasor/0/magnitude"] == 120
            assert run.latest_result.reference.version == version
            assert len(run.latest_result.values) == 30
            assert run.snapshot()["window_samples"] == 120
        finally:
            run.stop()
        assert 1 not in src.commands and 2 not in src.commands
