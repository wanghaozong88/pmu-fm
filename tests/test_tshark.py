"""Optional second, independent verification on captured loopback TCP traffic."""

import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pmu_fm.runtime import Runtime
from tests.helpers import FakeModel, Simulator, wait_for
from tests.test_runtime import config


@pytest.mark.external
@pytest.mark.parametrize("version", [1, 2])
def test_live_wireshark_capture(tmp_path, version):
    if os.environ.get("PMU_FM_CAPTURE") != "1":
        pytest.skip(
            "Live Wireshark capture not enabled: set PMU_FM_CAPTURE=1 with loopback capture permission"
        )
    tshark = os.environ.get("PMU_FM_TSHARK") or shutil.which("tshark")
    if not tshark and Path("C:/Program Files/Wireshark/tshark.exe").exists():
        tshark = "C:/Program Files/Wireshark/tshark.exe"
    if not tshark:
        pytest.skip("TShark not installed; independent capture verification NOT performed")
    pytest.importorskip("pySynphasor")
    from tests.external_receiver import ExternalReceiver

    interface = os.environ.get(
        "PMU_FM_CAPTURE_INTERFACE", r"\Device\NPF_Loopback" if os.name == "nt" else "lo"
    )
    capture = tmp_path / "output.pcapng"
    with Simulator() as src:
        cfg = config(tmp_path, src)
        cfg["network"]["frame_version"] = version
        cfg["source"]["channels"] = ["10/freq", "20/freq"]
        run = Runtime(cfg, _model_factory=FakeModel)
        proc = receiver = None
        try:
            run.start()
            wait_for(lambda: run.counters["model_ready"])
            args = [
                tshark,
                "-i",
                interface,
                "-f",
                f"tcp port {cfg['network']['port']}",
                "-a",
                "duration:4",
                "-w",
                str(capture),
            ]
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            time.sleep(0.8)
            receiver = ExternalReceiver(
                "127.0.0.1", cfg["network"]["port"], cfg["network"]["id_code"], version
            )
            receiver.command(5)
            receiver.command(2)
            for _ in range(3):
                receiver.forecast()
            receiver.command(1)
            out, err = proc.communicate(timeout=10)
            assert proc.returncode == 0, err.decode(errors="replace")
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                proc.communicate(timeout=5)
            if receiver:
                receiver.close()
            run.stop()
    decoded = subprocess.run(
        [
            tshark,
            "-r",
            str(capture),
            "-d",
            f"tcp.port=={cfg['network']['port']},synphasor",
            "-Y",
            "synphasor",
            "-T",
            "pdml",
        ],
        capture_output=True,
        check=True,
    )
    tree = ET.fromstring(decoded.stdout)
    fields = list(tree.iter("field"))
    checksums = [f for f in fields if f.attrib.get("name") == "synphasor.checksum.status"]
    assert checksums and all(f.attrib.get("show") == "1" for f in checksums)
    versions = [f.attrib.get("show") for f in fields if f.attrib.get("name") == "synphasor.version"]
    assert versions and all(v == str(version) for v in versions)
    assert not any(f.attrib.get("name") == "_ws.malformed" for f in fields)
    analogs = [f.attrib for f in fields if f.attrib.get("name") == "synphasor.analog_value"]
    assert len(analogs) >= 6
    assert any(
        f.attrib.get("name") == "synphasor.data.data_modified" and f.attrib.get("value") == "1"
        for f in fields
    )
    frequencies = [
        float(f.attrib["show"]) for f in fields if f.attrib.get("name") == "synphasor.actual_frequency_value"
    ]
    assert any(abs(f - 50.01) < 0.0001 for f in frequencies)
    assert any(59.96 < f < 60.04 for f in frequencies)
    assert any(
        f.attrib.get("name") == "synphasor.conf.numpmu" and f.attrib.get("show") == "2" for f in fields
    )
    destination = os.environ.get("PMU_FM_EVIDENCE_DIR")
    if destination:
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(capture, dest / f"wireshark-output-v{version}.pcapng")
        (dest / f"wireshark-output-v{version}.pdml").write_bytes(decoded.stdout)
