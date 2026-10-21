import json
import queue
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from pmu_fm.cli import main
from pmu_fm.config import DEFAULTS, dumps
from pmu_fm.network import Receiver
from pmu_fm.protocol import START, ProtocolError, config_frame, decode, parse_config
from pmu_fm.test_config_input import load_test_config_frame
from tests.helpers import FakeModel, Simulator, wait_for
from tests.test_runtime import config, free_port


# Synthetic CFG-2: ID 30001, 10 Hz, one voltage phasor and 21 analogs; no device capture.
EXAMPLE = Path(__file__).parent / "fixtures" / "synthetic_cfg2.hex"


def test_synthetic_configuration():
    frame = load_test_config_frame(EXAMPLE, 30001)
    cfg = parse_config(frame)
    assert frame.version == 1
    assert cfg.time_base == 1_000_000 and cfg.data_rate == 10
    assert len(cfg.pmus) == 1
    assert cfg.pmus[0].station == "SYNTHETIC_TEST"
    assert cfg.pmus[0].analog_names == tuple(f"Analog{i:02}" for i in range(1, 22))
    assert cfg.pmus[0].phasor_names == ("Voltage",)
    assert len(cfg.pmus[0].analog_names) == 21
    assert cfg.pmus[0].nominal_hz == 50


@pytest.mark.parametrize("style", ["hyphen", "spaces", "compact"])
def test_hex_formats(tmp_path, style):
    raw = bytes.fromhex(EXAMPLE.read_text().replace("-", ""))
    text = {"hyphen": raw.hex("-"), "spaces": "\n".join(raw.hex(" ").split()), "compact": raw.hex()}[style]
    path = tmp_path / "frame.hex"
    path.write_text("\n" + text + "\n", encoding="utf-8-sig")
    assert load_test_config_frame(path, 30001) == decode(raw)


@pytest.mark.parametrize("case", ["crc", "truncated", "two", "text", "oversize", "id", "type"])
def test_invalid_import(tmp_path, case):
    raw = bytes.fromhex(EXAMPLE.read_text().replace("-", ""))
    if case == "crc":
        raw = raw[:-1] + bytes([raw[-1] ^ 1])
    elif case == "truncated":
        raw = raw[:-1]
    elif case == "two":
        raw *= 2
    elif case == "type":
        from pmu_fm.protocol import REQUEST_CFG2, command_frame

        raw = command_frame(30001, REQUEST_CFG2)
    text = "invalid" if case == "text" else "AA" * 140_000 if case == "oversize" else raw.hex()
    path = tmp_path / "frame.hex"
    path.write_text(text)
    with pytest.raises(ProtocolError):
        load_test_config_frame(path, 1 if case == "id" else 30001)


def test_cli_check_and_gui_boundary(tmp_path, capsys):
    cfg = deepcopy(DEFAULTS)
    cfg["source"]["id_code"] = 30001
    path = tmp_path / "settings.toml"
    path.write_text(dumps(cfg))
    assert main(["check", str(path), "--schema-only", "--test-config-frame", str(EXAMPLE)]) == 0
    output = capsys.readouterr().out
    result = json.loads(output[output.index("{"):])
    assert result["configuration_origin"] == "test_import"
    assert result["command_version"] is None
    assert len(result["channels"]) == 27
    with pytest.raises(SystemExit):
        main(["gui", "--test-config-frame", str(EXAMPLE)])
    with pytest.raises(SystemExit):
        main(["check", str(path), "--online", "--test-config-frame", str(EXAMPLE)])


@pytest.mark.parametrize("mode", ["connect", "listen"])
@pytest.mark.parametrize("command_version", ["auto", 1, 2])
@pytest.mark.parametrize("allow_start", [False, True])
def test_receiver_online_data_reconnect_and_config_updates(mode, command_version, allow_start):
    with Simulator(
        multi=False, version=2, require_start=allow_start,
        target_port=free_port() if mode == "listen" else None,
    ) as src:
        settings = DEFAULTS["source"] | {
            "port": src.port, "mode": mode, "command_version": command_version,
            "allow_start": allow_start, "reconnect_seconds": 0.05,
        }
        events, logs = queue.Queue(100), []
        receiver = Receiver(
            settings, events, lambda k, _m, **d: logs.append((k, d)),
            _test_config_frame=decode(config_frame(src.cfg)),
        )
        try:
            receiver.start()
            assert events.get(timeout=3) == ("config", src.cfg)
            assert events.get(timeout=3)[0] == "sample"
            src.disconnect_clients()
            wait_for(lambda: src.connections >= 2)
            wait_for(lambda: len([k for k, _ in logs if k == "test_configuration_imported"]) >= 2)
            # Wait for the new connection's first measurement, not just accept().
            # Otherwise its server thread can capture the changed cfg as its
            # initial cfg and never announce a transition from the imported cfg.
            reconnected = False
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                kind, _value = events.get(timeout=1)
                if kind == "config":
                    reconnected = True
                elif kind == "sample" and reconnected:
                    break
            else:
                pytest.fail("Reconnected source did not provide a measurement")
            if allow_start:
                wait_for(lambda: len(src.commands) >= 2)
                assert src.commands == [START, START]
                assert src.command_versions == [2 if command_version == "auto" else command_version] * 2
            else:
                assert src.commands == []
            src.cfg = replace(src.cfg, data_rate=10)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                kind, value = events.get(timeout=1)
                if kind == "config" and value == src.cfg:
                    break
            else:
                pytest.fail("Source configuration change was not received")
        finally:
            receiver.close()
        assert not receiver.thread.is_alive()


def test_runtime_records_import_and_rejects_changed_online_config(tmp_path):
    from pmu_fm.runtime import Runtime

    with Simulator(multi=False) as src:
        runtime = Runtime(
            config(tmp_path, src), _model_factory=FakeModel,
            _test_config_frame=decode(config_frame(src.cfg)),
        )
        try:
            runtime.start()
            wait_for(lambda: runtime.snapshot().get("window_samples", 0) > 0)
            assert src.commands == []
            src.cfg = replace(src.cfg, data_rate=10)
            wait_for(lambda: runtime.snapshot()["phase"] == "failed")
            assert "configuration changed" in runtime.snapshot()["last_error"]
        finally:
            runtime.stop()
        saved = json.loads((runtime.run_directory / "input_config.json").read_text())
        assert saved["data_rate"] == 20
        assert runtime.counters["test_configuration_imported"] == 1


def test_import_does_not_suppress_live_config_change_request(monkeypatch):
    from pmu_fm.protocol import REQUEST_CFG2
    from tests import helpers

    original = helpers.data_frame

    def flagged_frame(cfg, timestamp, blocks):
        return original(cfg, timestamp, [dict(b, stat=0x0400) for b in blocks])

    monkeypatch.setattr(helpers, "data_frame", flagged_frame)
    with Simulator(multi=False) as src:
        events = queue.Queue(100)
        receiver = Receiver(
            DEFAULTS["source"] | {"port": src.port}, events, lambda *_a, **_k: None,
            _test_config_frame=decode(config_frame(src.cfg)),
        )
        try:
            receiver.start()
            assert events.get(timeout=3)[0] == "config"
            assert events.get(timeout=3) == ("config_pending", None)
            assert events.get(timeout=3) == ("config", src.cfg)
            assert src.commands == [REQUEST_CFG2]
        finally:
            receiver.close()
