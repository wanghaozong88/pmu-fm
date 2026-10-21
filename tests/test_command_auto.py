"""Automatic command choice against an independent, version-strict TCP source."""

import json
import queue
import threading
import time
from copy import deepcopy

import pytest

from pmu_fm.config import ConfigError, DEFAULTS, dumps, load, normalize, validate
from pmu_fm.network import Receiver, SourceTransport, inspect_source, listen_socket
from pmu_fm.protocol import ProtocolError, REQUEST_CFG2, START, STOP
from pmu_fm.runtime import Runtime
from tests.helpers import FakeModel, wait_for
from tests.test_runtime import config, free_port


def source_settings(source, mode, command_version="auto"):
    return DEFAULTS["source"] | {
        "mode": mode, "port": source.port, "command_version": command_version,
        "timeout_seconds": 0.35, "reconnect_seconds": 0.1,
    }


def independent_source(**kwargs):
    pytest.importorskip("pySynphasor", reason="Independent source unavailable; command auto NOT verified")
    from tests.external_source import ExternalSource

    return ExternalSource(**kwargs)


@pytest.mark.parametrize("value", ["auto", 1, 2])
def test_command_setting_roundtrip_preserves_explicit_override(tmp_path, value):
    cfg = deepcopy(DEFAULTS)
    cfg["source"]["command_version"] = value
    validate(cfg, False, False)
    path = tmp_path / "settings.toml"
    path.write_text(dumps(cfg), encoding="utf-8")
    assert load(path)["source"]["command_version"] == value
    assert normalize({})["source"]["command_version"] == "auto"
    assert normalize({})["network"]["frame_version"] == 1


@pytest.mark.parametrize("value", [True, False, 0, 3, 1.0, "1", "2", "Auto", "latest"])
def test_invalid_command_setting(value):
    with pytest.raises(ConfigError, match="command_version"):
        normalize({"source": {"command_version": value}})


@pytest.mark.external
@pytest.mark.parametrize("mode", ["connect", "listen"])
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("close_on_wrong", [False, True])
def test_silent_source_bounded_fallback_and_no_start(mode, version, close_on_wrong):
    with independent_source(
        target_port=free_port() if mode == "listen" else None,
        version=3 - version, accepted_command_version=version,
        data_before_config=False, require_start=True, close_on_wrong_version=close_on_wrong,
    ) as src:
        events = []
        settings = source_settings(src, mode) | {"allow_start": True}
        inspected = inspect_source(settings, emit=lambda k, _m, **d: events.append((k, d)))
        assert inspected.version == 3 - version
        expected = [1] if version == 1 else [1, 2]
        assert src.command_versions == expected and src.commands == [REQUEST_CFG2] * len(expected)
        assert src.connections == len(expected)
        selected = [d for k, d in events if k == "command_version_selected"]
        assert selected[0]["command_version"] == version
        assert selected[0]["configuration_version"] == 3 - version
        assert settings["command_version"] == "auto"
    if mode == "listen":
        with listen_socket("127.0.0.1", settings["port"], 1):
            pass


@pytest.mark.external
@pytest.mark.parametrize("mode", ["connect", "listen"])
@pytest.mark.parametrize("hint", [1, 2])
@pytest.mark.parametrize("accept_hint", [False, True])
def test_input_version_is_only_a_hint(mode, hint, accept_hint):
    accepted = hint if accept_hint else 3 - hint
    with independent_source(
        target_port=free_port() if mode == "listen" else None,
        version=hint, accepted_command_version=accepted,
    ) as src:
        transport = SourceTransport(source_settings(src, mode), threading.Event(), lambda *_a, **_k: None)
        try:
            transport.connect()
            cfg = transport.read_configuration()
            assert cfg.version == hint and transport.command_version == accepted
            assert src.command_versions == ([hint] if accept_hint else [hint, 3 - hint])
            assert src.connections == (1 if accept_hint else 2)
        finally:
            transport.close()
        assert all(c == REQUEST_CFG2 for c in src.commands)


@pytest.mark.external
@pytest.mark.parametrize("setting", [1, 2])
def test_manual_override_does_not_follow_received_version_or_fallback(setting):
    with independent_source(version=3 - setting, accepted_command_version=setting) as src:
        inspected = inspect_source(source_settings(src, "connect", setting))
        assert inspected.version == 3 - setting
        assert src.command_versions == [setting]
    with independent_source(version=3 - setting, accepted_command_version=3 - setting) as src:
        with pytest.raises(TimeoutError, match="timeout alone does not establish"):
            inspect_source(source_settings(src, "connect", setting))
        assert src.command_versions == [setting] and src.connections == 1


@pytest.mark.external
def test_cli_online_inspection_reports_command_version(tmp_path, capsys):
    from pmu_fm.cli import main

    with independent_source(version=1, accepted_command_version=2, data_before_config=False) as src:
        cfg = deepcopy(DEFAULTS)
        cfg["source"].update(source_settings(src, "connect"), allow_start=True)
        path = tmp_path / "inspect.toml"
        path.write_text(dumps(cfg), encoding="utf-8")
        assert main(["check", str(path), "--online", "--schema-only"]) == 0
        assert '"command_version": 2' in capsys.readouterr().out
        assert src.commands == [REQUEST_CFG2, REQUEST_CFG2] and src.command_versions == [1, 2]


@pytest.mark.external
@pytest.mark.parametrize("mode", ["connect", "listen"])
def test_reconnect_reselects_and_start_uses_selected_command_version(tmp_path, mode):
    with independent_source(
        target_port=free_port() if mode == "listen" else None,
        version=1, accepted_command_version=2, data_before_config=False, require_start=True,
    ) as src:
        cfg = config(tmp_path, src)
        cfg["source"].update(source_settings(src, mode), channels=["10/freq"], allow_start=True)
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.counters["forecast_completed"])
            assert run.snapshot()["input_command_version"] == 2
            assert run.snapshot()["source_frame_version"] == 1
            assert src.accepted_commands == [(2, REQUEST_CFG2), (2, START)]
            # Subsequent CFG-2 requests on this connection use the selected command
            # version, even though all received frame headers contain version 1.
            run.receiver.transport.command(REQUEST_CFG2)
            wait_for(lambda: src.accepted_commands.count((2, REQUEST_CFG2)) == 2)
            assert src.accepted_commands.count((2, START)) == 1
            src.accepted_command_version = 1
            src.sock.shutdown(2)
            wait_for(lambda: run.snapshot().get("input_command_version") == 1)
            wait_for(lambda: (1, START) in src.accepted_commands)
            assert run.snapshot()["phase"] == "running"
        finally:
            run.stop()
        assert STOP not in src.commands
        events = [json.loads(line) for line in (run.run_directory / "events.jsonl").read_text().splitlines()]
        selections = [e for e in events if e["event"] == "command_version_selected"]
        assert [e["command_version"] for e in selections] == [2, 1]


@pytest.mark.external
def test_no_configuration_despite_data_does_not_extend_probe_timeout():
    # Test source accepts no command versions, but continuously sends valid DATA.
    with independent_source(accepted_command_version=0) as src:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="Check endpoint, stream ID"):
            inspect_source(source_settings(src, "connect"))
        assert time.monotonic() - started < 2
        assert src.commands == [REQUEST_CFG2, REQUEST_CFG2]
        assert src.command_versions == [1, 2]


@pytest.mark.external
@pytest.mark.parametrize("mode", ["connect", "listen"])
def test_cancel_during_auto_probe_cleans_up_without_start(mode):
    with independent_source(
        target_port=free_port() if mode == "listen" else None,
        accepted_command_version=0, data_before_config=False,
    ) as src:
        settings = source_settings(src, mode) | {"timeout_seconds": 5.0, "allow_start": True}
        receiver = Receiver(settings, queue.Queue(), lambda *_a, **_kw: None)
        receiver.start()
        try:
            wait_for(lambda: src.commands)
        finally:
            started = time.monotonic()
            receiver.close()
        assert time.monotonic() - started < 2 and not receiver.thread.is_alive()
        assert src.commands == [REQUEST_CFG2]
    if mode == "listen":
        with listen_socket("127.0.0.1", settings["port"], 1):
            pass


@pytest.mark.external
@pytest.mark.parametrize("damage", ["crc", "id", "config"])
def test_invalid_reply_fails_without_alternate_version(damage):
    src = independent_source(data_before_config=False)
    original = src.configuration

    def broken():
        if damage == "crc":
            raw = original()
            return raw[:-1] + bytes([raw[-1] ^ 1])
        if damage == "id":
            src.ident = 99
            return original()
        from fractions import Fraction

        return src.frame(3, b"\0", Fraction(100))

    src.configuration = broken
    with src:
        with pytest.raises(ProtocolError):
            inspect_source(source_settings(src, "connect"))
        assert src.command_versions == [1] and src.connections == 1


@pytest.mark.external
def test_cfg_and_fragmented_data_survive_configuration_acquisition():
    from fractions import Fraction

    with independent_source(data_before_config=False, require_start=True) as src:
        original = src.configuration
        frame = src.data(0, Fraction(100))
        following = src.data(1, Fraction(100))
        src.configuration = lambda: original() + frame + following[:9]
        transport = SourceTransport(source_settings(src, "connect"), threading.Event(), lambda *_a, **_k: None)
        try:
            transport.connect()
            transport.read_configuration()
            frames = transport.read_frames()
            assert [f.kind for f in frames] == [3, 0]
            assert frames[1].timestamp(1_000_000) == 100
            src.sock.sendall(following[9:])
            frames = transport.read_frames()
            assert len(frames) == 1 and frames[0].timestamp(1_000_000) == Fraction(2001, 20)
        finally:
            transport.close()


@pytest.mark.external
@pytest.mark.parametrize("mode", ["connect", "listen"])
def test_unsolicited_configuration_is_preserved_and_inference_is_disclosed(mode):
    from fractions import Fraction

    src = independent_source(
        target_port=free_port() if mode == "listen" else None,
        version=2, accepted_command_version=0,
    )
    original = src.stream

    def push_configuration():
        src.sock.sendall(src.configuration() + src.data(0, Fraction(100)))
        original()

    src.stream = push_configuration
    with src:
        records = []
        transport = SourceTransport(
            source_settings(src, mode), threading.Event(), lambda k, _m, **d: records.append((k, d))
        )
        try:
            transport.connect()
            cfg = transport.read_configuration()
            assert cfg.version == transport.command_version == 2
            frames = transport.read_frames()
            assert [f.kind for f in frames[:2]] == [3, 0]
            assert frames[1].timestamp(1_000_000) == 100
            wait_for(lambda: src.commands == [REQUEST_CFG2])
            selected = next(d for k, d in records if k == "command_version_selected")
            assert selected["basis"] == "Unsolicited CFG-2; command version inferred, not verified"
        finally:
            transport.close()
        assert src.connections == 1 and src.command_versions == [2]
