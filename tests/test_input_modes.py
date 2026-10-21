"""Real TCP tests for both roles, versions, control boundaries and cleanup."""

import queue
import socket
import threading
import time
from copy import deepcopy
from dataclasses import replace

import pytest

from pmu_fm.config import DEFAULTS, ConfigError, dumps, load, validate
from pmu_fm.network import Receiver, inspect_source, listen_socket
from pmu_fm.runtime import Runtime
from tests.helpers import FakeModel, Simulator, wait_for
from tests.test_runtime import config, free_port


@pytest.mark.parametrize("mode", ["connect", "listen"])
@pytest.mark.parametrize("version", [1, 2])
def test_roles_versions_and_explicit_start(tmp_path, mode, version):
    port = free_port() if mode == "listen" else None
    with Simulator(target_port=port, version=version, require_start=True) as src:
        cfg = config(tmp_path, src)
        cfg["source"].update(mode=mode, command_version=version, allow_start=True)
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.counters["forecast_completed"])
            assert src.commands[:2] == [5, 2]
            assert set(src.command_versions) == {version}
            assert run.latest_result.reference.version == version
            assert list(run.latest_result.rows())[0]["source_frame_version"] == version
            assert run.snapshot()["input_state"] == "receiving"
            assert run.snapshot()["source_frame_version"] == version
        finally:
            run.stop()
        assert 1 not in src.commands
        assert not run.receiver.thread.is_alive()


@pytest.mark.parametrize("version", [1, 2])
def test_listener_inspection_then_runtime_reconnect(tmp_path, version):
    with Simulator(target_port=free_port(), version=version) as src:
        cfg = config(tmp_path, src)
        cfg["source"].update(mode="listen", command_version=version, allow_start=True)
        inspected = inspect_source(cfg["source"])
        assert inspected.version == version
        assert 1 not in src.commands and 2 not in src.commands  # inspection ignores allow_start
        run = Runtime(cfg, _model_factory=FakeModel)
        try:
            run.start()
            wait_for(lambda: run.counters["forecast_completed"])
            assert src.connections >= 2
            old = run.counters["forecast_completed"]
            src.disconnect_clients()
            wait_for(lambda: run.counters["disconnected"])
            wait_for(lambda: run.counters["forecast_completed"] > old)
            assert src.connections >= 3
        finally:
            run.stop()
        assert 1 not in src.commands


def test_listener_rejects_second_source_and_changes_config(tmp_path):
    with Simulator(target_port=free_port()) as src:
        cfg = config(tmp_path, src)
        cfg["source"]["mode"] = "listen"
        run = Runtime(cfg, _model_factory=FakeModel)
        extra = None
        try:
            run.start()
            wait_for(lambda: run.counters["forecast_completed"])
            first_peer = run.snapshot()["input_peer"]
            extra = socket.create_connection(("127.0.0.1", src.port), timeout=1)
            wait_for(lambda: run.counters["input_client_rejected"])
            assert run.snapshot()["input_peer"] == first_peer
            old = run.counters["forecast_completed"]
            wait_for(lambda: run.counters["forecast_completed"] > old)
            src.cfg = replace(src.cfg, pmus=(replace(src.cfg.pmus[0], config_count=2),) + src.cfg.pmus[1:])
            wait_for(lambda: run.snapshot()["phase"] == "failed")
            assert "configuration changed" in run.snapshot()["last_error"]
        finally:
            if extra:
                extra.close()
            run.stop()
        assert 1 not in src.commands


def test_cancel_listener_inspection_and_stop_waiting_runtime(tmp_path):
    settings = deepcopy(DEFAULTS["source"])
    settings.update(mode="listen", port=free_port())
    stop, results = threading.Event(), queue.Queue()

    def inspect():
        try:
            inspect_source(settings, stop)
        except Exception as exc:
            results.put(exc)

    thread = threading.Thread(target=inspect)
    thread.start()
    time.sleep(0.1)
    stop.set()
    thread.join(2)
    assert not thread.is_alive() and isinstance(results.get_nowait(), InterruptedError)
    # The same endpoint can be rebound immediately after cancel/stop.
    cfg = deepcopy(DEFAULTS)
    cfg["source"].update(settings, channels=["10/freq"])
    cfg["network"]["enabled"] = False
    cfg["files"]["directory"] = str(tmp_path)
    run = Runtime(cfg, _model_factory=FakeModel)
    for _ in range(2):
        run.start()
        wait_for(lambda: run.snapshot().get("input_state") == "waiting_for_source")
        assert run.counters["input_error"] == 0
        run.stop()
        assert not run.receiver.thread.is_alive()


def test_listener_silent_peer_timeout_no_start_before_valid_config():
    settings = deepcopy(DEFAULTS["source"])
    settings.update(mode="listen", port=free_port(), timeout_seconds=0.2, allow_start=True)
    events, records = queue.Queue(), []
    receiver = Receiver(settings, events, lambda k, *a, **kw: records.append(k))
    receiver.start()
    peer = socket.create_connection(("127.0.0.1", settings["port"]), timeout=1)
    try:
        wait_for(lambda: "input_error" in records)
        assert "upstream_start" not in records
        wait_for(lambda: receiver.transport.phase == "waiting_for_source")
    finally:
        peer.close()
        receiver.close()
    assert not receiver.thread.is_alive()


def test_bind_failure_configuration_and_legacy_defaults(tmp_path):
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('schema_version = 1\n[source]\nhost = "127.0.0.1"\nport = 4712\n', encoding="utf-8")
    old = load(legacy)
    assert old["source"]["mode"] == "connect" and old["source"]["command_version"] == "auto"
    assert old["network"]["frame_version"] == 1
    for section, key, value in [
        ("source", "mode", "udp"),
        ("source", "command_version", 3),
        ("network", "frame_version", 0),
    ]:
        cfg = deepcopy(DEFAULTS)
        cfg[section][key] = value
        with pytest.raises(ConfigError):
            validate(cfg, False, False)
    cfg = deepcopy(DEFAULTS)
    cfg["source"].update(mode="listen", port=cfg["network"]["port"])
    with pytest.raises(ConfigError, match="bind endpoints"):
        validate(cfg, False, False)
    cfg["source"].update(port=free_port(), command_version=2, channels=["10/freq"])
    path = tmp_path / "listen.toml"
    path.write_text(dumps(cfg), encoding="utf-8")
    assert load(path)["source"]["mode"] == "listen"
    with listen_socket("127.0.0.1", cfg["source"]["port"], 1):
        with pytest.raises(OSError, match="Cannot bind source listener"):
            inspect_source(cfg["source"])
