import sys
import time

import pytest

from pmu_fm.cli import main
from pmu_fm.config import BACKENDS, DEFAULTS, dumps


@pytest.fixture(scope="module")
def tk_desktop():
    # One Tcl interpreter for the GUI cases. Multiple destroyed Tk interpreters
    # can retain callbacks/variables until GC, which is unsafe around worker threads.
    import tkinter as tk
    import gc

    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()
    gc.collect()


@pytest.fixture
def tk_window(tk_desktop):
    import tkinter as tk
    import gc

    window = tk.Toplevel(tk_desktop)
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass
    gc.collect()


def test_cli_static_no_gui(tmp_path, capsys):
    path = tmp_path / "example.toml"
    path.write_text(dumps(DEFAULTS), encoding="utf-8")
    assert "tkinter" not in sys.modules
    assert main(["check", str(path), "--schema-only"]) == 0
    assert "Static configuration check passed" in capsys.readouterr().out
    assert "tkinter" not in sys.modules
    assert main(["check", str(path)]) == 2


@pytest.mark.gui
@pytest.mark.parametrize("backend", ["timestral", "timesfm30"])
def test_gui_new_model_selection(backend, tk_window):
    from copy import deepcopy
    from pmu_fm.gui import App

    app = App(tk_window, deepcopy(DEFAULTS))
    app.variables[("model", "backend")].set(backend)
    app.update_plan()
    assert app.gather()["model"]["backend"] == backend
    assert BACKENDS[backend].model_id in app.version_label.cget("text")
    if backend == "timesfm30":
        assert "non-commercial, non-production" in app.version_label.cget("text")


@pytest.mark.gui
@pytest.mark.parametrize("mode,version", [("connect", 1), ("connect", 2), ("listen", 1), ("listen", 2)])
@pytest.mark.parametrize("command_mode", ["auto", "manual"])
def test_gui_connect_start_stop_close(tmp_path, monkeypatch, mode, version, command_mode, tk_window):
    import tkinter as tk
    from pmu_fm import gui
    from pmu_fm.runtime import Runtime
    from tests.helpers import FakeModel, Simulator
    from tests.test_runtime import config, free_port
    from pmu_fm.config import load

    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda _title, text, **_kw: errors.append(text))
    monkeypatch.setattr(gui, "Runtime", lambda cfg: Runtime(cfg, _model_factory=FakeModel))
    root = tk_window
    try:
        with Simulator(target_port=free_port() if mode == "listen" else None, version=version) as src:
            cfg = config(tmp_path, src)
            setting = "auto" if command_mode == "auto" else version
            cfg["source"].update(mode=mode, command_version=setting)
            cfg["source"]["timestamp_tolerance_seconds"] = 0.001
            cfg["network"]["frame_version"] = version
            app = gui.App(root, cfg)
            assert app.gather()["source"]["command_version"] == setting
            command_field = app.field_labels[("source", "command_version")]
            assert command_field.master["text"] == "Advanced / limits"
            assert app.variables[("source", "command_version")].get() == (
                "Auto" if setting == "auto" else str(setting)
            )
            assert app.field_labels[("source", "host")]["text"] == (
                "Local bind address" if mode == "listen" else "Remote source address"
            )
            app.variables[("model", "input_interval_seconds")].set("1/30")
            assert app.gather()["model"]["input_interval_seconds"] == "1/30"
            app.variables[("model", "input_interval_seconds")].set("0")
            app.connect()

            def pump(predicate, timeout=10):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    root.update()
                    if predicate():
                        return
                    time.sleep(0.02)
                raise AssertionError(f"GUI timeout; errors={errors}")

            pump(lambda: app.source_config is not None)
            assert app.source_config.version == version
            assert f"command version {version}" in app.source_origin.get()
            assert "10/freq" in app.tree.get_children("pmu:10")
            saved = tmp_path / "saved.toml"
            monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **_kw: str(saved))
            app.save_config()
            assert load(saved)["source"]["mode"] == mode
            assert load(saved)["source"]["command_version"] == setting
            assert load(saved)["source"]["timestamp_tolerance_seconds"] == 0.001
            assert load(saved)["network"]["frame_version"] == version
            # Reject an invalid tolerance before opening sockets or starting a worker.
            app.variables[("source", "timestamp_tolerance_seconds")].set("0.1")
            app.start()
            assert app.runtime is None and not app.busy
            assert len(errors) == 1 and "required tolerance <" in errors.pop()
            app.variables[("source", "timestamp_tolerance_seconds")].set("0.001")
            app.start()
            pump(lambda: app.runtime and app.runtime.counters["forecast_completed"] >= 1)
            assert str(app.connect_button["state"]) == "disabled"
            assert app.runtime.latest_result.reference.version == version
            assert app.runtime.snapshot()["input_command_version"] == version
            assert (
                app.stop_button.winfo_rooty() + app.stop_button.winfo_height()
                <= root.winfo_rooty() + root.winfo_height()
            )
            monkeypatch.setattr(gui.messagebox, "askyesno", lambda *_a, **_k: False)
            app.on_close()
            assert not app.closing and app.runtime.snapshot()["running"]
            monkeypatch.setattr(gui.messagebox, "askyesno", lambda *_a, **_k: True)
            app.on_close()
            pump(lambda: not app.runtime.snapshot()["running"])
            assert app.closing and not errors
    finally:
        try:
            if "app" in locals() and app.runtime:
                app.runtime.stop()
            root.destroy()
        except tk.TclError:
            pass


@pytest.mark.gui
def test_gui_cancel_listen_read(monkeypatch, tk_window):
    from copy import deepcopy
    from pmu_fm import gui
    from tests.test_runtime import free_port
    from pmu_fm.network import listen_socket

    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda _title, text, **_kw: errors.append(text))
    cfg = deepcopy(DEFAULTS)
    cfg["source"].update(mode="listen", port=free_port())
    root = tk_window
    app = gui.App(root, cfg)
    try:
        app.connect()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not app.inspecting:
            root.update()
        assert str(app.stop_button["text"]) == "Cancel read"
        assert str(app.stop_button["state"]) == "normal"
        app.stop()
        while time.monotonic() < deadline and app.busy:
            root.update()
            time.sleep(0.02)
        assert not app.busy and not app.inspecting and not errors
        assert "cancelled" in app.status.get()
        with listen_socket("127.0.0.1", cfg["source"]["port"], 1):
            pass
    finally:
        app.inspect_stop.set()
        root.destroy()


def pump_gui(root, predicate, errors, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"GUI timeout; errors={errors}")


@pytest.mark.gui
@pytest.mark.parametrize("mode,version", [("connect", 1), ("connect", 2), ("listen", 1), ("listen", 2)])
def test_gui_import_start_stop_and_online_switch(tmp_path, monkeypatch, mode, version, tk_window):
    from pmu_fm import gui
    from pmu_fm.protocol import config_frame
    from pmu_fm.runtime import Runtime
    from tests.helpers import FakeModel, Simulator
    from tests.test_runtime import config, free_port

    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda _title, text, **_kw: errors.append(text))
    monkeypatch.setattr(gui, "Runtime", lambda cfg, **kw: Runtime(cfg, _model_factory=FakeModel, **kw))
    root, app = tk_window, None
    try:
        with Simulator(target_port=free_port() if mode == "listen" else None, version=version) as src:
            cfg = config(tmp_path, src)
            cfg["source"]["mode"] = mode
            path = tmp_path / "capture.hex"
            path.write_text(config_frame(src.cfg).hex("-"))
            app = gui.App(root, cfg)
            monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_kw: str(path))
            app.import_button.invoke()
            pump_gui(root, lambda: not app.busy, errors)
            assert app.imported_frame is not None and app.source_config == src.cfg
            assert src.connections == 0  # Importing must not open an input socket.
            assert "10/freq" in app.tree.get_children("pmu:10")
            assert "capture.hex" in app.source_origin.get()
            assert "History:" in app.plan_label["text"]
            app.start()
            pump_gui(root, lambda: app.runtime and app.runtime.counters["forecast_completed"] >= 1, errors)
            assert app.runtime.counters["test_configuration_imported"] == 1
            assert app.runtime.latest_result.reference.version == version
            assert src.commands == []
            assert str(app.import_button["state"]) == "disabled"
            assert str(app.clear_import_button["state"]) == "disabled"
            assert app.clear_import_button.winfo_rootx() + app.clear_import_button.winfo_width() <= (
                root.winfo_rootx() + root.winfo_width()
            )
            app.stop()
            pump_gui(root, lambda: not app.runtime.snapshot()["running"], errors)
            pump_gui(root, lambda: str(app.import_button["state"]) == "normal", errors)
            app.connect_button.invoke()
            pump_gui(root, lambda: not app.busy, errors)
            assert app.imported_frame is None and app.imported_path is None
            assert "online" in app.source_origin.get()
            assert src.commands and not errors
    finally:
        if app and app.runtime:
            app.runtime.stop()
        root.destroy()


@pytest.mark.gui
def test_gui_import_validation_settings_and_clear(tmp_path, monkeypatch, tk_window):
    from copy import deepcopy
    from pathlib import Path
    from pmu_fm import gui
    from pmu_fm.protocol import decode

    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda _title, text, **_kw: errors.append(text))
    cfg = deepcopy(DEFAULTS)
    cfg["source"].update(id_code=30001, channels=["30001/freq"])
    path = Path(__file__).parent / "fixtures" / "synthetic_cfg2.hex"
    raw = bytes.fromhex(path.read_text().replace("-", ""))
    root = tk_window
    app = gui.App(root, cfg)
    try:
        monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_kw: str(path))
        app.import_configuration()
        pump_gui(root, lambda: not app.busy, errors)
        assert not errors and app.imported_frame == decode(raw)
        assert "30001/freq" in app.tree.get_children("pmu:30001")

        bad = tmp_path / "bad.hex"
        bad.write_text((raw[:-1] + bytes([raw[-1] ^ 1])).hex("-"))
        monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_kw: str(bad))
        app.import_configuration()
        pump_gui(root, lambda: not app.busy, errors)
        assert "CRC mismatch" in errors.pop()
        assert app.imported_frame == decode(raw)  # Failed replacement retains the valid import.
        monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_kw: "")
        app.import_configuration()
        assert app.imported_frame == decode(raw) and not app.busy

        app.variables[("source", "port")].set("9999")
        app.start()
        assert "Source settings changed" in errors.pop()
        assert app.runtime is None and not app.busy
        app.check()
        assert "Source settings changed" in errors.pop()
        app.variables[("source", "port")].set(str(cfg["source"]["port"]))
        app.variables[("source", "id_code")].set("2221")
        monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_kw: str(path))
        app.import_configuration()
        pump_gui(root, lambda: not app.busy, errors)
        assert "ID differs" in errors.pop()
        app.start()
        assert "Source settings changed" in errors.pop()
        app.variables[("source", "id_code")].set("30001")

        saved = tmp_path / "settings.toml"
        monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **_kw: str(saved))
        app.save_config()
        assert "not saved in TOML" in app.status.get()
        assert "synthetic_cfg2.hex" not in saved.read_text()
        app.clear_import_button.invoke()
        assert app.imported_frame is None and app.source_config is None
        assert not app.tree.get_children() and app.selected == {"30001/freq"}
        assert "online" in app.source_origin.get()
        assert "Read or import" in app.plan_label["text"]
        app.import_configuration()
        pump_gui(root, lambda: not app.busy, errors)
        assert app.imported_frame is not None
        monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_kw: str(saved))
        app.load_config()
        assert app.imported_frame is None and app.source_config is None
        assert not errors
    finally:
        root.destroy()
