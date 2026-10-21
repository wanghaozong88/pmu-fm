"""Tk is only presentation and configuration; Runtime owns all execution."""

from __future__ import annotations

import queue
import threading
from copy import deepcopy
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import BACKENDS, DEFAULTS, dumps, load, normalize, sample_plan, validate
from .network import inspect_source
from .protocol import channels, parse_config
from .runtime import Runtime
from .test_config_input import load_test_config_frame


class App:
    def __init__(self, root, cfg=None):
        self.root = root
        self.cfg = normalize(cfg or DEFAULTS)
        # GUI paths are interpreted from the launch directory; loaded TOML is resolved by load().
        self.runtime = None
        self.source_config = None
        self.source_signature = None
        self.imported_frame = None
        self.imported_path = None
        self.selected = set(self.cfg["source"]["channels"])
        self.variables, self.controls = {}, []
        self.field_labels = {}
        self.messages = queue.Queue()
        self.busy, self.closing = False, False
        self.inspecting = False
        self.was_running = False
        self.inspect_stop = threading.Event()
        root.title("PMU Foundation Model Bridge")
        root.geometry("840x800")
        root.minsize(760, 760)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="PMU Foundation Model Bridge",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        source_tab, model_tab, output_tab = [ttk.Frame(self.tabs, padding=12) for _ in range(3)]
        for tab, title in zip(
            (source_tab, model_tab, output_tab),
            ("Source & channels", "Local model & schedule", "Outputs & limits"),
        ):
            self.tabs.add(tab, text=title)
            tab.columnconfigure(1, weight=1)
        self.field(source_tab, 0, "Connection mode", "source", "mode", choices=["connect", "listen"])
        self.field(source_tab, 1, "Remote source address", "source", "host")
        self.field(source_tab, 2, "Remote TCP port", "source", "port")
        self.field(source_tab, 3, "Stream IDCODE", "source", "id_code")
        self.field(source_tab, 4, "Allow upstream START (explicit permission)", "source", "allow_start")
        self.field(
            source_tab,
            5,
            "Timestamp tolerance (seconds; 0 = wire tick)",
            "source",
            "timestamp_tolerance_seconds",
        )
        source_actions = ttk.Frame(source_tab)
        source_actions.grid(row=6, column=0, columnspan=3, sticky="w", pady=5)
        self.connect_button = ttk.Button(
            source_actions, text="Connect and read configuration", command=self.connect
        )
        self.connect_button.pack(side="left", padx=(0, 6))
        self.controls.append(self.connect_button)
        self.import_button = ttk.Button(
            source_actions, text="Import CFG-2 (.hex)", command=self.import_configuration
        )
        self.import_button.pack(side="left", padx=(0, 6))
        self.clear_import_button = ttk.Button(
            source_actions, text="Clear import", command=self.clear_import
        )
        self.clear_import_button.pack(side="left")
        self.controls.extend((self.import_button, self.clear_import_button))
        self.source_origin = tk.StringVar(value="Configuration source: online")
        ttk.Label(source_tab, textvariable=self.source_origin, wraplength=700).grid(
            row=7, column=0, columnspan=3, sticky="w"
        )
        treeframe = ttk.Frame(source_tab)
        treeframe.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=4)
        source_tab.rowconfigure(8, weight=1)
        self.tree = ttk.Treeview(treeframe, columns=("id", "unit"), selectmode="browse", height=10)
        self.tree.heading("#0", text="Select channels (click checkbox / Space)")
        self.tree.heading("id", text="Stable ID")
        self.tree.heading("unit", text="Unit")
        self.tree.column("#0", width=340)
        self.tree.column("id", width=210)
        self.tree.column("unit", width=90)
        scrollbar = ttk.Scrollbar(treeframe, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<ButtonRelease-1>", self.toggle)
        self.tree.bind("<space>", self.toggle)
        self.source_note = tk.StringVar(
            value=f"{len(self.selected)} saved selections; read CFG-2 to view channels."
        )
        ttk.Label(source_tab, textvariable=self.source_note, wraplength=720).grid(
            row=9, column=0, columnspan=3, sticky="w"
        )

        self.field(model_tab, 0, "Model", "model", "backend", choices=list(BACKENDS))
        self.version_label = ttk.Label(model_tab, text="", wraplength=720)
        self.version_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        self.field(model_tab, 2, "Local weight directory", "model", "local_dir", browse="directory")
        self.field(model_tab, 3, "Weight revision / version", "model", "revision")
        self.field(model_tab, 4, "Device", "model", "device", choices=["cpu", "cuda"])
        self.field(model_tab, 5, "Trusted Python preprocessing", "script", "path", browse="script")
        self.field(model_tab, 6, "History window (seconds)", "model", "context_seconds")
        self.field(model_tab, 7, "Forecast horizon (seconds)", "model", "horizon_seconds")
        self.field(model_tab, 8, "Call interval (seconds)", "model", "interval_seconds")
        self.field(
            model_tab, 9, "Model interval (seconds, e.g. 1/30; 0 = source)", "model", "input_interval_seconds"
        )
        self.plan_label = ttk.Label(
            model_tab, text="Read source configuration to calculate sample counts.", wraplength=720
        )
        self.plan_label.grid(row=10, column=0, columnspan=3, sticky="w", pady=8)
        self.field(output_tab, 0, "Enable C37.118.2 TCP output", "network", "enabled")
        self.field(output_tab, 1, "Bind address (local by default)", "network", "host")
        self.field(output_tab, 2, "Output TCP port", "network", "port")
        self.field(
            output_tab, 3, "Fixed prediction leads (seconds, comma-separated)", "network", "leads_seconds"
        )
        self.field(output_tab, 4, "Enable prediction CSV", "files", "enabled")
        self.field(output_tab, 5, "Run metadata / CSV directory", "files", "directory", browse="directory")
        advanced = ttk.LabelFrame(output_tab, text="Advanced / limits", padding=6)
        advanced.grid(row=6, column=0, columnspan=3, sticky="ew", pady=4)
        advanced.columnconfigure(1, weight=1)
        self.field(
            advanced, 0, "Input command version", "source", "command_version", choices=["Auto", 1, 2]
        )
        for row, (title, section, key) in enumerate(
            [
                ("Output frame version (1 or 2)", "network", "frame_version"),
                ("Output stream ID", "network", "id_code"),
                ("First virtual PMU ID", "network", "pmu_id_start"),
                ("Source timeout (s)", "source", "timeout_seconds"),
                ("Reorder wait (s)", "source", "reorder_seconds"),
                ("Script timeout (s)", "script", "timeout_seconds"),
                ("Inference timeout (s)", "model", "timeout_seconds"),
                ("Worker RSS limit (MiB)", "model", "memory_mb"),
                ("Missing samples script may repair (default 0)", "script", "max_missing_samples"),
            ], start=1,
        ):
            self.field(advanced, row, title, section, key)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=8)
        for title, command in (
            ("Load config", self.load_config),
            ("Save config", self.save_config),
            ("Check", self.check),
            ("Start", self.start),
        ):
            button = ttk.Button(actions, text=title, command=command)
            button.pack(side="left", padx=(0, 6))
            self.controls.append(button)
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        self.status = tk.StringVar(
            value="Stopped. Prepare local weights and explicitly read source configuration."
        )
        ttk.Label(
            outer, textvariable=self.status, justify="left", wraplength=800, relief="sunken", padding=8
        ).pack(fill="x")
        self.apply(self.cfg)
        self.root.after(150, self.poll)

    def field(self, parent, row, title, section, key, choices=None, browse=None):
        default = DEFAULTS[section][key]
        variable = tk.BooleanVar() if isinstance(default, bool) else tk.StringVar()
        self.variables[(section, key)] = variable
        if isinstance(default, bool):
            control = ttk.Checkbutton(parent, text=title, variable=variable)
            control.grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        else:
            caption = ttk.Label(parent, text=title)
            caption.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
            self.field_labels[(section, key)] = caption
            control = (
                ttk.Combobox(parent, textvariable=variable, values=choices, state="readonly")
                if choices
                else ttk.Entry(parent, textvariable=variable)
            )
            control.grid(row=row, column=1, sticky="ew", pady=3)
        self.controls.append(control)
        if browse:

            def choose():
                path = (
                    filedialog.askdirectory(parent=self.root)
                    if browse == "directory"
                    else filedialog.askopenfilename(parent=self.root, filetypes=[("Python script", "*.py")])
                )
                if path:
                    variable.set(path)

            button = ttk.Button(parent, text="Browse", command=choose)
            button.grid(row=row, column=2, padx=(6, 0))
            self.controls.append(button)
        variable.trace_add("write", lambda *_: self.update_plan())

    def apply(self, cfg):
        self.cfg = deepcopy(cfg)
        self.selected = set(cfg["source"]["channels"])
        for (section, key), var in self.variables.items():
            value = cfg[section][key]
            if section == "source" and key == "command_version" and value == "auto":
                value = "Auto"
            var.set(", ".join(map(str, value)) if isinstance(value, list) else value)
        self.update_plan()

    def gather(self):
        cfg = deepcopy(self.cfg)
        for (section, key), var in self.variables.items():
            original, value = DEFAULTS[section][key], var.get()
            try:
                if section == "source" and key == "command_version":
                    value = "auto" if value.lower() == "auto" else int(value)
                elif isinstance(original, list):
                    value = [float(x.strip()) for x in value.split(",") if x.strip()]
                elif section == "model" and key == "input_interval_seconds" and "/" in value:
                    value = value.strip()
                elif not isinstance(original, bool):
                    value = type(original)(value)
                cfg[section][key] = value
            except ValueError as exc:
                raise ValueError(f"{section}.{key}: invalid value") from exc
        cfg["source"]["channels"] = sorted(self.selected)
        return normalize(cfg, Path.cwd())

    def update_plan(self):
        if hasattr(self, "connect_button"):
            listening = self.variables[("source", "mode")].get() == "listen"
            self.field_labels[("source", "host")].configure(
                text="Local bind address" if listening else "Remote source address"
            )
            self.field_labels[("source", "port")].configure(
                text="Local listening port" if listening else "Remote TCP port"
            )
            self.connect_button.configure(
                text="Listen and read configuration" if listening else "Connect and read configuration"
            )
        if not hasattr(self, "plan_label"):
            return
        try:
            cfg = self.gather()
            backend = BACKENDS[cfg["model"]["backend"]]
            self.version_label.configure(
                text=f"{backend.model_id} · {backend.package} {' / '.join(backend.versions)}"
                + (f"\n{backend.notice}" if backend.notice else "")
            )
            if self.source_config and self.source_signature == self.signature(cfg):
                plan = sample_plan(cfg, self.source_config.interval, self.source_config.time_base)
                self.plan_label.configure(
                    text=f"Source: {1 / self.source_config.interval} samples/s · "
                    f"History: {plan['source_context']} source / {plan['context']} model samples · "
                    f"Forecast: {plan['horizon']} steps · Model spacing: {plan['dt']} s"
                )
            elif self.imported_frame is not None:
                self.plan_label.configure(text="Source settings changed. Re-import the CFG-2 file or clear import.")
            else:
                self.plan_label.configure(text="Read or import source configuration to calculate sample counts.")
        except (ValueError, KeyError) as exc:
            self.plan_label.configure(text=str(exc))

    @staticmethod
    def signature(cfg):
        return tuple(cfg["source"][k] for k in ("mode", "host", "port", "id_code", "command_version"))

    def toggle(self, e):
        if self.busy or (self.runtime and self.runtime.snapshot()["running"]):
            return
        iid = self.tree.focus() if e.keysym == "space" else self.tree.identify_row(e.y)
        if not iid:
            return
        ids = self.tree.get_children(iid) or (iid,)
        for channel in ids:
            if channel.startswith("pmu:"):
                continue
            if channel in self.selected:
                self.selected.remove(channel)
            else:
                self.selected.add(channel)
            text = self.tree.item(channel, "text")[4:]
            self.tree.item(channel, text=("[x] " if channel in self.selected else "[ ] ") + text)
        self.source_note.set(
            f"{len(self.selected)} channels selected. Digital states are quality metadata only."
        )

    def set_locked(self, locked):
        for control in self.controls:
            control.configure(
                state="disabled"
                if locked
                else ("readonly" if isinstance(control, ttk.Combobox) else "normal")
            )
        self.clear_import_button.configure(
            state="disabled" if locked or self.imported_frame is None else "normal"
        )
        self.stop_button.configure(
            text="Cancel read" if self.inspecting else "Stop",
            state="normal"
            if self.inspecting or (self.runtime and self.runtime.snapshot()["running"])
            else "disabled",
        )

    def background(self, fn, callback):
        self.busy = True
        self.set_locked(True)

        def work():
            try:
                self.messages.put((callback, fn(), None))
            except Exception as exc:
                self.messages.put((callback, None, str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def connect(self):
        try:
            cfg = self.gather()
            validate(cfg, require_weights=False, require_channels=False)
            self.inspect_stop.clear()
            self.inspecting = True
            self.status.set(
                "Waiting for source connection (up to 30 seconds; Cancel read to stop)..."
                if cfg["source"]["mode"] == "listen"
                else "Reading CFG-2 from the specified source..."
            )

            def read_source():
                selected = {}

                def record(kind, _message, **details):
                    if kind == "command_version_selected":
                        selected.update(details)

                source = inspect_source(cfg["source"], self.inspect_stop, emit=record)
                return source, selected["command_version"]

            def connected(result):
                source, command_version = result
                self.imported_frame = self.imported_path = None
                self.source_origin.set(f"Configuration source: online; command version {command_version}")
                self.show_source(cfg, source)
                self.status.set(
                    "Configuration received. Select channels. Inspection socket closed; source must reconnect on Start."
                    if cfg["source"]["mode"] == "listen"
                    else "Configuration received. Select scalar channels before starting."
                )
                self.update_plan()

            self.background(read_source, connected)
        except Exception as exc:
            self.show_error(str(exc))

    def show_source(self, cfg, source):
        self.source_config, self.source_signature = source, self.signature(cfg)
        self.tree.delete(*self.tree.get_children())
        channel_info = channels(source)
        for pmu in source.pmus:
            parent = f"pmu:{pmu.id_code}"
            self.tree.insert("", "end", iid=parent, text=f"PMU {pmu.id_code} · {pmu.station}", open=True)
            for ch in channel_info:
                if ch["source_pmu"] == pmu.id_code:
                    self.tree.insert(
                        parent, "end", iid=ch["id"],
                        text=("[x] " if ch["id"] in self.selected else "[ ] ") + ch["name"],
                        values=(ch["id"], ch["unit"]),
                    )
        missing = self.selected - {ch["id"] for ch in channel_info}
        self.source_note.set(
            f"CFG-2 version {source.version}: {len(source.pmus)} PMUs, interval {source.interval}s. "
            + (f"Missing saved channels: {sorted(missing)}" if missing else "")
        )
        self.update_plan()

    def import_configuration(self):
        path = filedialog.askopenfilename(
            parent=self.root, title="Import CFG-2 hexadecimal frame",
            filetypes=[("Hexadecimal CFG-2", "*.hex"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            cfg = self.gather()
            validate(cfg, require_weights=False, require_channels=False)

            def imported(frame):
                self.imported_frame, self.imported_path = frame, Path(path)
                self.source_origin.set(f"Configuration source: imported {Path(path).name} (this session only)")
                self.show_source(cfg, parse_config(frame))
                self.status.set(
                    "CFG-2 imported. Select channels and Start to receive online DATA. "
                    "Initial CFG-2 request skipped; command version unverified. Import is not saved in TOML."
                )

            self.status.set("Validating imported CFG-2...")
            self.background(lambda: load_test_config_frame(path, cfg["source"]["id_code"]), imported)
        except Exception as exc:
            self.show_error(str(exc))

    def clear_import(self):
        self.imported_frame = self.imported_path = None
        self.source_config = self.source_signature = None
        self.tree.delete(*self.tree.get_children())
        self.source_origin.set("Configuration source: online")
        self.source_note.set(f"{len(self.selected)} saved selections; read or import CFG-2 to view channels.")
        self.status.set("Import cleared. Start will request CFG-2 from the online source.")
        self.update_plan()

    def validate_import(self, cfg):
        if self.imported_frame is not None and self.source_signature != self.signature(cfg):
            raise ValueError("Source settings changed. Re-import the CFG-2 file or clear import before starting.")

    def check(self):
        try:
            cfg = self.gather()
            self.validate_import(cfg)

            def check_config():
                notes = validate(cfg)
                if self.source_config and self.source_signature == self.signature(cfg):
                    sample_plan(cfg, self.source_config.interval, self.source_config.time_base)
                return "Check passed; no model executed. " + " ".join(notes)

            self.background(check_config, lambda notes: self.status.set(notes))
        except Exception as exc:
            self.show_error(str(exc))

    def start(self):
        try:
            cfg = self.gather()
            self.validate_import(cfg)
            if self.source_config and self.source_signature == self.signature(cfg):
                sample_plan(cfg, self.source_config.interval, self.source_config.time_base)
            self.runtime = (
                Runtime(cfg, _test_config_frame=self.imported_frame)
                if self.imported_frame is not None else Runtime(cfg)
            )
            self.background(self.runtime.start, lambda _: None)
        except Exception as exc:
            self.show_error(str(exc))

    def stop(self):
        if self.inspecting:
            self.inspect_stop.set()
            self.status.set("Cancelling configuration read...")
            return
        if self.runtime:
            self.runtime.request_stop()
            self.status.set("Stopping worker, sockets and file writer...")

    def load_config(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("TOML", "*.toml")])
        if path:
            try:
                cfg = load(path)
                self.clear_import()
                self.apply(cfg)
                self.source_note.set(
                    f"Loaded {len(self.selected)} saved selections; read or import CFG-2 to view channels."
                )
            except Exception as exc:
                self.show_error(str(exc))

    def save_config(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".toml", filetypes=[("TOML", "*.toml")]
        )
        if path:
            try:
                Path(path).write_text(dumps(self.gather()), encoding="utf-8")
                self.status.set(
                    f"Saved {path}" + ("; imported CFG-2 is not saved in TOML." if self.imported_frame else "")
                )
            except Exception as exc:
                self.show_error(str(exc))

    def show_error(self, text):
        self.status.set(text)
        messagebox.showerror("PMU-FM", text, parent=self.root)

    def poll(self):
        try:
            while True:
                callback, value, error = self.messages.get_nowait()
                self.busy = False
                cancelled = self.inspecting and self.inspect_stop.is_set()
                self.inspecting = False
                if cancelled:
                    self.status.set("Configuration read cancelled; local socket closed.")
                elif error:
                    self.show_error(error)
                else:
                    callback(value)
        except queue.Empty:
            pass
        running = self.runtime and self.runtime.snapshot()["running"]
        self.set_locked(self.busy or bool(running))
        if self.runtime and (running or self.was_running) and not self.busy and not self.inspecting:
            s = self.runtime.snapshot()
            counts = s["counters"]
            self.status.set(
                f"{s['phase']} · input {s.get('input_state', 'stopped')} · "
                f"CMD v{s.get('input_command_version') or '?'} · model {s['worker']}\n"
                f"Input freshness {s.get('input_freshness_seconds', 'n/a')} s · frame version {s.get('source_frame_version', 'n/a')}\n"
                f"Window {s['window_samples']} samples · last inference {s['last_inference_seconds']} s\n"
                f"Forecasts {counts.get('forecast_completed', 0)} · skipped/replaced {counts.get('window_skipped', 0) + counts.get('pending_replaced', 0)} · errors {counts.get('inference_error', 0)}\n"
                f"TCP sends {counts.get('sent_to_socket', 0)} · file rows {s.get('file_rows', 0)} · file dropped {s.get('file_dropped', 0)}\n"
                f"Last error: {s['last_error'] or 'none'}"
            )
        self.was_running = bool(running)
        if self.closing and not running and not self.busy:
            self.root.destroy()
            return
        self.root.after(200, self.poll)

    def on_close(self):
        if self.runtime and self.runtime.snapshot()["running"]:
            if not messagebox.askyesno(
                "Stop run?",
                "Stop the active run, close sockets and flush records before closing?",
                parent=self.root,
            ):
                return
            self.stop()
        self.inspect_stop.set()
        self.closing = True
        if not self.busy and not (self.runtime and self.runtime.snapshot()["running"]):
            self.root.destroy()


def launch(path=None):
    root = tk.Tk()
    App(root, load(path) if path else None)
    root.mainloop()
