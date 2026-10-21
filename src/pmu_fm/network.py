"""TCP I/O; no protocol command received downstream ever reaches upstream."""

from __future__ import annotations

import queue
import select
import socket
import threading
import time

from .protocol import (
    CFG2,
    COMMAND,
    DATA,
    HEADER,
    REQUEST_CFG2,
    REQUEST_HEADER,
    START,
    STOP,
    Framer,
    ProtocolError,
    command_frame,
    config_frame,
    encode,
    now,
    parse_command,
    parse_config,
    parse_data,
)


def close_socket(sock):
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


def listen_socket(host, port, backlog):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        option = socket.SO_EXCLUSIVEADDRUSE if hasattr(socket, "SO_EXCLUSIVEADDRUSE") else socket.SO_REUSEADDR
        sock.setsockopt(socket.SOL_SOCKET, option, 1)
        sock.bind((host, port))
        sock.listen(backlog)
        sock.settimeout(0.2)
        return sock
    except Exception:
        sock.close()
        raise


class SourceTransport:
    """One source socket in either TCP role; shared by inspection and running input."""

    def __init__(self, settings, stop, emit):
        self.settings, self.stop, self.emit = settings, stop, emit
        self.listener = self.sock = None
        self.peer, self.phase = None, "stopped"
        self.command_version = None
        self.framer, self.pending_frames = Framer(), []
        if settings.get("mode", "connect") == "listen":
            try:
                self.listener = listen_socket(settings["host"], settings["port"], 1)
            except OSError as exc:
                raise OSError(
                    f"Cannot bind source listener {settings['host']}:{settings['port']}: {exc}. "
                    "Choose an available local address/port; another receiver may already own it."
                ) from exc

    def connect(self, deadline=None):
        if self.stop.is_set():
            raise InterruptedError("Source connection cancelled")
        self.phase = "waiting_for_source" if self.listener else "connecting"
        self.emit(
            "input_waiting" if self.listener else "input_connecting",
            f"TCP {self.settings['host']}:{self.settings['port']}",
        )
        if self.listener:
            while not self.stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        "No source connected to the input listener before the inspection timeout"
                    )
                try:
                    self.sock, address = self.listener.accept()
                    self.peer = f"{address[0]}:{address[1]}"
                    break
                except socket.timeout:
                    continue
        else:
            self.sock = socket.create_connection(
                (self.settings["host"], self.settings["port"]),
                timeout=min(2, self.settings["timeout_seconds"]),
            )
            address = self.sock.getpeername()
            self.peer = f"{address[0]}:{address[1]}"
        if self.stop.is_set():
            self.disconnect()
            raise InterruptedError("Source connection cancelled")
        self.sock.settimeout(0.2)
        self.command_version = None
        self.framer, self.pending_frames = Framer(), []
        self.phase = "waiting_for_configuration"

    def command(self, command, *, version=None):
        if command == STOP:
            raise ValueError("Upstream STOP is prohibited")
        if self.stop.is_set():
            raise InterruptedError("Source operation cancelled")
        if version is not None and command != REQUEST_CFG2:
            raise ValueError("Only CFG-2 requests may probe a command version")
        if command == START and not self.settings.get("allow_start", False):
            raise ValueError("Upstream START requires explicit permission")
        version = self.command_version if version is None else version
        if version is None:
            raise ValueError("Read valid CFG-2 before sending other upstream commands")
        self.sock.sendall(
            command_frame(self.settings["id_code"], command, version=version)
        )

    def receive(self, timeout=0.2):
        if self.stop.is_set():
            raise InterruptedError("Source operation cancelled")
        sock, listener = self.sock, self.listener
        sock.settimeout(timeout)
        if listener:
            readable, _, _ = select.select([sock, listener], [], [], timeout)
            if listener in readable:
                extra, address = listener.accept()
                close_socket(extra)
                self.emit(
                    "input_client_rejected",
                    "One upstream is already connected",
                    peer=f"{address[0]}:{address[1]}",
                )
            if sock not in readable:
                raise socket.timeout()
        return sock.recv(65536)

    def read_frames(self, timeout=0.2):
        if self.stop.is_set():
            raise InterruptedError("Source operation cancelled")
        if self.pending_frames:
            frames, self.pending_frames = self.pending_frames, []
            return frames
        raw = self.receive(timeout)
        if not raw:
            raise ConnectionError("Source closed TCP connection")
        frames = self.framer.feed(raw)
        if any(f.id_code != self.settings["id_code"] for f in frames):
            raise ProtocolError("Source stream ID differs from source.id_code")
        return frames

    def read_configuration(self):
        """At most two CFG-2 probes, shared by inspection and running input.

        Incoming version is only a hint. Retry on a fresh socket so a delayed
        response to the first request cannot select the second command version.
        C37.118 has no request transaction ID: even a valid CFG-2 after a request
        is evidence for this choice, not proof of device-wide version support.
        """
        setting = self.settings.get("command_version", "auto")
        timeout = self.settings["timeout_seconds"]
        hint, unsolicited = None, False
        if setting == "auto":
            deadline = time.monotonic() + min(0.1, timeout / 5)
            while time.monotonic() < deadline:
                try:
                    frames = self.read_frames(min(0.2, max(0.001, deadline - time.monotonic())))
                except socket.timeout:
                    continue
                if frames:
                    hint = frames[0].version
                    for index, frame in enumerate(frames):
                        if frame.kind == CFG2:
                            parse_config(frame)
                            hint, unsolicited = frame.version, True
                            self.pending_frames = frames[index:]
                            break
                    break
        preferred = hint or 1
        versions = [preferred, 3 - preferred] if setting == "auto" else [setting]
        failures = []
        for attempt, version in enumerate(versions):
            if self.stop.is_set():
                raise InterruptedError("Source configuration read cancelled")
            deadline = time.monotonic() + timeout
            try:
                if attempt:
                    self.disconnect()
                    self.connect(deadline)
                self.phase = "waiting_for_configuration"
                self.emit(
                    "command_version_probe", f"Requesting CFG-2 with command version {version}",
                    command_version=version, mode=setting, attempt=attempt + 1,
                    received_version_hint=hint, peer=self.peer,
                )
                self.command(REQUEST_CFG2, version=version)
                while time.monotonic() < deadline:
                    try:
                        frames = self.read_frames(min(0.2, max(0.001, deadline - time.monotonic())))
                    except socket.timeout:
                        continue
                    for index, frame in enumerate(frames):
                        if frame.kind == CFG2:
                            cfg = parse_config(frame)
                            self.command_version = version
                            # Preserve coalesced DATA and partial TCP frames for the runtime.
                            self.pending_frames = frames[index:]
                            self.emit(
                                "command_version_selected",
                                f"Command version {version} selected; received CFG-2 version {cfg.version}",
                                command_version=version, configuration_version=cfg.version,
                                mode=setting, peer=self.peer,
                                basis=(
                                    "Unsolicited CFG-2; command version inferred, not verified"
                                    if unsolicited and attempt == 0
                                    else "Valid CFG-2 received after request; no protocol version negotiation"
                                ),
                            )
                            return cfg
                raise TimeoutError("No valid CFG-2 received within source timeout")
            except OSError as exc:
                if self.stop.is_set():
                    raise InterruptedError("Source configuration read cancelled") from exc
                failures.append(f"version {version}: {exc}")
                self.emit("command_version_attempt_failed", failures[-1], command_version=version)
        raise TimeoutError(
            "No valid CFG-2 after " + "; ".join(failures)
            + ". Check endpoint, stream ID and device/protocol compatibility; "
            "a timeout alone does not establish a version mismatch"
        )

    def disconnect(self):
        close_socket(self.sock)
        self.sock, self.peer = None, None
        self.command_version = None
        self.phase = "waiting_for_source" if self.listener else "reconnecting"

    def close(self):
        self.disconnect()
        close_socket(self.listener)
        self.listener = None
        self.phase = "stopped"


def inspect_source(settings: dict, stop: threading.Event | None = None, *, emit=None):
    """Request CFG-2 using the selected TCP role; close afterwards, never START/STOP."""
    stop = stop or threading.Event()
    timeout = settings["timeout_seconds"]
    if settings.get("mode", "connect") == "listen":
        timeout = max(30, timeout)
    transport = SourceTransport(settings, stop, emit or (lambda *_a, **_kw: None))
    try:
        transport.connect(time.monotonic() + timeout)
        return transport.read_configuration()
    finally:
        transport.close()


class Receiver:
    def __init__(self, settings: dict, events: queue.Queue, emit, *, _test_config_frame=None):
        self.settings, self.events, self.emit = settings, events, emit
        self.test_config_frame = _test_config_frame
        if _test_config_frame is not None:
            parse_config(_test_config_frame)
            if _test_config_frame.id_code != settings["id_code"]:
                raise ProtocolError("Imported CFG-2 ID differs from source.id_code")
        self.stop_event = threading.Event()
        self.transport = None
        self.thread = threading.Thread(target=self._run, name="pmu-input", daemon=True)
        self.last_data_mono = None

    def start(self):
        self.transport = SourceTransport(self.settings, self.stop_event, self.emit)
        self.thread.start()

    def _put(self, kind, value):
        try:
            self.events.put_nowait((kind, value))
        except queue.Full:
            self.emit("input_queue_overflow", "Input queue full; samples will have a visible gap")
            if kind != "sample":
                raise ConnectionError("Control event could not be queued; reconnecting")

    def _run(self):
        s = self.settings
        while not self.stop_event.is_set():
            try:
                self.transport.connect()
                self.emit("connected", "Source TCP connected; waiting for CFG-2", peer=self.transport.peer)
                if self.test_config_frame is None:
                    self.transport.read_configuration()
                else:
                    frame = self.test_config_frame
                    version = s.get("command_version", "auto")
                    self.transport.command_version = frame.version if version == "auto" else version
                    self.transport.pending_frames = [frame]
                    self.emit(
                        "test_configuration_imported",
                        "TEST ONLY: using imported CFG-2; command version is unverified",
                        configuration=parse_config(frame).to_dict(),
                        command_version=self.transport.command_version,
                        configuration_origin="test_import",
                    )
                cfg, flagged, awaiting = None, False, False
                start_sent = False
                last_data = time.monotonic()
                while not self.stop_event.is_set():
                    if time.monotonic() - last_data > s["timeout_seconds"]:
                        raise TimeoutError(
                            "Source data timeout (configuration/heartbeat is not measurement data)"
                        )
                    try:
                        frames = self.transport.read_frames()
                    except socket.timeout:
                        continue
                    for frame in frames:
                        if frame.id_code != s["id_code"]:
                            raise ProtocolError("Source stream ID changed")
                        if frame.kind == CFG2:
                            cfg = parse_config(frame)
                            self._put("config", cfg)
                            awaiting = False
                            self.transport.phase = "waiting_for_data"
                            if s["allow_start"] and not start_sent:
                                self.transport.command(START)
                                start_sent = True
                                self.emit(
                                    "upstream_start", "Explicitly permitted START sent after valid CFG-2"
                                )
                        elif frame.kind == DATA and cfg is not None and not awaiting:
                            sample = parse_data(frame, cfg)
                            changed = any(stat & 0x0400 for stat in sample.statuses.values())
                            if changed and not flagged:
                                self._put("config_pending", None)
                                self.transport.command(REQUEST_CFG2)
                                awaiting = True
                                self.transport.phase = "waiting_for_configuration"
                            else:
                                self._put("sample", sample)
                                self.last_data_mono = last_data = time.monotonic()
                                self.transport.phase = "receiving"
                            flagged = changed
            except (OSError, ValueError) as exc:
                if not self.stop_event.is_set():
                    self.emit("input_error", str(exc))
                    try:
                        self._put("disconnected", None)
                    except ConnectionError:
                        pass
            finally:
                self.transport.disconnect()
                self.last_data_mono = None
            if s.get("mode", "connect") == "connect":
                self.stop_event.wait(s["reconnect_seconds"])
        self.transport.close()

    def close(self):
        self.stop_event.set()
        if self.transport:
            # Keep references intact until the reader exits; select/accept may be active.
            close_socket(self.transport.sock)
            close_socket(self.transport.listener)
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
        if self.thread.is_alive():
            self.emit(
                "input_error",
                "OS name resolution/connection cleanup is still pending; no further commands will be sent",
            )


class Client:
    def __init__(self, server, sock, address):
        self.server, self.sock = server, sock
        self.ident = f"{address[0]}:{address[1]}"
        self.queue = queue.Queue(maxsize=server.settings["queue_size"])
        self.stop = threading.Event()
        self.enabled, self.want_config, self.config_sent = False, False, False
        self.thread = threading.Thread(target=self.run, name=f"pmu-output-{self.ident}", daemon=True)

    def offer(self, raw, forecast_id=None):
        try:
            self.queue.put_nowait((raw, forecast_id))
        except queue.Full:
            self.server.emit(
                "slow_client", "Client queue full; disconnecting", client=self.ident, forecast_id=forecast_id
            )
            self.stop.set()

    def run(self):
        pending, forecast_id, started = b"", None, 0
        self.sock.setblocking(False)
        framer = Framer()
        try:
            while not self.stop.is_set() and not self.server.stop_event.is_set():
                if self.want_config and not self.config_sent and self.server.cfg_bytes:
                    self.offer(self.server.cfg_bytes)
                    self.config_sent = True
                if not pending:
                    try:
                        pending, forecast_id = self.queue.get_nowait()
                        started = time.monotonic()
                    except queue.Empty:
                        pass
                if pending and time.monotonic() - started > self.server.settings["send_timeout_seconds"]:
                    raise TimeoutError("Client send timeout")
                readable, writable, _ = select.select([self.sock], [self.sock] if pending else [], [], 0.03)
                if readable:
                    raw = self.sock.recv(4096)
                    if not raw:
                        break
                    for frame in framer.feed(raw):
                        if frame.kind != COMMAND or frame.id_code != self.server.settings["id_code"]:
                            raise ProtocolError("Output expects commands for its own stream ID")
                        cmd = parse_command(frame)
                        if cmd == REQUEST_CFG2:
                            self.want_config, self.config_sent = True, False
                        elif cmd == REQUEST_HEADER:
                            header = b"PMU-FM derived fixed-lead forecast. Frame time=t_ref. See run mapping; not raw PMU measurements."
                            self.offer(
                                encode(
                                    HEADER,
                                    frame.id_code,
                                    now(),
                                    self.server.time_base,
                                    header,
                                    version=self.server.settings["frame_version"],
                                )
                            )
                        elif cmd == START:
                            self.enabled, self.want_config = True, True
                        elif cmd == STOP:
                            self.enabled = False
                            # Finish a partial frame to preserve TCP framing. Discard queued forecasts.
                            while True:
                                try:
                                    _, old_id = self.queue.get_nowait()
                                    if old_id:
                                        self.server.emit(
                                            "send_cancelled",
                                            "Downstream STOP",
                                            client=self.ident,
                                            forecast_id=old_id,
                                        )
                                except queue.Empty:
                                    break
                            self.config_sent = False
                        else:
                            self.server.emit("unsupported_command", str(cmd), client=self.ident)
                if writable and pending:
                    try:
                        count = self.sock.send(pending)
                    except BlockingIOError:
                        continue
                    if not count:
                        raise ConnectionError("Client socket closed during send")
                    pending = pending[count:]
                    if not pending and forecast_id:
                        self.server.emit(
                            "sent_to_socket",
                            "TCP send completed; receiver delivery unconfirmed",
                            client=self.ident,
                            forecast_id=forecast_id,
                        )
        except (OSError, ValueError) as exc:
            self.server.emit("client_error", str(exc), client=self.ident, forecast_id=forecast_id)
        finally:
            if pending and forecast_id:
                self.server.emit(
                    "send_failed",
                    "Connection closed with unsent data",
                    client=self.ident,
                    forecast_id=forecast_id,
                )
            while True:
                try:
                    _, old_id = self.queue.get_nowait()
                    if old_id:
                        self.server.emit(
                            "send_failed",
                            "Connection closed before send",
                            client=self.ident,
                            forecast_id=old_id,
                        )
                except queue.Empty:
                    break
            close_socket(self.sock)
            self.stop.set()


class Publisher:
    def __init__(self, settings, emit):
        self.settings, self.emit = settings, emit
        self.stop_event = threading.Event()
        self.clients, self.lock = [], threading.Lock()
        self.cfg_bytes = None
        self.time_base = 1_000_000
        self.sock = listen_socket(settings["host"], settings["port"], settings["max_clients"])
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._accept, name="pmu-output", daemon=True)
        self.thread.start()

    def _accept(self):
        while not self.stop_event.is_set():
            try:
                sock, address = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self.lock:
                self.clients = [c for c in self.clients if not c.stop.is_set()]
                if len(self.clients) >= self.settings["max_clients"]:
                    close_socket(sock)
                    self.emit("client_limit", "Connection limit reached")
                    continue
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
                client = Client(self, sock, address)
                self.clients.append(client)
                client.thread.start()

    def configure(self, cfg):
        if self.cfg_bytes is not None:
            raise RuntimeError("Published configuration is immutable; stop and restart to change it")
        self.cfg_bytes = config_frame(cfg)
        self.time_base = cfg.time_base

    def publish(self, raw: bytes, forecast_id: str):
        self.emit(
            "network_submitted", "Forecast offered to active output connections", forecast_id=forecast_id
        )
        with self.lock:
            active = [c for c in self.clients if c.enabled and not c.stop.is_set()]
            for client in active:
                if not client.config_sent:
                    client.offer(self.cfg_bytes)
                    client.config_sent = True
                client.offer(raw, forecast_id)
        if not active:
            self.emit("no_subscribers", "No active downstream clients", forecast_id=forecast_id)

    def close(self):
        self.stop_event.set()
        close_socket(self.sock)
        self.thread.join(timeout=1)
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            client.stop.set()
            close_socket(client.sock)
        for client in clients:
            client.thread.join(timeout=1)
