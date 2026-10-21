"""Independent live TCP source: literal IEEE CFG-2/DATA layout + third-party CRC.

No pmu_fm imports. It supports either TCP role, one/two PMUs and frame versions
1/2. Fresh synthetic values only; not a product input or replay implementation.
"""

import math
import socket
import struct
import threading
import time
from fractions import Fraction

from pySynphasor.crc import crc16xmodem


class ExternalSource:
    def __init__(
        self, *, target_port=None, version=1, rate=20, pmu55=False, millisecond_time=False,
        accepted_command_version=None, data_before_config=True, require_start=False,
        close_on_wrong_version=False,
    ):
        self.version, self.rate, self.pmu55 = version, rate, pmu55
        self.millisecond_time = millisecond_time
        self.accepted_command_version = accepted_command_version
        self.data_before_config, self.require_start = data_before_config, require_start
        self.close_on_wrong_version = close_on_wrong_version
        self.connections = 0
        self.accepted_commands = []
        self.ident = 55 if pmu55 else 1
        self.target_port = target_port
        self.listener = self.sock = None
        self.port = target_port
        if target_port is None:
            self.listener = socket.socket()
            self.listener.bind(("127.0.0.1", 0))
            self.listener.listen(1)
            self.listener.settimeout(0.1)
            self.port = self.listener.getsockname()[1]
        self.stop = threading.Event()
        self.commands, self.command_versions, self.errors = [], [], []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def frame(self, kind, payload, stamp):
        if self.millisecond_time and kind == 0:
            stamp = Fraction(round(stamp * 1000), 1000)
        soc = int(stamp)
        ticks = round((stamp - soc) * 1_000_000)
        if ticks == 1_000_000:
            soc, ticks = soc + 1, 0
        body = (
            struct.pack(">BBHHII", 0xAA, kind * 16 + self.version, 16 + len(payload), self.ident, soc, ticks)
            + payload
        )
        return body + struct.pack(">H", crc16xmodem(body, 0xFFFF))

    def configuration(self):
        payload = struct.pack(">IH", 1_000_000, 1 if self.pmu55 else 2)
        for ident, nominal in [(55, 0)] if self.pmu55 else [(10, 0), (20, 1)]:
            payload += f"PMU{ident}".encode().ljust(16, b" ")
            payload += struct.pack(">HHHHH", ident, 11 if self.pmu55 else 8, 1 if self.pmu55 else 0, 0, 0)
            if self.pmu55:
                payload += b"Phasor_V       ".ljust(16, b" ") + struct.pack(">I", 0)
            payload += struct.pack(">HH", nominal, 0)
        payload += struct.pack(">h", self.rate)
        return self.frame(3, payload, Fraction(time.time_ns(), 10**9))

    def data(self, number, start):
        payload = struct.pack(">H", 0)
        if self.pmu55:
            payload += struct.pack(">ff", 120.0, 0.1)
        payload += struct.pack(">ff", 60 + 0.03 * math.sin(number * 0.1), 0.01)
        if not self.pmu55:
            payload += struct.pack(">Hff", 0, 50.01, -0.02)
        return self.frame(0, payload, start + Fraction(number, self.rate))

    def run(self):
        while not self.stop.is_set():
            try:
                if self.listener:
                    self.sock, _ = self.listener.accept()
                else:
                    self.sock = socket.create_connection(("127.0.0.1", self.target_port), timeout=0.2)
                self.connections += 1
                self.stream()
            except OSError:
                pass  # disconnect/retry is the expected inspection/start lifecycle
            except Exception as exc:
                self.errors.append(repr(exc))
                return
            finally:
                if self.sock:
                    self.sock.close()
                    self.sock = None
            self.stop.wait(0.05)

    def stream(self):
        self.sock.settimeout(0.005)
        pending = bytearray()
        start = Fraction(time.time_ns() // 1_000_000, 1000)
        deadline, number = time.monotonic(), 0
        configured, enabled = False, not self.require_start
        while not self.stop.is_set():
            try:
                raw = self.sock.recv(4096)
                if not raw:
                    return
                pending.extend(raw)
                while len(pending) >= 4:
                    size = struct.unpack(">H", pending[2:4])[0]
                    assert size == 18, "Unexpected command frame length"
                    if len(pending) < size:
                        break
                    frame = bytes(pending[:size])
                    del pending[:size]
                    assert frame[0] == 0xAA and frame[1] >> 4 == 4
                    assert frame[1] & 15 in (1, 2)
                    assert struct.unpack(">H", frame[4:6])[0] == self.ident
                    assert crc16xmodem(frame[:-2], 0xFFFF) == int.from_bytes(frame[-2:], "big")
                    command = struct.unpack(">H", frame[14:16])[0]
                    self.commands.append(command)
                    self.command_versions.append(frame[1] & 15)
                    if self.accepted_command_version is not None and frame[1] & 15 != self.accepted_command_version:
                        if self.close_on_wrong_version:
                            return
                        continue
                    self.accepted_commands.append((frame[1] & 15, command))
                    if command == 5:
                        self.sock.sendall(self.configuration())
                        configured = True
                    elif command == 2:
                        enabled = True
            except socket.timeout:
                pass
            if enabled and (self.data_before_config or configured) and time.monotonic() >= deadline:
                self.sock.sendall(self.data(number, start))
                number += 1
                deadline += 1 / self.rate
            elif not enabled or not (self.data_before_config or configured):
                # No backlog burst after a command starts the test source.
                start = Fraction(time.time_ns() // 1_000_000, 1000)
                deadline = time.monotonic()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        sock = self.sock
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.listener:
            self.listener.close()
        self.thread.join(2)
        assert not self.thread.is_alive()
        assert not self.errors, self.errors
