"""Test-only online source and fake model. These are not product modes."""

import math
import socket
import threading
import time
from fractions import Fraction

from pmu_fm.protocol import REQUEST_CFG2, Configuration, Framer, PMU, config_frame, data_frame, parse_command


REAL_MODELS = [
    ("chronos2", "PMU_FM_CHRONOS_DIR"),
    ("timesfm25", "PMU_FM_TIMESFM_DIR"),
    ("timestral", "PMU_FM_TIMESTRAL_DIR"),
    ("timesfm30", "PMU_FM_TIMESFM3_DIR"),
]


class FakeModel:
    """Clearly labeled test double; never reachable from product config/CLI."""

    delay = 0

    def __init__(self, settings):
        self.settings = settings

    def load(self):
        return {
            "model_id": "TEST_ONLY_FAKE",
            "package_version": "test",
            "revision": "test",
            "weights_sha256": "not-real-weights",
            "point_statistic": "test-last-value",
        }

    def predict(self, values, horizon):
        import numpy as np

        time.sleep(self.delay)
        return np.tile(values[-1], (horizon, 1))

    def close(self):
        pass


class SlowModel(FakeModel):
    delay = 0.7


class HungModel(FakeModel):
    delay = 60


class Simulator:
    def __init__(
        self,
        multi=True,
        rate=20,
        drop=(),
        duplicate=(),
        reorder=False,
        version=1,
        target_port=None,
        require_start=False,
    ):
        pmus = (PMU("TEST_A", 10, 12, analog_names=("AN",), anunits=(1,)),)
        if multi:
            pmus += (PMU("TEST_B", 20, 0, nominal_hz=50),)
        self.cfg = Configuration(1, 1_000_000, pmus, rate, version)
        self.target_port, self.require_start = target_port, require_start
        self.command_versions = []
        self.drop, self.duplicate, self.reorder = set(drop), set(duplicate), reorder
        self.commands, self.connections = [], 0
        self.stop = threading.Event()
        self.sock = None
        self.port = target_port
        if target_port is None:
            self.sock = socket.socket()
            self.sock.bind(("127.0.0.1", 0))
            self.port = self.sock.getsockname()[1]
            self.sock.listen(4)
            self.sock.settimeout(0.1)
        self.clients = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def run(self):
        while not self.stop.is_set():
            if self.target_port is not None:
                try:
                    conn = socket.create_connection(("127.0.0.1", self.target_port), timeout=0.2)
                except OSError:
                    self.stop.wait(0.05)
                    continue
                self.connections += 1
                self.clients.append((conn, self.thread))
                self.client(conn)
                self.stop.wait(0.05)
                continue
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connections += 1
            thread = threading.Thread(target=self.client, args=(conn,), daemon=True)
            self.clients.append((conn, thread))
            thread.start()

    def client(self, conn):
        conn.settimeout(0.005)
        framer, count = Framer(), 0
        start = Fraction(time.time_ns() // 1_000_000 * 1_000_000, 1_000_000_000)
        deadline, cfg = time.monotonic(), self.cfg
        held = None
        enabled = not self.require_start
        try:
            while not self.stop.is_set():
                try:
                    raw = conn.recv(4096)
                    if not raw:
                        return
                    for frame in framer.feed(raw):
                        cmd = parse_command(frame)
                        self.commands.append(cmd)
                        self.command_versions.append(frame.version)
                        if cmd == REQUEST_CFG2:
                            conn.sendall(config_frame(self.cfg))
                        if cmd == 2:
                            enabled = True
                except socket.timeout:
                    pass
                if cfg != self.cfg:
                    cfg = self.cfg
                    conn.sendall(config_frame(cfg))
                if not enabled:
                    deadline = time.monotonic()
                    continue
                if time.monotonic() < deadline:
                    continue
                f = 60 + 0.03 * math.sin(count * 0.1)
                blocks = [{"stat": 0, "freq": f, "dfreq": 0.01, "analog": [count % 20]}]
                if len(cfg.pmus) > 1:
                    blocks.append({"stat": 0, "freq": 50.01, "dfreq": -0.02})
                frame = data_frame(cfg, start + count * cfg.interval, blocks)
                if count not in self.drop:
                    if self.reorder and count % 10 == 0:
                        held = frame
                    else:
                        conn.sendall(frame)
                        if held:
                            conn.sendall(held)
                            held = None
                    if count in self.duplicate:
                        conn.sendall(frame)
                count += 1
                deadline += float(cfg.interval)
        except OSError:
            pass
        finally:
            conn.close()

    def disconnect_clients(self):
        for conn, _ in self.clients:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def __exit__(self, *_):
        self.stop.set()
        if self.sock:
            self.sock.close()
        self.disconnect_clients()
        self.thread.join(1)
        for _, thread in self.clients:
            thread.join(1)


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.03)
    raise AssertionError("Condition not met before timeout")
