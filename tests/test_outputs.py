import csv
import queue
import socket
import threading
import time
from collections import Counter
from copy import deepcopy
from fractions import Fraction

from pmu_fm.config import DEFAULTS
from pmu_fm.network import Publisher
from pmu_fm.outputs import FileWriter
from pmu_fm.protocol import Configuration, Framer, PMU, START, command_frame, data_frame
from pmu_fm.records import Forecast
from tests.helpers import FakeModel, wait_for
from tests.test_runtime import sample


def test_one_slow_client_does_not_block_another():
    settings = deepcopy(DEFAULTS["network"])
    settings.update(port=0, queue_size=8, send_timeout_seconds=0.3)
    counters = Counter()

    def emit(kind, *_a, **_kw):
        counters[kind] += 1

    pub = Publisher(settings, emit)
    cfg = Configuration(
        settings["id_code"],
        1_000_000,
        (PMU("Derived", 50001, 12, analog_names=tuple(f"A{i}" for i in range(1000)), anunits=(1,) * 1000),),
        100,
    )
    pub.configure(cfg)
    slow, fast = socket.socket(), socket.socket()
    slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256)
    slow.connect(("127.0.0.1", pub.port))
    fast.connect(("127.0.0.1", pub.port))
    fast.settimeout(0.1)
    stop, received = threading.Event(), []

    def drain():
        framer = Framer()
        while not stop.is_set():
            try:
                raw = fast.recv(65536)
                if not raw:
                    break
                received.extend(framer.feed(raw))
            except socket.timeout:
                pass
            except OSError:
                break

    thread = threading.Thread(target=drain)
    thread.start()
    try:
        for client in (slow, fast):
            client.sendall(command_frame(settings["id_code"], START))
        wait_for(lambda: len(pub.clients) == 2 and all(c.enabled for c in pub.clients))
        # Let the independent reader drain the large CFG before beginning the load.
        wait_for(lambda: any(f.kind == 3 for f in received))
        for i in range(300):
            raw = data_frame(
                cfg,
                Fraction(100) + Fraction(i, 100),
                [{"stat": 0x0200, "freq": 60, "dfreq": 0.01, "analog": [float(i)] * 1000}],
            )
            pub.publish(raw, f"test-{i}")
            time.sleep(0.005)
        wait_for(lambda: counters["slow_client"] + counters["client_error"] > 0)
        assert sum(f.kind == 0 for f in received) > 50
        wait_for(lambda: any(f.kind == 0 and f.soc >= 102 for f in received))
    finally:
        stop.set()
        slow.close()
        fast.close()
        pub.close()
        thread.join(2)


def test_disk_full_reported_and_writer_queue_bounded(tmp_path, monkeypatch):
    settings = deepcopy(DEFAULTS["files"])
    events = []

    def fail(*_a, **_kw):
        raise OSError(28, "No space left on device (injected)")

    monkeypatch.setattr(csv.DictWriter, "writerow", fail)
    writer = FileWriter(tmp_path / "run", settings, lambda *a, **k: events.append((a, k)))
    forecast = Forecast(
        "run",
        "id",
        FakeModel({}).load(),
        sample(100),
        Fraction(99),
        Fraction(1),
        time.time_ns(),
        {"10/freq": {"name": "Frequency", "unit": "Hz", "source_pmu": 10}},
        [[60.1]],
        "valid",
        0.1,
    )
    writer.offer("forecast", forecast)
    wait_for(lambda: writer.failed)
    writer.close()
    assert any(a[0] == "file_error" and "space" in a[1] for a, _ in events)
    # Exercise the actual nonblocking enqueue path against a saturated queue.
    writer.failed = False
    writer.stop_event.clear()
    writer.queue = queue.Queue(maxsize=1)
    assert writer.offer("event", {})
    assert not writer.offer("event", {})
    assert writer.dropped == 1
