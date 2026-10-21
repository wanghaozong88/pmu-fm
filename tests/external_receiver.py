"""Independent TCP receiver, using pySynphasor's unmodified packet/CRC implementation.

This file intentionally imports no pmu_fm module. pySynphasor 0.1.0 has global
format variables, so this test receiver supports uniform PMU block layouts only.
"""

import socket
import struct
import time
from fractions import Fraction

import pySynphasor.synphasor as independent
from pySynphasor.crc import crc16xmodem
from synchrophasor.frame import CommandFrame


class ExternalReceiver:
    def __init__(self, host, port, ident, version=1, command_version=None):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.ident, self.cfg = ident, None
        self.frames = []
        self.version = version
        self.command_version = version if command_version is None else command_version

    def command(self, cmd):
        # pySynphasor's builder initializes CRC to zero. Use the independent
        # pyPMU command builder (correct C37.118 initial value 0xFFFF) instead.
        frame = CommandFrame(self.ident, cmd, soc=int(time.time()))
        frame.set_version(self.command_version)
        self.sock.sendall(frame.convert2bytes())

    def read_exact(self, count):
        data = b""
        while len(data) < count:
            chunk = self.sock.recv(count - len(data))
            if not chunk:
                raise ConnectionError("TCP source disconnected")
            data += chunk
        return data

    def receive(self):
        header = self.read_exact(4)
        size = struct.unpack(">H", header[2:4])[0]
        assert 16 <= size <= 65535
        raw = header + self.read_exact(size - 4)
        assert crc16xmodem(raw[:-2], 0xFFFF) == int.from_bytes(raw[-2:], "big"), "Independent CRC failure"
        packet = independent.synphasor(raw)
        assert packet.version == self.version and packet.idcode == self.ident and packet.framesize == len(raw)
        self.frames.append(raw)
        if packet.type == 3:
            self.cfg = packet[independent.synphasor_cfg2]
            layouts = {(p.phnmr, p.annmr, p.dgnmr, p.phasor_dtype, p.analog, p.freq) for p in self.cfg.pmu}
            assert len(layouts) == 1, "Independent receiver supports uniform block layouts only"
            first = self.cfg.pmu[0]
            independent.pmu_count = self.cfg.num_pmu
            independent.phasor_count, independent.analog_count, independent.digital_count = (
                first.phnmr,
                first.annmr,
                first.dgnmr,
            )
            independent.phasor_type, independent.analog_type, independent.freq_type = (
                first.phasor_dtype,
                first.analog,
                first.freq,
            )
        return packet

    def forecast(self):
        while True:
            packet = self.receive()
            if packet.type == 0:
                assert self.cfg is not None
                timestamp = Fraction(packet.soc) + Fraction(packet.fracsec & 0xFFFFFF, self.cfg.TIME_BASE)
                return timestamp, packet[independent.synphasor_data].pmu_data

    def close(self):
        self.sock.close()
