"""C37.118 CFG-2/DATA subset, accepting frame versions 1 and 2. No CFG-3."""

from __future__ import annotations

import binascii
import math
import struct
import time
from dataclasses import asdict, dataclass
from fractions import Fraction

DATA, HEADER, CFG1, CFG2, COMMAND = 0, 1, 2, 3, 4
STOP, START, REQUEST_HEADER, REQUEST_CFG1, REQUEST_CFG2 = 1, 2, 3, 4, 5


class ProtocolError(ValueError):
    pass


def ns(value: Fraction) -> int:
    return round(value * 1_000_000_000)


def now() -> Fraction:
    return Fraction(time.time_ns(), 1_000_000_000)


@dataclass(frozen=True)
class Frame:
    kind: int
    id_code: int
    soc: int
    ticks: int
    time_quality: int
    payload: bytes
    version: int = 1

    def timestamp(self, time_base: int) -> Fraction:
        if not 0 <= self.ticks < time_base:
            raise ProtocolError("FRACSEC is outside TIME_BASE")
        return Fraction(self.soc) + Fraction(self.ticks, time_base)


def encode(
    kind: int,
    id_code: int,
    timestamp: Fraction,
    time_base: int,
    payload: bytes,
    time_quality: int = 0,
    *,
    version: int = 1,
) -> bytes:
    if type(version) is not int or version not in (1, 2):
        raise ProtocolError("Supported frame versions are 1 and 2")
    if kind not in (DATA, HEADER, CFG1, CFG2, COMMAND):
        raise ProtocolError("Unsupported frame type (CFG-3/2024 not supported)")
    if not 1 <= id_code <= 65534 or not 1 <= time_base <= 0xFFFFFF:
        raise ProtocolError("IDCODE or TIME_BASE out of range")
    soc = timestamp.numerator // timestamp.denominator
    ticks = round((timestamp - soc) * time_base)
    if ticks == time_base:
        soc, ticks = soc + 1, 0
    size = 16 + len(payload)
    if size > 65535 or not 0 <= soc <= 0xFFFFFFFF:
        raise ProtocolError("Frame length or SOC out of range")
    body = (
        struct.pack(">BBHHII", 0xAA, (kind << 4) | version, size, id_code, soc, (time_quality << 24) | ticks)
        + payload
    )
    return body + struct.pack(">H", binascii.crc_hqx(body, 0xFFFF))


def decode(raw: bytes) -> Frame:
    if len(raw) < 16 or raw[0] != 0xAA or raw[1] & 0x80:
        raise ProtocolError("Invalid C37.118 sync/reserved bit")
    version = raw[1] & 0x0F
    if version not in (1, 2):
        raise ProtocolError(f"Unsupported C37.118 frame version {version}; expected 1 or 2")
    _, typ, size, id_code, soc, fracsec = struct.unpack(">BBHHII", raw[:14])
    if size != len(raw) or not 1 <= id_code <= 65534:
        raise ProtocolError("Invalid frame size or IDCODE")
    if binascii.crc_hqx(raw[:-2], 0xFFFF) != int.from_bytes(raw[-2:], "big"):
        raise ProtocolError("CRC mismatch")
    kind = (typ >> 4) & 7
    if kind not in (DATA, HEADER, CFG1, CFG2, COMMAND):
        raise ProtocolError("Unsupported frame type (CFG-3/2024 not supported)")
    return Frame(kind, id_code, soc, fracsec & 0xFFFFFF, fracsec >> 24, raw[14:-2], version)


class Framer:
    """Bounded incremental TCP reassembly. Invalid frames close the connection."""

    def __init__(self):
        self.pending = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self.pending.extend(data)
        result = []
        while len(self.pending) >= 4:
            if self.pending[0] != 0xAA:
                raise ProtocolError("Invalid sync byte")
            size = int.from_bytes(self.pending[2:4], "big")
            if size < 16:
                raise ProtocolError("Invalid frame length")
            if len(self.pending) < size:
                break
            result.append(decode(bytes(self.pending[:size])))
            del self.pending[:size]
        return result


class Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, size: int) -> bytes:
        if self.pos + size > len(self.data):
            raise ProtocolError("Truncated payload")
        out = self.data[self.pos : self.pos + size]
        self.pos += size
        return out

    def unpack(self, fmt: str):
        return struct.unpack(">" + fmt, self.take(struct.calcsize(">" + fmt)))

    def finish(self):
        if self.pos != len(self.data):
            raise ProtocolError("Unexpected trailing fields")


def label(value: str) -> bytes:
    raw = value.encode("ascii", errors="strict")
    if len(raw) > 16:
        raise ProtocolError("C37.118 labels must fit 16 ASCII bytes")
    return raw.ljust(16, b" ")


@dataclass(frozen=True)
class PMU:
    station: str
    id_code: int
    format: int
    phasor_names: tuple[str, ...] = ()
    analog_names: tuple[str, ...] = ()
    digital_names: tuple[str, ...] = ()
    phunits: tuple[int, ...] = ()
    anunits: tuple[int, ...] = ()
    digunits: tuple[int, ...] = ()
    nominal_hz: int = 60
    config_count: int = 1


@dataclass(frozen=True)
class Configuration:
    id_code: int
    time_base: int
    pmus: tuple[PMU, ...]
    data_rate: int
    version: int = 1

    @property
    def interval(self) -> Fraction:
        return Fraction(1, self.data_rate) if self.data_rate > 0 else Fraction(-self.data_rate)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if type(self.version) is not int or self.version not in (1, 2):
            raise ProtocolError("Supported frame versions are 1 and 2")
        if not 1 <= self.time_base <= 0xFFFFFF or not self.data_rate or not self.pmus:
            raise ProtocolError("Invalid time base, report rate or empty configuration")
        if not -32768 <= self.data_rate <= 32767:
            raise ProtocolError("DATA_RATE out of range")
        if len({p.id_code for p in self.pmus}) != len(self.pmus):
            raise ProtocolError("Duplicate PMU IDs")
        for p in self.pmus:
            if not 1 <= p.id_code <= 65534 or p.format & ~15 or p.nominal_hz not in (50, 60):
                raise ProtocolError("Unsupported PMU ID, FORMAT or FNOM")
            if len(p.phunits) != len(p.phasor_names) or len(p.anunits) != len(p.analog_names):
                raise ProtocolError("Channel/unit counts do not match")
            if len(p.digital_names) != 16 * len(p.digunits):
                raise ProtocolError("Digital words require 16 channel names each")
            if any(u >> 24 not in (0, 1) for u in p.phunits):
                raise ProtocolError("Unsupported phasor type")


def config_frame(cfg: Configuration, timestamp: Fraction | None = None) -> bytes:
    cfg.validate()
    payload = struct.pack(">IH", cfg.time_base, len(cfg.pmus))
    for p in cfg.pmus:
        payload += label(p.station) + struct.pack(
            ">HHHHH", p.id_code, p.format, len(p.phasor_names), len(p.analog_names), len(p.digunits)
        )
        payload += b"".join(label(n) for n in p.phasor_names + p.analog_names + p.digital_names)
        payload += b"".join(struct.pack(">I", u) for u in p.phunits + p.anunits + p.digunits)
        payload += struct.pack(">HH", int(p.nominal_hz == 50), p.config_count)
    payload += struct.pack(">h", cfg.data_rate)
    return encode(
        CFG2,
        cfg.id_code,
        now() if timestamp is None else timestamp,
        cfg.time_base,
        payload,
        version=cfg.version,
    )


def parse_config(frame: Frame) -> Configuration:
    if frame.kind != CFG2:
        raise ProtocolError("CFG-2 is required")
    r = Reader(frame.payload)
    time_base, count = r.unpack("IH")
    if time_base & 0xFF000000:
        raise ProtocolError("Reserved TIME_BASE bits set")
    pmus = []
    for _ in range(count):
        station = r.take(16).decode("ascii", errors="replace").rstrip(" \x00")
        ident, fmt, pn, an, dn = r.unpack("HHHHH")
        names = tuple(
            r.take(16).decode("ascii", errors="replace").rstrip(" \x00") for _ in range(pn + an + 16 * dn)
        )
        phunits = tuple(r.unpack("I")[0] for _ in range(pn))
        anunits = tuple(r.unpack("I")[0] for _ in range(an))
        digunits = tuple(r.unpack("I")[0] for _ in range(dn))
        fnom, cc = r.unpack("HH")
        if fnom & ~1:
            raise ProtocolError("Unsupported FNOM flags")
        pmus.append(
            PMU(
                station,
                ident,
                fmt,
                names[:pn],
                names[pn : pn + an],
                names[pn + an :],
                phunits,
                anunits,
                digunits,
                50 if fnom else 60,
                cc,
            )
        )
    cfg = Configuration(frame.id_code, time_base, tuple(pmus), r.unpack("h")[0], frame.version)
    r.finish()
    cfg.validate()
    return cfg


def channel_id(pmu: int, field: str, index: int | None = None, component: str = "") -> str:
    return (
        f"{pmu}/{field}" + (f"/{index}" if index is not None else "") + (f"/{component}" if component else "")
    )


def channels(cfg: Configuration) -> list[dict]:
    result = []
    for p in cfg.pmus:
        common = {"source_pmu": p.id_code, "station": p.station, "nominal_hz": p.nominal_hz}
        for field, name, unit in (("freq", "Frequency", "Hz"), ("dfreq", "ROCOF", "Hz/s")):
            result.append(common | {"id": channel_id(p.id_code, field), "name": name, "unit": unit})
        for i, name in enumerate(p.phasor_names):
            unit = "A" if p.phunits[i] >> 24 else "V"
            for part in ("magnitude", "angle", "real", "imag"):
                result.append(
                    common
                    | {
                        "id": channel_id(p.id_code, "phasor", i, part),
                        "name": f"{name} {part}",
                        "unit": "rad" if part == "angle" else unit,
                    }
                )
        for i, name in enumerate(p.analog_names):
            result.append(
                common
                | {
                    "id": channel_id(p.id_code, "analog", i),
                    "name": name,
                    "unit": "source-defined",
                    "anunit": p.anunits[i],
                }
            )
    return result


@dataclass(frozen=True)
class Sample:
    timestamp: Fraction
    received_ns: int
    values: dict[str, float]
    statuses: dict[int, int]
    time_quality: int
    soc: int
    ticks: int
    time_base: int
    digital: dict[int, tuple[int, ...]]
    version: int = 1

    def valid(self, source_pmus) -> bool:
        # Strict default: UTC locked, no leap transition, invalid/error/unlocked flags.
        return self.time_quality == 0 and all(
            not (self.statuses.get(p, 0xE000) & 0xE030) and self.statuses.get(p, 0xE000) & 0x01C0 != 0x01C0
            for p in source_pmus
        )


def _signed24(value: int) -> int:
    value &= 0xFFFFFF
    return value - (1 << 24) if value & (1 << 23) else value


def parse_data(frame: Frame, cfg: Configuration, received_ns: int | None = None) -> Sample:
    if frame.kind != DATA or frame.id_code != cfg.id_code:
        raise ProtocolError("DATA frame ID does not match configuration")
    r = Reader(frame.payload)
    values, statuses, digital = {}, {}, {}
    for p in cfg.pmus:
        statuses[p.id_code] = r.unpack("H")[0]
        for i, unit in enumerate(p.phunits):
            floating, polar = bool(p.format & 2), bool(p.format & 1)
            a, b = r.unpack("ff" if floating else ("Hh" if polar else "hh"))
            if not floating:
                if not polar:
                    a = math.nan if a == -32768 else a
                    b = math.nan if b == -32768 else b
                elif b == -32768:
                    b = math.nan
                scale = (unit & 0xFFFFFF) * 1e-5
                a *= scale
                b *= 1e-4 if polar else scale
            if polar:
                mag, ang = a, b
                real, imag = a * math.cos(b), a * math.sin(b)
            else:
                real, imag = a, b
                mag, ang = math.hypot(a, b), math.atan2(b, a)
            for part, val in zip(("real", "imag", "magnitude", "angle"), (real, imag, mag, ang)):
                values[channel_id(p.id_code, "phasor", i, part)] = val
        freq, dfreq = r.unpack("ff" if p.format & 8 else "hh")
        if not p.format & 8:
            freq = math.nan if freq == -32768 else p.nominal_hz + freq * 0.001
            dfreq = math.nan if dfreq == -32768 else dfreq * 0.01
        values[channel_id(p.id_code, "freq")] = freq
        values[channel_id(p.id_code, "dfreq")] = dfreq
        for i, unit in enumerate(p.anunits):
            val = r.unpack("f" if p.format & 4 else "h")[0]
            if not p.format & 4:
                val *= _signed24(unit)
            values[channel_id(p.id_code, "analog", i)] = val
        digital[p.id_code] = tuple(r.unpack("H")[0] for _ in p.digunits)
    r.finish()
    return Sample(
        frame.timestamp(cfg.time_base),
        time.time_ns() if received_ns is None else received_ns,
        values,
        statuses,
        frame.time_quality,
        frame.soc,
        frame.ticks,
        cfg.time_base,
        digital,
        frame.version,
    )


def data_frame(cfg: Configuration, timestamp: Fraction, blocks: list[dict], time_quality: int = 0) -> bytes:
    """Encode physical units. Used by the output and test-only simulator."""
    if len(blocks) != len(cfg.pmus):
        raise ProtocolError("PMU block count mismatch")
    payload = bytearray()
    for p, block in zip(cfg.pmus, blocks):
        payload.extend(struct.pack(">H", block["stat"]))
        phasors, analogs = block.get("phasors", []), block.get("analog", [])
        if len(phasors) != len(p.phunits) or len(analogs) != len(p.anunits):
            raise ProtocolError("Data channel count mismatch")
        for (a, b), unit in zip(phasors, p.phunits):
            if p.format & 2:
                payload.extend(struct.pack(">ff", a, b))
            else:
                scale = (unit & 0xFFFFFF) * 1e-5
                payload.extend(
                    struct.pack(
                        ">Hh" if p.format & 1 else ">hh",
                        round(a / scale),
                        round(b / (1e-4 if p.format & 1 else scale)),
                    )
                )
        f, df = block["freq"], block["dfreq"]
        if p.format & 8:
            payload.extend(struct.pack(">ff", f, df))
        else:
            payload.extend(struct.pack(">hh", round((f - p.nominal_hz) * 1000), round(df * 100)))
        for a, unit in zip(analogs, p.anunits):
            payload.extend(
                struct.pack(">f", a) if p.format & 4 else struct.pack(">h", round(a / _signed24(unit)))
            )
        dig = block.get("digital", [])
        if len(dig) != len(p.digunits):
            raise ProtocolError("Digital count mismatch")
        for word in dig:
            payload.extend(struct.pack(">H", word))
    return encode(
        DATA, cfg.id_code, timestamp, cfg.time_base, bytes(payload), time_quality, version=cfg.version
    )


def command_frame(id_code: int, command: int, *, version: int = 1) -> bytes:
    return encode(COMMAND, id_code, now(), 1_000_000, struct.pack(">H", command), version=version)


def parse_command(frame: Frame) -> int:
    if frame.kind != COMMAND or len(frame.payload) != 2:
        raise ProtocolError("Only basic two-byte commands are supported")
    return struct.unpack(">H", frame.payload)[0]
