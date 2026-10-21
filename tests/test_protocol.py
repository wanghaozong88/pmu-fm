from fractions import Fraction
import binascii
import math
import struct

import pytest

from pmu_fm.protocol import (
    Configuration,
    PMU,
    ProtocolError,
    Framer,
    config_frame,
    data_frame,
    decode,
    parse_config,
    parse_data,
)


@pytest.mark.parametrize("fmt", range(16))
@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("version", [1, 2])
def test_all_formats(fmt, multiple, version):
    p = PMU(
        "Test", 10, fmt, ("V",), ("A",), tuple(f"D{i}" for i in range(16)), (100000,), (2,), (0xFFFF,), 60
    )
    ps = (p, PMU("Other", 11, fmt, nominal_hz=50)) if multiple else (p,)
    cfg = Configuration(1, 1_000_000, ps, 30, version)
    assert parse_config(decode(config_frame(cfg))) == cfg
    t = Fraction(1_700_000_000) + Fraction(123457, 1_000_000)
    block = {
        "stat": 0,
        "phasors": [(100, 0.5) if fmt & 1 else (100, 50)],
        "freq": 59.983,
        "dfreq": -0.12,
        "analog": [20],
        "digital": [0x55],
    }
    blocks = [block] + ([{"stat": 0, "freq": 50.015, "dfreq": 0.34}] if multiple else [])
    sample = parse_data(decode(data_frame(cfg, t, blocks)), cfg)
    assert sample.version == version
    assert sample.timestamp == t
    assert sample.values["10/freq"] == pytest.approx(59.983, abs=3e-6)
    assert sample.values["10/dfreq"] == pytest.approx(-0.12)
    assert sample.values["10/analog/0"] == 20
    assert sample.values["10/phasor/0/angle"] == pytest.approx(0.5 if fmt & 1 else math.atan2(50, 100))
    if multiple:
        assert sample.values["11/freq"] == pytest.approx(50.015, abs=3e-6)


def test_independent_hand_built_integer_vector():
    # Literal wire payload, not the project's encoder: STAT, rectangular phasor,
    # FREQ=-25 mHz, DFREQ=12 cHz/s, ANALOG=-3.
    cfg = Configuration(
        7, 1_000_000, (PMU("Vector", 8, 0, ("I",), ("AN",), phunits=(0x010186A0,), anunits=(2,)),), 60
    )
    body = bytes.fromhex("aa02 001c 0007 6553f100 0001e241 0000 0003 fffc ffe7 000c fffd")
    raw = body + struct.pack(">H", binascii.crc_hqx(body, 0xFFFF))
    sample = parse_data(decode(raw), cfg)
    assert sample.values["8/freq"] == 59.975
    assert sample.values["8/dfreq"] == 0.12
    assert sample.values["8/analog/0"] == -6
    assert sample.values["8/phasor/0/magnitude"] == 5


def test_crc_partial_and_quality():
    cfg = Configuration(1, 10_000_000, (PMU("A", 2, 8),), 10)
    raw = data_frame(cfg, Fraction(100_000_001, 10_000_000), [{"stat": 0x2000, "freq": 60, "dfreq": 0}], 1)
    framer = Framer()
    frames = []
    for byte in raw:
        frames.extend(framer.feed(bytes([byte])))
    assert len(frames) == 1
    s = parse_data(frames[0], cfg)
    assert s.ticks == 1 and not s.valid([2])
    with pytest.raises(ProtocolError, match="CRC"):
        decode(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ProtocolError):
        parse_data(frames[0], Configuration(4, 10_000_000, cfg.pmus, 10))


@pytest.mark.parametrize("sync", [0x00, 0x03, 0x81, 0x82, 0x52])
def test_unsupported_version_reserved_bit_and_cfg3(sync):
    # Independent literal frame, with valid CRC so rejection must be semantic.
    body = bytes([0xAA, sync]) + bytes.fromhex("0010 0007 6553f100 00000000")
    raw = body + struct.pack(">H", binascii.crc_hqx(body, 0xFFFF))
    with pytest.raises(ProtocolError):
        decode(raw)


def test_cfg2_and_data_versions_are_separate_and_preserved():
    from dataclasses import replace
    from pmu_fm.protocol import command_frame, parse_command, encode, COMMAND

    cfg = Configuration(55, 1_000_000, (PMU("PMU55", 55, 8),), 30, 1)
    assert config_frame(cfg)[1] == 0x31
    raw = data_frame(replace(cfg, version=2), Fraction(10), [{"stat": 0, "freq": 60, "dfreq": 0}])
    assert parse_data(decode(raw), cfg).version == 2
    framer = Framer()
    frames = framer.feed(config_frame(cfg) + raw + command_frame(55, 5, version=2))
    assert [f.version for f in frames] == [1, 2, 2]
    assert parse_command(frames[-1]) == 5
    for version in (0, 3, True):
        with pytest.raises(ProtocolError):
            encode(COMMAND, 55, Fraction(10), 1_000_000, b"\0\5", version=version)
