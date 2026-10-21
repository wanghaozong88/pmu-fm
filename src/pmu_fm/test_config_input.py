"""Shared GUI/CLI loader for a captured CFG-2; never imports measurements."""

import re
from pathlib import Path

from .protocol import ProtocolError, decode, parse_config


def load_test_config_frame(path, id_code):
    # Bound allocation even for accidentally selected large recordings.
    with Path(path).open("rb") as stream:
        raw = stream.read(262_141)
    if len(raw) > 262_140:
        raise ProtocolError("Test configuration frame file is too large")
    text = raw.decode("utf-8-sig").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{2}(?:[-\s]*[0-9a-fA-F]{2})*", text):
        raise ProtocolError("Expected one hexadecimal CFG-2 frame (hyphens/whitespace allowed)")
    frame = decode(bytes.fromhex(re.sub(r"[-\s]", "", text)))
    parse_config(frame)
    if frame.id_code != id_code:
        raise ProtocolError("Imported CFG-2 ID differs from source.id_code")
    return frame
