"""Trusted user scripts and validated, bounded DataFrame contract."""

from __future__ import annotations

import copy
import hashlib
import math
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ConfigError, timestamp_tolerance
from .protocol import ns


def load_script(path, content=None):
    if not path:
        return None, None
    content = Path(path).read_bytes() if content is None else content
    if len(content) > 1_000_000:
        raise ConfigError("Trusted script exceeds 1 MB source limit")
    module = types.ModuleType("pmu_fm_user_script")
    module.__file__ = str(path)
    exec(compile(content, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "preprocess", None)):
        raise ConfigError("User script must define preprocess(data, metadata)")
    return module, hashlib.sha256(content).hexdigest()


def input_frame(window, channel_info):
    columns = list(window.channel_ids)
    frame = pd.DataFrame(
        [[s.values.get(c, float("nan")) if s else float("nan") for c in columns] for s in window.samples],
        columns=columns,
        index=pd.to_datetime([ns(t) for t in window.timestamps], unit="ns", utc=True),
    )
    metadata = {
        "channels": {c: copy.deepcopy(channel_info[c]) for c in columns},
        "interval_seconds": str(window.interval),
        "source_timestamp_tolerance_seconds": str(window.tolerance),
        "source_max_timestamp_deviation_seconds": str(window.max_timestamp_deviation),
        "sampling_policy": "Nominal spacing within bounded tolerance; original source timestamps retained",
        "quality": "valid source window"
        if np.isfinite(frame.to_numpy()).all()
        else "Limited missing input; explicit script repair required",
        "wire_time": [
            {
                "soc": s.soc,
                "ticks": s.ticks,
                "time_base": s.time_base,
                "received_ns": s.received_ns,
                "stat": s.statuses,
                "time_quality": s.time_quality,
                "digital": s.digital,
                "frame_version": s.version,
            }
            if s
            else None
            for s in window.samples
        ],
    }
    return frame, metadata


def validate_frame(frame, metadata, expected_count, dt, channels_cap, start_ns, ref_ns):
    if not isinstance(frame, pd.DataFrame) or frame.shape[0] != expected_count:
        raise ConfigError(f"Script must return a DataFrame with {expected_count} rows")
    if not 1 <= frame.shape[1] <= channels_cap or not frame.columns.is_unique:
        raise ConfigError("Script returned duplicate/too many/zero channels")
    if any(not isinstance(c, str) or len(c) > 160 for c in frame.columns):
        raise ConfigError("Script channel IDs must be strings of at most 160 characters")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ConfigError("Script must return a timezone-aware UTC DatetimeIndex")
    if str(frame.index.tz) != "UTC" or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ConfigError("Script timestamps must be UTC, strictly increasing and unique")
    stamps = frame.index.as_unit("ns").asi8
    if stamps[0] < start_ns or stamps[-1] != ref_ns:
        raise ConfigError(
            "Script cannot add future/earlier data or change t_ref; resampling must end at t_ref"
        )
    expected = np.array(
        [ref_ns - round((expected_count - 1 - i) * dt * 1_000_000_000) for i in range(expected_count)],
        dtype=np.int64,
    )
    # Input may carry explicitly permitted timestamp rounding. Future targets use 1 ns only.
    if not isinstance(metadata, dict):
        raise ConfigError("Script metadata must be a dictionary")
    tolerance = int(metadata.get("timestamp_tolerance_ns", 1))
    if np.any(np.abs(stamps - expected) > tolerance):
        raise ConfigError("Script returned an irregular interval or changed the configured input interval")
    if not isinstance(metadata.get("channels"), dict):
        raise ConfigError("Script metadata.channels is required")
    if set(metadata["channels"]) != set(frame.columns):
        raise ConfigError("Script columns and metadata.channels must match exactly")
    if Fraction(str(metadata.get("interval_seconds", "0"))) != dt or not metadata.get("quality"):
        raise ConfigError("Script must declare actual interval_seconds and quality description")
    for c in frame.columns:
        ch = metadata["channels"][c]
        if not isinstance(ch, dict) or not all(k in ch for k in ("name", "unit", "source_pmu")):
            raise ConfigError(f"Script metadata for {c} requires name, unit and source_pmu")
        if not all(isinstance(ch[k], str) and 0 < len(ch[k]) <= 160 for k in ("name", "unit")):
            raise ConfigError(f"Invalid name/unit for {c}")
        if (
            not pd.api.types.is_numeric_dtype(frame[c])
            or pd.api.types.is_complex_dtype(frame[c])
            or pd.api.types.is_bool_dtype(frame[c])
        ):
            raise ConfigError("Script output must contain real numeric scalars")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ConfigError("Script output contains NaN or infinity")
    metadata["channels"] = {c: metadata["channels"][c] for c in frame.columns}
    return frame, metadata


def preprocess(window, channel_info, module, plan, cap):
    frame, metadata = input_frame(window, channel_info)
    original_frame = frame.copy(deep=True)
    tolerance = max(
        1, math.ceil(timestamp_tolerance(window.tolerance, min(window.interval, plan["dt"])) * 1_000_000_000)
    )
    metadata["timestamp_tolerance_ns"] = tolerance
    original = copy.deepcopy(metadata)
    if module:
        frame, metadata = module.preprocess(frame.copy(deep=True), copy.deepcopy(metadata))
    if isinstance(metadata, dict):
        metadata["timestamp_tolerance_ns"] = tolerance  # script cannot weaken time validation
        for key in (
            "source_timestamp_tolerance_seconds",
            "source_max_timestamp_deviation_seconds",
            "sampling_policy",
        ):
            metadata[key] = original[key]
    frame, metadata = validate_frame(
        frame, metadata, plan["context"], plan["dt"], cap, ns(window.start), ns(window.reference.timestamp)
    )
    original_sources = {info["source_pmu"] for info in original["channels"].values()}
    for c, ch in metadata["channels"].items():
        if ch["source_pmu"] not in original_sources:
            raise ConfigError("Derived channel source_pmu must refer to a selected source PMU")
        # Any transformed channel must explicitly describe its physical meaning.
        if module and c not in original["channels"] and not ch.get("transform"):
            raise ConfigError("Derived channels require metadata.channels[id].transform")
        if (
            module
            and c in original_frame
            and not frame[c].equals(original_frame[c])
            and not ch.get("transform")
        ):
            raise ConfigError("Changed values require an explicit transform description in channel metadata")
        if ch.get("normalized") and not callable(getattr(module, "inverse_transform", None)):
            if ch["unit"] != "dimensionless" or c in original["channels"]:
                raise ConfigError(
                    "Normalization without inverse requires a new derived ID and dimensionless unit"
                )
    return frame, metadata


def inverse(points, future_index, metadata, module, plan, cap):
    frame = pd.DataFrame(points, index=future_index, columns=list(metadata["channels"]))
    metadata = copy.deepcopy(metadata)
    metadata["timestamp_tolerance_ns"] = 1
    if module and callable(getattr(module, "inverse_transform", None)):
        frame, metadata = module.inverse_transform(frame.copy(deep=True), copy.deepcopy(metadata))
    if isinstance(metadata, dict):
        metadata["timestamp_tolerance_ns"] = 1  # input uncertainty cannot move forecast targets
    return validate_frame(
        frame,
        metadata,
        plan["horizon"],
        plan["dt"],
        cap,
        int(future_index[0].value),
        int(future_index[-1].value),
    )
