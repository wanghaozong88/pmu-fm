"""Versioned TOML; static checking has no network or model side effects."""

from __future__ import annotations

import json
import math
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

DEFAULTS = {
    "schema_version": 1,
    "source": {
        "mode": "connect",
        "host": "127.0.0.1",
        "port": 4712,
        "id_code": 1,
        "allow_start": False,
        "command_version": "auto",
        "timeout_seconds": 5.0,
        "reconnect_seconds": 2.0,
        "reorder_seconds": 0.1,
        "timestamp_tolerance_seconds": 0.0,
        "channels": [],
        "queue_size": 1024,
    },
    "model": {
        "backend": "chronos2",
        "local_dir": "models/chronos-2",
        "revision": "local",
        "device": "cpu",
        "context_seconds": 4.0,
        "horizon_seconds": 1.0,
        "interval_seconds": 1.0,
        "input_interval_seconds": 0.0,
        "cpu_threads": 4,
        "load_timeout_seconds": 180.0,
        "timeout_seconds": 120.0,
        "memory_mb": 4096,
        "max_channels": 32,
    },
    "script": {"path": "", "timeout_seconds": 10.0, "max_missing_samples": 0},
    "network": {
        "enabled": True,
        "frame_version": 1,
        "host": "127.0.0.1",
        "port": 4713,
        "id_code": 50000,
        "pmu_id_start": 50001,
        "leads_seconds": [1.0],
        "max_clients": 8,
        "queue_size": 4,
        "send_timeout_seconds": 2.0,
    },
    "files": {
        "enabled": True,
        "directory": "runs",
        "queue_size": 64,
        "rows_per_segment": 100000,
        "flush_seconds": 1.0,
        "debug_windows": False,
    },
}


@dataclass(frozen=True)
class Backend:
    model_id: str
    package: str
    versions: tuple[str, ...]
    extra: str
    max_context: int
    notice: str = ""


BACKENDS = {
    "chronos2": Backend("amazon/chronos-2", "chronos-forecasting", ("2.3.2",), "chronos", 8192),
    "timesfm25": Backend(
        "google/timesfm-2.5-200m-pytorch", "timesfm", ("2.0.2", "3.0.2"), "timesfm", 16384
    ),
    "timestral": Backend(
        "mistralai/Timestral (5M / 28M / 123M / 209M)", "timestral", ("1.0.0",), "timestral", 32768
    ),
    "timesfm30": Backend(
        "google/timesfm-3.0-pytorch", "timesfm", ("3.0.2",), "timesfm3", 15360,
        "TimesFM-3 local weights: non-commercial, non-production use only.",
    ),
}


class ConfigError(ValueError):
    pass


def validate_checkpoint(settings: dict) -> tuple[Path, str]:
    """Check local artifacts and identify a supported architecture without importing ML packages."""
    backend = settings["backend"]
    if backend not in BACKENDS:
        raise ConfigError(f"Unsupported model.backend: {backend}")
    path = Path(settings["local_dir"]).resolve()
    if not path.is_dir() or not (path / "model.safetensors").is_file():
        raise ConfigError("model.local_dir must contain local model.safetensors; prepare weights separately")
    identity = BACKENDS[backend].model_id
    if backend == "timesfm25":
        return path, identity
    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"model.local_dir/config.json: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError("model.local_dir/config.json must contain an object")
    if backend == "chronos2":
        valid = cfg.get("architectures") == ["Chronos2Model"]
    elif backend == "timestral":
        # Published sizes share the same patching and prediction contract.
        sizes = {192: ("5M", 3, 512, 8), 384: ("28M", 6, 1024, 12),
                 640: ("123M", 10, 1712, 20), 768: ("209M", 12, 2048, 24)}
        size = sizes.get(cfg.get("d_model")) if type(cfg.get("d_model")) is int else None
        valid = bool(size) and all((
            cfg.get("num_heads") == size[1],
            cfg.get("mlp_hidden_dim") == size[2],
            cfg.get("block_schedule") == ["time", "time", "time", "time_variate"] * (size[3] // 4),
            cfg.get("patch_size") == 32,
            cfg.get("block_size") == 32,
            cfg.get("context_length") == 32768,
            cfg.get("quantile_knots") == [0.01] + [i / 20 for i in range(1, 20)] + [0.99],
        ))
        if valid:
            identity = f"mistralai/Timestral-{size[0]}"
    else:
        stack = cfg.get("transformer_config")
        transformer = stack.get("transformer") if isinstance(stack, dict) else None
        valid = isinstance(transformer, dict) and all((
            cfg.get("input_patch_len") == 32,
            cfg.get("output_patch_len") == 64,
            cfg.get("quantiles") == [i / 10 for i in range(1, 10)],
            stack.get("num_layers") == 20,
            transformer.get("model_dims") == 1280,
            transformer.get("num_heads") == 16,
            cfg.get("use_variate_attention") is True,
        ))
    if not valid:
        raise ConfigError(f"model.local_dir is not a supported {identity} checkpoint")
    return path, identity


def validate_model_window(backend: str, context: int, horizon: int):
    if backend not in BACKENDS:
        raise ConfigError(f"Unsupported model.backend: {backend}")
    limit = BACKENDS[backend].max_context
    if context < 2 or context > limit or not 1 <= horizon <= 1024:
        raise ConfigError(f"Model context/horizon exceeds supported limits: context 2..{limit}, horizon 1..1024")
    if backend == "timesfm25" and ((context + 31) // 32 * 32 + (horizon + 127) // 128 * 128 > 16384):
        raise ConfigError("TimesFM 2.5 patch-rounded context + horizon exceeds 16384")


def seconds(value) -> Fraction:
    return Fraction(str(value))


def timestamp_tolerance(configured, interval, time_base=None) -> Fraction:
    """Bound timing uncertainty independently of wire encoding resolution."""
    tolerance = seconds(configured)
    if time_base is not None:
        tolerance = max(tolerance, Fraction(1, time_base))
    if tolerance < 0 or tolerance * 4 >= interval:
        raise ConfigError(
            f"source.timestamp_tolerance_seconds={configured}: effective tolerance "
            f"{float(tolerance):.9g} s must be non-negative and less than 1/4 "
            f"of the shortest source/model sample interval ({float(interval):.9g} s); "
            f"required tolerance < {float(interval / 4):.9g} s. "
            "Reduce Timestamp tolerance in the Source tab. "
            "TIME_BASE sets the wire tick, not timestamp accuracy."
        )
    return tolerance


def steps(duration, interval: Fraction, name: str) -> int:
    number = seconds(duration) / interval
    if number.denominator != 1 or number <= 0:
        raise ConfigError(f"{name} must be a positive whole number of samples (interval={interval} s)")
    return int(number)


def report_rate(interval) -> int:
    interval = seconds(interval)
    rate = 1 / interval
    if rate.denominator == 1 and 1 <= rate <= 32767:
        return int(rate)
    if interval.denominator == 1 and 1 <= interval <= 32768:
        return -int(interval)
    raise ConfigError("model.interval_seconds cannot be represented by C37.118 DATA_RATE")


def normalize(raw: dict, base: Path | None = None) -> dict:
    cfg = deepcopy(DEFAULTS)
    if raw.get("schema_version", 1) != 1:
        raise ConfigError("schema_version must be 1")
    for section, value in raw.items():
        if section not in cfg:
            raise ConfigError(f"Unknown configuration section: {section}")
        if section == "schema_version":
            continue
        if not isinstance(value, dict):
            raise ConfigError(f"{section} must be a TOML table")
        for key, setting in value.items():
            if key not in cfg[section]:
                raise ConfigError(f"Unknown setting: {section}.{key}")
            expected = cfg[section][key]
            valid_type = (type(setting) is type(expected)) or (
                isinstance(expected, float) and type(setting) in (int, float)
            )
            if section == "source" and key == "command_version":
                if setting != "auto" and not (type(setting) is int and setting in (1, 2)):
                    raise ConfigError('source.command_version must be "auto", 1 or 2')
                valid_type = True
            if section == "model" and key == "input_interval_seconds" and isinstance(setting, str):
                try:
                    valid_type = seconds(setting) >= 0
                except (ValueError, ZeroDivisionError):
                    valid_type = False
            if not valid_type:
                raise ConfigError(f"{section}.{key}: expected {type(expected).__name__}")
            cfg[section][key] = setting
    if base is not None:
        for section, key in (("model", "local_dir"), ("script", "path"), ("files", "directory")):
            if cfg[section][key]:
                cfg[section][key] = str((base / cfg[section][key]).resolve())
    return cfg


def validate(cfg: dict, require_weights: bool = True, require_channels: bool = True) -> list[str]:
    cfg = normalize(cfg)
    s, m, n, f = (cfg[k] for k in ("source", "model", "network", "files"))
    if s["mode"] not in ("connect", "listen"):
        raise ConfigError("source.mode must be connect or listen")
    if n["frame_version"] not in (1, 2):
        raise ConfigError("network.frame_version must be 1 or 2")
    for section, settings in cfg.items():
        if not isinstance(settings, dict):
            continue
        for key, value in settings.items():
            if type(value) in (float, int) and (not math.isfinite(value) or value < 0):
                raise ConfigError(f"{section}.{key} must be finite and non-negative")
            if (
                key.endswith("seconds")
                and key
                not in (
                    "reorder_seconds",
                    "input_interval_seconds",
                    "leads_seconds",
                    "timestamp_tolerance_seconds",
                )
                or key
                in (
                    "queue_size",
                    "rows_per_segment",
                    "max_clients",
                    "cpu_threads",
                    "memory_mb",
                    "max_channels",
                )
            ):
                if value <= 0:
                    raise ConfigError(f"{section}.{key} must be positive")
    for section in ("source", "network"):
        if not 1 <= cfg[section]["port"] <= 65535 or not cfg[section]["host"].strip():
            raise ConfigError(f"{section}: host and port (1..65535) required")
        if not 1 <= cfg[section]["id_code"] <= 65534:
            raise ConfigError(f"{section}.id_code must be 1..65534")
    if not n["enabled"] and not f["enabled"]:
        raise ConfigError("Enable at least one output")
    if n["id_code"] == s["id_code"] or not 1 <= n["pmu_id_start"] <= 65534 - m["max_channels"]:
        raise ConfigError("network IDs must be distinct from source and fit all virtual PMUs")
    if (
        s["mode"] == "listen"
        and n["enabled"]
        and s["port"] == n["port"]
        and (s["host"] == n["host"] or "0.0.0.0" in (s["host"], n["host"]))
    ):
        raise ConfigError("Input listener and output listener must use different bind endpoints")
    if m["backend"] not in BACKENDS or m["device"] not in ("cpu", "cuda"):
        raise ConfigError(f"model.backend must be {'/'.join(BACKENDS)}; device must be cpu/cuda")
    if not m["revision"].strip():
        raise ConfigError("model.revision must identify the local weight version")
    if not isinstance(s["channels"], list) or any(not isinstance(c, str) for c in s["channels"]):
        raise ConfigError("source.channels must contain stable channel ID strings")
    if len(set(s["channels"])) != len(s["channels"]):
        raise ConfigError("source.channels contains duplicates")
    if require_channels and not s["channels"]:
        raise ConfigError("source.channels: connect and select at least one channel")
    if len(s["channels"]) > m["max_channels"] or m["max_channels"] > 128:
        raise ConfigError("Too many channels (maximum 128; default cap 32)")
    if n["enabled"]:
        report_rate(m["interval_seconds"])
        leads = n["leads_seconds"]
        if not leads or len(leads) > 8 or len(set(leads)) != len(leads):
            raise ConfigError("network.leads_seconds must contain 1..8 unique fixed leads")
        if any(
            type(h) not in (int, float) or not math.isfinite(h) or h <= 0 or h > m["horizon_seconds"]
            for h in leads
        ):
            raise ConfigError("network.leads_seconds must be within the prediction horizon")
    if cfg["script"]["path"]:
        p = Path(cfg["script"]["path"])
        if not p.is_file() or p.suffix.lower() != ".py":
            raise ConfigError("script.path must name an existing trusted .py file")
    if cfg["script"]["max_missing_samples"] > 32 or (
        cfg["script"]["max_missing_samples"] and not cfg["script"]["path"]
    ):
        raise ConfigError("script.max_missing_samples requires a script and must be 0..32")
    if require_weights:
        validate_checkpoint(m)
    notes = [
        "TCP input: "
        + ("listen for one source" if s["mode"] == "listen" else "connect to source")
        + "; accepts frame versions 1 and 2.",
        f"Upstream command version: {s['command_version']} (CFG-2 requests only during selection).",
        "Per-channel forecasting; no joint multivariate modeling.",
        "Upstream START permission enabled."
        if s["allow_start"]
        else "Upstream STOP is never sent; START disabled.",
    ]
    if BACKENDS[m["backend"]].notice:
        notes.append(BACKENDS[m["backend"]].notice)
    return notes


def sample_plan(cfg: dict, source_interval: Fraction, source_time_base: int | None = None) -> dict:
    m = cfg["model"]
    dt = seconds(m["input_interval_seconds"]) or source_interval
    timestamp_tolerance(
        cfg["source"]["timestamp_tolerance_seconds"], min(source_interval, dt), source_time_base
    )
    context = steps(m["context_seconds"], dt, "model.context_seconds")
    horizon = steps(m["horizon_seconds"], dt, "model.horizon_seconds")
    source_context = steps(m["context_seconds"], source_interval, "source context")
    steps(m["interval_seconds"], source_interval, "model.interval_seconds")
    if context < 2 or source_context < 2 or source_context > 65536:
        raise ConfigError("Context requires 2..65536 source samples and at least 2 model samples")
    validate_model_window(m["backend"], context, horizon)
    if dt != source_interval and not cfg["script"]["path"]:
        raise ConfigError("A preprocessing script is required to change the input interval")
    leads = (
        [steps(h, dt, "network lead") for h in cfg["network"]["leads_seconds"]]
        if cfg["network"]["enabled"]
        else []
    )
    return {
        "context": context,
        "source_context": source_context,
        "horizon": horizon,
        "dt": dt,
        "leads": leads,
    }


def load(path) -> dict:
    path = Path(path).resolve()
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        return normalize(raw, path.parent)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def dumps(cfg: dict) -> str:
    # Only the deliberately small schema is serialized, no general TOML framework.
    lines = ["schema_version = 1", ""]
    for section, settings in cfg.items():
        if section == "schema_version":
            continue
        lines.append(f"[{section}]")
        for key, value in settings.items():
            lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)
