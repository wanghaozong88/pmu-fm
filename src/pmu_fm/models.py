"""Official local inference adapters; test doubles are confined to tests."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import os

from .config import BACKENDS, ConfigError, validate_checkpoint, validate_model_window


def enforce_offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tensor_keys(model, path):
    # TimesFM-3's Hub loader uses strict=False. Reject missing tensors before
    # inference so initialized parameters can never masquerade as loaded weights.
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        saved = set(handle.keys())
    expected = set(model.state_dict())
    if saved != expected:
        raise ConfigError(
            f"Incomplete or incompatible TimesFM-3 weights: "
            f"{len(expected - saved)} missing / {len(saved - expected)} unexpected tensors"
        )


class LocalModel:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.compiled_shape = None
        self.info = {}

    def load(self):
        enforce_offline()
        m = self.settings
        path, ident = validate_checkpoint(m)
        backend = BACKENDS[m["backend"]]
        try:
            version = importlib.metadata.version(backend.package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ConfigError(f"Install pmu-fm[{backend.extra}] in this Python environment") from exc
        if version not in backend.versions:
            raise ConfigError(
                f"{backend.package} {' / '.join(backend.versions)} required for {m['backend']}; "
                f"found {version}. Install pmu-fm[{backend.extra}]"
            )
        import torch

        if m["device"] not in ("cpu", "cuda"):
            raise ConfigError("model.device must be cpu/cuda")
        torch.set_num_threads(m["cpu_threads"])
        if m["device"] == "cuda" and not torch.cuda.is_available():
            raise ConfigError(
                "CUDA requested but unavailable; select CPU or install a compatible CUDA torch build"
            )
        if m["backend"] == "chronos2":
            from chronos import Chronos2Pipeline

            self.model = Chronos2Pipeline.from_pretrained(
                str(path), device_map=m["device"], local_files_only=True, torch_dtype=torch.float32
            )
        elif m["backend"] == "timesfm25":
            from timesfm import TimesFM_2p5_200M_torch

            self.model = TimesFM_2p5_200M_torch(torch_compile=False)
            # The pinned official implementation auto-selects CUDA in its constructor.
            # Set its explicit device before load_checkpoint moves any weight tensors.
            self.model.model.device = torch.device(m["device"])
            self.model.model.device_count = 1
            self.model.load_checkpoint(str(path), torch_compile=False)
        elif m["backend"] == "timestral":
            from timestral import TimestralForecaster

            self.model = TimestralForecaster.from_pretrained(
                str(path), device=m["device"], local_files_only=True
            )
        elif m["backend"] == "timesfm30":
            from timesfm3 import ModelConfig, TimesFM3Forecaster

            self.model = TimesFM3Forecaster(ModelConfig(
                checkpoint_path=str(path), device=m["device"], local_files_only=True,
                per_core_batch_size=min(m["max_channels"], 8),
            ))
            validate_tensor_keys(self.model.model, path / "model.safetensors")
        # Hash the actual local weights once per load, independently of a user revision label.
        self.info = {
            "model_id": ident,
            "backend": m["backend"],
            "package": backend.package,
            "package_version": version,
            "revision": m["revision"],
            "weights_sha256": file_sha256(path / "model.safetensors"),
            "torch": torch.__version__,
            "device": m["device"],
            "point_statistic": "median",
            "channel_mode": "independent_univariate",
        }
        if (path / "config.json").is_file():
            self.info["config_sha256"] = file_sha256(path / "config.json")
        if backend.notice:
            self.info["model_license_notice"] = backend.notice
        if m["backend"] in ("timestral", "timesfm30"):
            self.info["nonnegative_clipping"] = False
            self.info["symmetric_averaging"] = m["backend"] == "timestral"
        return self.info

    def predict(self, values, horizon):
        import numpy as np
        import torch

        if self.model is None:
            raise RuntimeError("Model is not loaded")
        values = np.asarray(values)
        if values.ndim != 2:
            raise ConfigError("Invalid model window or horizon")
        count, width = values.shape
        if not 1 <= width <= self.settings["max_channels"] or not np.isfinite(values).all():
            raise ConfigError("Invalid model window or channel count")
        validate_model_window(self.settings["backend"], count, horizon)
        contexts = [values[:, i].astype(np.float32, copy=True) for i in range(width)]
        if not all(np.isfinite(c).all() for c in contexts):
            raise ConfigError("Model input exceeds float32 range; apply an explicit normalization script")
        with torch.inference_mode():
            if self.settings["backend"] == "chronos2":
                if (
                    count > min(8192, self.model.model_context_length)
                    or horizon > self.model.model_prediction_length
                ):
                    raise ConfigError("Requested context/horizon exceeds local Chronos-2 checkpoint")
                _, points = self.model.predict_quantiles(
                    contexts,
                    prediction_length=horizon,
                    context_length=count,
                    quantile_levels=[0.5],
                    cross_learning=False,
                    limit_prediction_length=True,
                )
                out = np.stack([point.detach().cpu().numpy()[0] for point in points], axis=1)
            elif self.settings["backend"] == "timesfm25":
                from timesfm import ForecastConfig

                padded_context = (count + 31) // 32 * 32
                padded_horizon = (horizon + 127) // 128 * 128
                if padded_context + padded_horizon > 16384:
                    raise ConfigError("TimesFM patch-rounded context + horizon exceeds 16384")
                if self.compiled_shape != (count, horizon):
                    self.model.compile(
                        ForecastConfig(
                            max_context=padded_context,
                            max_horizon=padded_horizon,
                            per_core_batch_size=min(width, 8),
                            normalize_inputs=True,
                            use_continuous_quantile_head=True,
                            force_flip_invariance=True,
                            infer_is_positive=False,
                            fix_quantile_crossing=True,
                        )
                    )
                    self.compiled_shape = (count, horizon)
                points, _ = self.model.forecast(horizon=horizon, inputs=contexts)
                out = np.asarray(points).T
            elif self.settings["backend"] == "timestral":
                if count > self.model.max_context:
                    raise ConfigError("Requested context exceeds local Timestral checkpoint")
                batches = []
                for start in range(0, width, 8):
                    # Batch axis separates channels; the variate axis is always one.
                    batch = np.stack(contexts[start:start + 8])[:, None, :]
                    forecast = self.model.forecast(
                        batch, prediction_length=horizon, symmetric_averaging=True, clip_nonnegative=False
                    )
                    median = forecast.median.detach().cpu().numpy()
                    if median.shape != (len(batch), 1, horizon):
                        raise RuntimeError(f"Timestral returned invalid predictions: {median.shape}")
                    batches.append(median[:, 0, :].T)
                out = np.concatenate(batches, axis=1)
            else:  # timesfm30; backend was validated above
                if count > self.model.global_context:
                    raise ConfigError("Requested context exceeds local TimesFM-3 limit")
                predictions = list(self.model.predict_batch(
                    contexts, horizon=horizon, return_quantiles=False,
                    use_symmetric_averaging=False, make_positive=False,
                ))
                if len(predictions) != width or any(
                    np.shape(p.forecast) != (horizon,) for p in predictions
                ):
                    raise RuntimeError("TimesFM-3 returned invalid predictions")
                out = np.stack([p.forecast for p in predictions], axis=1)
        if out.shape != (horizon, width) or not np.isfinite(out).all():
            raise RuntimeError(f"Model returned invalid predictions: {out.shape}")
        return out.astype(np.float64)

    def close(self):
        self.model = None
        self.compiled_shape = None
        gc.collect()
