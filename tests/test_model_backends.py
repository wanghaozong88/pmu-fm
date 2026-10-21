"""Adapter contract tests with explicit test doubles; real weights use test_models.py."""

import contextlib
import hashlib
import importlib.metadata
import json
import socket
import sys
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from pmu_fm.cli import main
from pmu_fm.config import (
    BACKENDS, DEFAULTS, ConfigError, dumps, sample_plan, validate_checkpoint,
)
from pmu_fm.models import LocalModel


def checkpoint(tmp_path, backend, variant="209M"):
    # Test-only architecture signatures; no third-party checkpoint is distributed.
    if backend == "timestral":
        width, heads, mlp, layers = {
            "5M": (192, 3, 512, 8), "28M": (384, 6, 1024, 12),
            "123M": (640, 10, 1712, 20), "209M": (768, 12, 2048, 24),
        }[variant]
        config = {
            "d_model": width, "num_heads": heads, "mlp_hidden_dim": mlp,
            "block_schedule": ["time", "time", "time", "time_variate"] * (layers // 4),
            "patch_size": 32, "block_size": 32, "context_length": 32768,
            "quantile_knots": [0.01] + [i / 20 for i in range(1, 20)] + [0.99],
        }
    else:
        config = {
            "input_patch_len": 32, "output_patch_len": 64,
            "quantiles": [i / 10 for i in range(1, 10)], "use_variate_attention": True,
            "transformer_config": {"num_layers": 20, "transformer": {"model_dims": 1280, "num_heads": 16}},
        }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"TEST DOUBLE ONLY: not model weights")
    settings = deepcopy(DEFAULTS["model"])
    settings.update(backend=backend, local_dir=str(tmp_path))
    return settings


@pytest.mark.parametrize("variant", ["5M", "28M", "123M", "209M"])
def test_timestral_checkpoint_identity_from_architecture_signature(tmp_path, variant):
    settings = checkpoint(tmp_path, "timestral", variant)
    assert validate_checkpoint(settings) == (tmp_path, f"mistralai/Timestral-{variant}")


@pytest.mark.parametrize("backend", ["timestral", "timesfm30"])
def test_wrong_family_and_missing_artifacts_are_rejected(tmp_path, backend):
    settings = checkpoint(tmp_path, backend)
    assert validate_checkpoint(settings)[1].startswith(("mistralai/Timestral-", "google/timesfm-3"))
    settings["backend"] = "timesfm30" if backend == "timestral" else "timestral"
    with pytest.raises(ConfigError, match="not a supported"):
        validate_checkpoint(settings)
    settings["backend"] = backend
    (tmp_path / "config.json").unlink()
    with pytest.raises(ConfigError, match="config.json"):
        validate_checkpoint(settings)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ConfigError, match="prepare weights separately"):
        validate_checkpoint(settings)


@pytest.mark.parametrize("bad_config", ["null", "[]", "{}", '{"transformer_config": null}', "not json"])
@pytest.mark.parametrize("backend", ["timestral", "timesfm30"])
def test_malformed_checkpoint_config_is_actionable(tmp_path, backend, bad_config):
    settings = checkpoint(tmp_path, backend)
    (tmp_path / "config.json").write_text(bad_config)
    with pytest.raises(ConfigError):
        validate_checkpoint(settings)


@pytest.mark.parametrize("backend,limit", [("timestral", 32768), ("timesfm30", 15360)])
def test_window_limit_rejects_silent_truncation(backend, limit):
    cfg = deepcopy(DEFAULTS)
    cfg["model"].update(backend=backend, context_seconds=limit, horizon_seconds=1024)
    assert sample_plan(cfg, Fraction(1))["context"] == limit
    cfg["model"]["context_seconds"] = limit + 1
    with pytest.raises(ConfigError, match=str(limit)):
        sample_plan(cfg, Fraction(1))
    cfg["model"].update(context_seconds=32, horizon_seconds=1025)
    with pytest.raises(ConfigError, match="1024"):
        sample_plan(cfg, Fraction(1))


class ArrayTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


@pytest.fixture
def stub_libraries(monkeypatch):
    calls = []
    monkeypatch.setattr(socket.socket, "connect", lambda *_a, **_k: pytest.fail("Network forbidden"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="TEST DOUBLE", set_num_threads=lambda n: None,
        cuda=SimpleNamespace(is_available=lambda: False), inference_mode=contextlib.nullcontext,
    ))
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "1.0.0" if package == "timestral" else "3.0.2")

    class TimestralDouble:
        max_context = 32768

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("load", path, kwargs))
            return cls()

        def forecast(self, batch, prediction_length, **kwargs):
            calls.append(("forecast", batch.copy(), kwargs))
            median = batch[:, :, -1:] + np.arange(1, prediction_length + 1)
            return SimpleNamespace(median=ArrayTensor(median))

    class TimesFMDouble:
        global_context = 15360

        def __init__(self, config):
            calls.append(("load", config.checkpoint_path, vars(config)))
            self.model = SimpleNamespace(state_dict=lambda: {"test_weight": None})

        def predict_batch(self, contexts, horizon, **kwargs):
            calls.append(("forecast", contexts, kwargs))
            return iter(SimpleNamespace(forecast=c[-1] + np.arange(1, horizon + 1)) for c in contexts)

    monkeypatch.setitem(sys.modules, "timestral", SimpleNamespace(TimestralForecaster=TimestralDouble))
    monkeypatch.setitem(sys.modules, "timesfm3", SimpleNamespace(
        ModelConfig=SimpleNamespace, TimesFM3Forecaster=TimesFMDouble,
    ))
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(
        safe_open=lambda *_a, **_k: contextlib.nullcontext(SimpleNamespace(keys=lambda: ["test_weight"]))
    ))
    return calls


@pytest.mark.parametrize("backend", ["timestral", "timesfm30"])
def test_offline_loading_median_channel_order_and_bounded_batches(tmp_path, backend, stub_libraries):
    settings = checkpoint(tmp_path, backend)
    model = LocalModel(settings)
    info = model.load()
    load_call = stub_libraries[0]
    assert load_call[1] == str(tmp_path)
    assert load_call[2]["local_files_only"] is True and load_call[2]["device"] == "cpu"
    values = np.arange(33 * 10, dtype=float).reshape(33, 10)
    out = model.predict(values, 7)
    np.testing.assert_array_equal(out, values[-1][None, :] + np.arange(1, 8)[:, None])
    assert info["point_statistic"] == "median"
    assert info["channel_mode"] == "independent_univariate"
    assert info["config_sha256"] == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    assert info["nonnegative_clipping"] is False
    if backend == "timestral":
        assert [x[1].shape for x in stub_libraries[1:]] == [(8, 1, 33), (2, 1, 33)]
        assert all(x[2]["clip_nonnegative"] is False for x in stub_libraries[1:])
    else:
        assert load_call[2]["per_core_batch_size"] == 8
        assert all(c.ndim == 1 for c in stub_libraries[1][1])
        assert stub_libraries[1][2]["make_positive"] is False
        assert "non-production" in info["model_license_notice"]
    model.close()
    with pytest.raises(RuntimeError, match="not loaded"):
        model.predict(values, 7)


@pytest.mark.parametrize("backend", ["timestral", "timesfm30"])
def test_inference_rejects_invalid_inputs_and_backend_outputs(tmp_path, backend, stub_libraries):
    model = LocalModel(checkpoint(tmp_path, backend))
    model.load()
    for values in [np.ones(4), np.ones((4, 0)), np.ones((4, 33)), np.full((4, 2), np.nan),
                   np.ones((BACKENDS[backend].max_context + 1, 1))]:
        with pytest.raises(ConfigError):
            model.predict(values, 4)
    if backend == "timestral":
        model.model.forecast = lambda *_a, **_k: SimpleNamespace(median=ArrayTensor(np.ones((1, 4))))
    else:
        model.model.predict_batch = lambda *_a, **_k: iter([SimpleNamespace(forecast=np.ones((4, 1)))])
    with pytest.raises(RuntimeError, match="invalid predictions"):
        model.predict(np.ones((4, 1)), 4)


def test_timesfm30_rejects_old_dependency_before_model_import(tmp_path, monkeypatch):
    settings = checkpoint(tmp_path, "timesfm30")
    monkeypatch.setattr(importlib.metadata, "version", lambda _package: "2.0.2")
    with pytest.raises(ConfigError, match=r"Install pmu-fm\[timesfm3\]"):
        LocalModel(settings).load()


def test_timesfm30_rejects_partial_weights(tmp_path, stub_libraries, monkeypatch):
    import safetensors

    monkeypatch.setattr(safetensors, "safe_open", lambda *_a, **_k: contextlib.nullcontext(
        SimpleNamespace(keys=lambda: [])
    ))
    with pytest.raises(ConfigError, match="1 missing"):
        LocalModel(checkpoint(tmp_path, "timesfm30")).load()


@pytest.mark.parametrize("backend", ["timestral", "timesfm30"])
def test_cli_new_model_checks_are_static(tmp_path, backend, monkeypatch, capsys):
    cfg = deepcopy(DEFAULTS)
    cfg["model"] = checkpoint(tmp_path, backend)
    cfg["source"]["channels"] = ["10/freq"]
    path = tmp_path / "config.toml"
    path.write_text(dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(LocalModel, "load", lambda _self: pytest.fail("Static check executed model"))
    monkeypatch.setattr(socket.socket, "connect", lambda *_a, **_k: pytest.fail("Static check connected"))
    assert main(["check", str(path)]) == 0
    text = capsys.readouterr().out
    assert "model execution not tested" in text
    if backend == "timesfm30":
        assert "non-commercial, non-production" in text
    # Config keeps the explicitly chosen backend on a save/load round trip.
    assert f'backend = "{backend}"' in path.read_text()


def test_wrong_timesfm_quantile_contract_rejected(tmp_path):
    settings = checkpoint(tmp_path, "timesfm30")
    path = tmp_path / "config.json"
    cfg = json.loads(path.read_text())
    cfg["quantiles"][4] = 0.55
    path.write_text(json.dumps(cfg))
    with pytest.raises(ConfigError, match="supported"):
        validate_checkpoint(settings)
