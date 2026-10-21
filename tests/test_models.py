import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pmu_fm.config import DEFAULTS
from pmu_fm.models import LocalModel
from tests.helpers import REAL_MODELS


@pytest.mark.real_model
@pytest.mark.parametrize(
    "backend,variable", REAL_MODELS
)
def test_real_local_model(backend, variable, monkeypatch):
    directory = os.environ.get(variable)
    if not directory or not (Path(directory) / "model.safetensors").is_file():
        pytest.skip(f"{variable} not set to an existing local checkpoint; real model NOT verified")
    settings = deepcopy(DEFAULTS["model"])
    settings.update(backend=backend, local_dir=directory, revision=Path(directory).name)
    # Fail if the supposedly local adapter attempts a connection.
    import socket

    monkeypatch.setattr(
        socket.socket, "connect", lambda *_a, **_k: pytest.fail("Local model attempted network I/O")
    )
    model = LocalModel(settings)
    try:
        info = model.load()
        values = np.column_stack([60 + 0.03 * np.sin(np.arange(32) / 4), np.linspace(50, 50.01, 32)])
        for _ in range(2):
            prediction = model.predict(values, 4)
            assert prediction.shape == (4, 2) and np.isfinite(prediction).all()
        assert len(info["weights_sha256"]) == 64 and info["point_statistic"] == "median"
        assert info["channel_mode"] == "independent_univariate"
        # Channel order must survive batching, without cross-channel information leakage.
        original = model.predict(values, 5)
        swapped = model.predict(values[:, ::-1], 5)
        np.testing.assert_allclose(swapped[:, ::-1], original, rtol=1e-5, atol=1e-5)
        changed = values.copy()
        changed[:, 1] *= 3
        np.testing.assert_allclose(model.predict(changed, 5)[:, 0], original[:, 0], rtol=1e-5, atol=1e-5)
    finally:
        model.close()
