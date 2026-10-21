"""Append a causal median channel for each selected voltage magnitude.

Keep every original column and timestamp. Each window starts with the available
history (1, 2, 3, 4 points), then uses the current point and four previous points.
No future samples, padding, missing-value repair, or state across windows.
"""

import numpy as np


WINDOW = 5


def preprocess(data, metadata):
    voltage_channels = [
        channel
        for channel in data.columns
        if channel.endswith("/magnitude") and metadata["channels"][channel]["unit"] == "V"
    ]
    if not voltage_channels:
        raise ValueError("Select at least one voltage phasor magnitude channel (unit V)")

    for channel in voltage_channels:
        if not np.isfinite(data[channel].to_numpy()).all():
            raise ValueError(f"{channel}: median filter requires finite input; missing repair is disabled")
        derived = f"{channel}_median{WINDOW}"
        if derived in data.columns:
            raise ValueError(f"Derived channel already exists: {derived}")
        data[derived] = data[channel].rolling(window=WINDOW, min_periods=1, center=False).median()
        info = metadata["channels"][channel]
        metadata["channels"][derived] = {
            **info,
            "id": derived,
            "name": f"{info['name']} median ({WINDOW})",
            "unit": "V",
            "source_channel": channel,
            "transform": (
                f"Causal trailing {WINDOW}-sample median; current and previous samples only; "
                "available-history warmup at each window start; no missing repair"
            ),
        }
    metadata["quality"] += (
        f"; original channels retained; derived voltage magnitude: trailing {WINDOW}-sample median "
        "with available-history warmup at each window start"
    )
    return data, metadata
