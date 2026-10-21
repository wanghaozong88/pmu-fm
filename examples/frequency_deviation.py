"""Convert selected absolute Hz channels into explicitly named deviations."""


def preprocess(data, metadata):
    rename = {}
    for channel in list(data.columns):
        info = metadata["channels"][channel]
        if channel.endswith("/freq") and info["unit"] == "Hz":
            derived = channel + "_deviation"
            nominal = info["nominal_hz"]
            data[channel] = data[channel] - nominal
            rename[channel] = derived
            metadata["channels"][derived] = {
                **metadata["channels"].pop(channel),
                "id": derived,
                "name": f"Frequency deviation from {nominal} Hz",
                "unit": "Hz deviation",
                "transform": f"absolute Hz minus source nominal {nominal} Hz",
                "source_channel": channel,
            }
    metadata["quality"] = "Valid historical measurements; absolute frequency converted to nominal deviation"
    return data.rename(columns=rename), metadata
