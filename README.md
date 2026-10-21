# PMU Foundation Model Bridge

**pmu-fm** connects online PMU/PDC data to local forecasting models, with a Tk GUI
and CLI sharing one runtime: **TCP input → per-channel forecasts → CSV/TCP output**.

## Quick start

Use **Python 3.12 with Tcl/Tk**. From the repository root on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[chronos]"
hf download amazon/chronos-2 --revision 0f8a440441931157957e2be1a9bce66627d99c76 --local-dir models/chronos-2
pmu-fm gui examples/config.toml
```

This explicitly downloads Chronos-2 for preparation; the application never downloads
weights automatically. Linux: use `python3`, `source .venv/bin/activate`, and install
your distribution's Tk package. For CUDA, replace the CPU wheel with the appropriate
[PyTorch 2.8.0 build](https://pytorch.org/get-started/previous-versions/#v280).

Choose one model extra; read its weight license before downloading:

| Model | Install extra / TOML backend | Official weights |
|---|---|---|
| Chronos-2 | `chronos` / `chronos2` | [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) |
| TimesFM 2.5 | `timesfm` / `timesfm25` | [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| Timestral | `timestral` / `timestral` | [mistralai/Timestral-209M](https://huggingface.co/mistralai/Timestral-209M) (also 5M/28M/123M) |
| TimesFM-3 | `timesfm3` / `timesfm30` | [google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch) |

Checkpoints need `model.safetensors` and, except for TimesFM 2.5, `config.json`.
Set the revision in the GUI; hashes and actual Timestral size are recorded.

1. Select **connect** or **listen**, enter address/port/stream ID, read configuration
   and select channels. A listening source must reconnect after inspection.
2. Select the model directory and `cpu`/`cuda`; set history, horizon and call interval.
   Optionally choose a trusted preprocessing script.
3. Enable CSV and/or network output, then **Check → Start**. Save as `*.local.toml`.
   **Stop** cleans up; closing an active window asks before stopping.

Output defaults to `127.0.0.1:4713`; remote clients need an explicit LAN bind address.
Upstream START requires opt-in; upstream STOP is never sent. Reconfigure while stopped.

```text
pmu-fm check examples/config.toml --schema-only
pmu-fm check my.local.toml
pmu-fm run my.local.toml
```

`check` is static; `--online` explicitly reads the source. Ctrl+C stops CLI runs.
TOML paths are relative to the configuration file.

## Examples

Edit the [configuration template](examples/config.toml),
[Timestral profile](examples/timestral.toml) or [TimesFM-3 profile](examples/timesfm3.toml)
for your endpoint, channels and weights. Scripts demonstrate [identity](examples/identity.py),
[frequency deviation](examples/frequency_deviation.py) and a
[causal voltage-magnitude median](examples/median_voltage.py).

Scripts implement `preprocess(data, metadata)` returning that pair on a copied UTC
DataFrame. Preserve reference/provenance and update channel, unit, interval and
transformation metadata. Optional `inverse_transform` preserves target times.
Scripts must be trusted: worker timeouts are **not a security sandbox**.

**No device?** Run `python -m tests.simulator` in another activated terminal at the
repository root. Connect the GUI to its printed loopback port, stream ID **1**, read
configuration and select frequency channels. It generates fresh synthetic online
data; inference still needs local weights. No DataStreamer is required.

## Semantics and scope

- Forecasts use history through `t_ref` for targets `t_ref + lead`. Network time is
  `t_ref`; CSV separates reference, target and completion times. Send events confirm
  local sends only. Per-run records include configuration, model metadata and mappings.
- Median forecasts use new logical identities and derived ANALOG channels;
  `output_mapping.json` explains them. FREQ/DFREQ remain actual reference-time
  measurements. CSV saves all leads. Invalid windows/overload leave gaps; no stale replay.
- TIME_BASE is wire resolution. Input rounding tolerance stays below 1/4 of both
  source/model intervals, preserves raw times and rejects ambiguity; targets stay exact.

Supports one TCP upstream (connect/listen), multiple PMU blocks, integer/float and
rectangular/polar inputs, and CFG-2/DATA with frame versions 1/2. Command version
`auto` tries at most two CFG-2 probes on separate connections; explicit 1/2 overrides
and an independent output version are available. Frame version differs from CFG type.
No CFG-3/full-2024 support, UDP, training, offline product input, cloud inference or
control actions. No automatic angle unwrap or universal vendor analog scaling.
This research tool has no protocol authentication/encryption or certification.

## Tests

```text
python -m pip install ".[test,interop]"
python -m pytest -q -rs
python -m pytest -q -m gui --run-gui
ruff check src tests
```

Tests use synthetic fixtures/sources and explicit test doubles; independent protocol
checks use pySynphasor/Scapy and pyPMU. For real models, install the extra, set
`PMU_FM_CHRONOS_DIR`, `PMU_FM_TIMESFM_DIR`, `PMU_FM_TIMESTRAL_DIR` or `PMU_FM_TIMESFM3_DIR`
to local weights, then run `python -m pytest -q -m real_model -rs` (CPU by default).
GUI needs a desktop; optional capture needs TShark/Npcap and `PMU_FM_CAPTURE=1`.
Skips are not passes.

Development baseline: Windows 11, Python 3.12.14, PyTorch 2.8.0 CPU; four backends
(Timestral-209M), independent receiver/CSV comparison. Tester-reported second-PC
checks (Windows 11, RTX 3060): 243 base/GUI passes (30 skips), 28 real-model CPU
passes across four runs (7 per backend), and Chronos-2 CUDA GUI/CLI, CSV and
stop/restart checks. CUDA with other backends, Linux, other Timestral sizes,
physical PMUs and long-duration runs remain unverified. Tests do not establish
forecasting accuracy. [requirements.txt](requirements.txt) records all development
dependencies, not a minimal install or portable CUDA lockfile. Report issues with
versions and redacted logs.

## License and third-party notices

Original code: [Apache-2.0](LICENSE). No weights, standards text, private PMU data or
third-party implementation source is bundled. Dependencies retain their licenses;
preserve applicable notices when redistributing them.

| Components | License |
|---|---|
| Python / Tk | PSF / Tcl-Tk |
| NumPy, pandas, psutil; PyTorch | BSD-3-Clause; PyTorch BSD-style and bundled notices |
| Chronos, TimesFM, Timestral source; Transformers, Accelerate, Hugging Face Hub | Apache-2.0 |
| pytest, Ruff; pySynphasor | MIT; pySynphasor copyright (c) 2023 Shuvangkar Chandra Das |
| pyPMU/synchrophasor (tests) | BSD-3-Clause; `utils.py` includes LGPL-3.0-or-later CRC notice |
| Scapy; optional Wireshark/TShark | GPL-2.0-only; GPL-2.0-or-later |

Chronos notice: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
GSF (MIT) was a protocol reference; no source is included. Chronos-2/TimesFM 2.5
weight cards declare Apache-2.0. [Timestral weights](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE)
use Modified MIT with revenue-based eligibility restrictions.
[TimesFM-3 weights](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
are **non-commercial, non-production; redistribution prohibited**.
The project's license does not replace model licenses.
