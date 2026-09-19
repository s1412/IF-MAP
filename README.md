# IF-MAP

Prompt-perturbation estimates of agent improvement potential.

## Core

`run.py` configures the experiment. `src/` implements perturbation pools,
potential scoring, TextGrad, evaluation, and reporting.

Metrics: **G-norm**, **I_abs**, **MeanAbs**. Ranking agreement: Kendall tau-b.

RQ2: DEV-A probes -> DEV-B candidate selection -> DEV-C guard -> TEST.
RQ3 optionally upgrades one agent model at a time on TEST.

## Setup

Python 3.10+. Install `requirements.txt`. For the optional dataset downloader,
also install `datasets` and `huggingface_hub`.

Copy `.env.example` to `.env` and fill credentials locally. Exported variables
take precedence. Never commit `.env` or credential files.

Paths are relative to the **IF-MAP project root**, not the shell working
directory. `../qwen3-8b` denotes an optional sibling model-service directory.
Set its paths in `run.py` for your deployment. Set `auto_switch=False` when
using an already managed service.

Workflow evaluation requires the compatible AFlow workflows, configuration,
datasets, and runtime. They are **not part of this core release**. Place them
under `aflow/` or set `IFIP_AFLOW_ROOT` to their directory. Configure external
provider credentials there separately; no private AFlow config is exported.

`EXPERIMENT['rq2']['perturb_mix']['reference_root']` points to optional
historical DEV-A references. Supply those artifacts for historical
reproduction, or set it to `None` for freshly generated candidates. Historical
TEST measurements are not used to generate candidates.

## Run

Review `PATHS`, `EXPERIMENT`, `METHOD`, and `RUNTIME` in `run.py`, then:

```bash
python run.py
```

Live runs call paid APIs and may manage local model services. For offline checks:

```bash
python -m unittest discover -s tests -p test_core_release.py -v
```

Legacy `IFIP_*` environment names, error tags, and cache schemas remain for
compatibility. Algorithms, prompts, temperatures, seeds, and token budgets
are unchanged by release cleanup.

## Export

Use `core_files.txt`, not the entire working directory:

```bash
zip -@ IF-MAP-core.zip < core_files.txt
```

See `CORE_FILES.md`. Results, traces, datasets, logs, third-party code, and Git
history require separate review and are excluded.
