# IF-MAP core files

Use `core_files.txt` as the upload allowlist (32 files).

| Scope | Files |
| --- | --- |
| Entry | `run.py` |
| Algorithm and runtime | `src/*.py` (listed below) |
| Workflow templates | `configs/workflows/who_when_ag.yaml`, `who_when_hc.yaml` |
| Setup | `requirements.txt`, `.env.example`, `.gitignore` |
| Optional data utility | `download_general_benchmarks.py` |
| Offline checks | `tests/test_core_release.py` |
| Docs and manifest | `README.md`, `CORE_FILES.md`, `core_files.txt` |

## Source modules

- `src/__init__.py`
- `src/aflow_adapter.py`
- `src/dataset_loader.py`
- `src/encoder.py`
- `src/errors.py`
- `src/evaluator.py`
- `src/llm_cache.py`
- `src/logger.py`
- `src/mas_runner.py`
- `src/paths.py`
- `src/perturbation.py`
- `src/perturbation_diag.py`
- `src/perturbation_pool_core.py`
- `src/pilot_reference.py`
- `src/privacy.py`
- `src/rq2_perturbation.py`
- `src/rq2_runner.py`
- `src/score.py`
- `src/textgrad_opt.py`
- `src/variant_generator.py`
- `src/workflow_loader.py`

## Excluded

Do not upload `aflow/`, `baselines/`, `autogen/`, `scripts_bench/`,
`analysis/`, paper folders, `data/`, `cache/`, `results/`, `.git/`,
`.env`, private credentials, logs, bytecode, or historical archives.
Their contents are not part of this sanitation pass.

`src/aflow_adapter.py` is the IF-MAP bridge, not the external `aflow/` tree.
Full workflow runs still require that compatible external runtime and datasets.

## Paths and privacy

Paths in config are project-relative. Runtime resolution may use absolute
paths internally; no host-specific absolute path is required in source.
Credentials come from environment variables or an explicitly configured file.
Run metadata omits hostname and private Git revision. Error reports and config
snapshots redact credentials and home-directory prefixes.

Dataset prompts, traces, and cached responses are not automatically anonymized:
they must retain their scientific content, and they are excluded from this release.
