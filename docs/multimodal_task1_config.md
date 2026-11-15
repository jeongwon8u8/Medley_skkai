# Task 1 — Multimodal configuration

Implemented 2026-11-15 (Asia/Seoul) on `feat/multimodal-framework`, based on commit `967ad1d83fb5c55277143bb48edbd9b309644543`. This task provides configuration interfaces only. Clinical-only, Architecture A and Architecture B are **configuration choices for future implementations**, not executable models. No Dataset, Collator, clinical serializer, model factory, optimizer or Trainer was implemented. Task 2 has not started.

## Environment and change boundaries

The terminal still runs as root on `59e256625f8b`, with an RTX 4090 (24564 MiB), torch `2.8.0+cu128`, torch CUDA `12.8` and CUDA available. This matches the previously inspected Remote SSH GPU container. RunPod account identity cannot be independently proven by these terminal checks. `/workspace` still resolves to root `overlay`; durable RunPod volume attachment remains **unconfirmed**.

Initial status contained only the untracked `docs/multimodal_task0_audit.md`. Its bytes and all existing tracked files were fingerprinted before implementation. New files:

- `medley_pd/configs/multimodal/{base,clinical_only,fusion_a,fusion_b}.json`
- `medley_pd/multimodal/__init__.py`
- `medley_pd/multimodal/config.py`
- `medley_pd/tests/test_multimodal_config.py`
- This document.

Existing Medley source, common configuration, fixed splits and experiment files were not changed. No commit, push or merge. No model/data download, extraction or training.

## API and precedence

From the `medley_pd` working directory:

```python
from multimodal.config import (
    ConfigError,
    load_multimodal_config,
    require_model_id,
    save_resolved_config,
)

config = load_multimodal_config(
    "configs/multimodal/fusion_b.json",
    overrides=["training.batch_size=2", "evaluation.seeds=[0,1]"],
)
assert config.effective_batch_size == 16  # single GPU: 2 * 8
settings = config.settings
medley = config.medley
variables = config.clinical_variables
status = config.split_validation  # "validated" or "deferred"
```

`load_multimodal_config(path, overrides=()) -> ResolvedConfig` accepts a full JSON object, a JSON object with `extends`, or a saved resolved snapshot. Precedence is explicit ancestor/base files, selected architecture file, then CLI overrides in order. An `extends` path is relative to the file declaring it. Inheritance is recursive, cycle-checked and depth-limited. Nested objects merge; lists/scalars replace. No caller mapping is mutated. Unknown keys and missing required settings fail with `ConfigError`, a `ValueError` subclass. Duplicate JSON keys, malformed JSON, wrong container types and invalid override paths also fail.

Override values parse as JSON first, then YAML through the already installed PyYAML, consistent with legacy dotted `key=value` usage. Examples: `false`, `null`, `[0,2]`, `2e-5`, bare `frozen`. Overrides may change known settings but cannot create arbitrary keys. Validation runs after all overrides; invalid supplied values never silently become defaults.

`ResolvedConfig` provides independent `settings` and `medley` dictionaries. The dataclass prevents attribute reassignment but the nested dictionaries are mutable; treat them as read-only during an experiment. `to_dict()` returns a deep copy. Saving revalidates configuration. `validate_config(settings)` is available for complete settings mappings. `recursive_merge(base, patch)` and `apply_overrides(settings, overrides)` are also public helpers; the latter alone does not validate the final result.

## Medley integration and source of truth

Integration calls the unchanged `medley.config.load_config(dataset, encoder, task, overrides=())`. Its existing COMMON -> dataset/encoder/task YAML -> private local paths/encoder overrides behavior is preserved, including reference propagation and split-root anchoring. The new layer does not call legacy `build_cfg` because that imports the sklearn fusion registry and is designed for the existing baseline experiment JSON schema.

`config.medley` captures existing dataset mappings, task definition, encoder settings, preprocessing, paths and full evaluation protocol. Clinical variables are selected from the existing dataset/task clinical sets; they are not redefined in multimodal JSON. The loader checks task target/transform, input exclusions, mapped columns, actual encoder channels/pooling dimensions and Medley evaluation defaults. It imports neither torch nor the model/data/training modules.

Development seed/fold selections live in `config.settings["evaluation"]`; `config.medley["evaluation"]` retains the complete original protocol. A future runner must use the selected settings while preserving the original split manifest. Existing training JSON/CLI semantics remain unchanged and do not consume these new files.

## Schema version 1

All keys below are required in a **resolved** settings object. Shipped architecture files inherit common values from `base.json`. The base is itself a valid Architecture B configuration. No additional framework dependency was introduced.

| Group | Fields, defaults and validation |
|---|---|
| General | `schema_version=1`; `architecture=clinical_only/fusion_a/fusion_b`; `experiment_name` is 1–128 ASCII letters/digits/underscores/hyphens, starts with letter/digit; `dataset=pd_eeg` or `iowa_rest`; `task=moca`; `clinical_set=C0/C1/C2`, default C1 |
| LLM | `model_id=null`, `revision=null` (otherwise nonempty strings); `tuning=qlora`; `quantization=nf4`; `compute_dtype=bfloat16` (also float16/float32); positive integer `lora_r=8`, `lora_alpha=16`, `max_text_tokens=256`; `lora_dropout=0.05` in [0,1) |
| EEG | `encoder=cbramod` or `scerebro`; `input_type=cached_features`; `feature_dim=3800` for CBraMod, 3420 for S-CEReBrO; positive dimensions, checked against resolved encoder settings |
| Feature adapter | `enabled=false`, positive integer `hidden_dim=128`, `dropout=0.1` in [0,1) |
| EEG pooling | `type=attention`; positive integer `num_queries=4`, `hidden_dim=256`, `num_heads=4`; hidden_dim divisible by num_heads |
| EEG projector | `type=linear`, Boolean `layer_norm=true` |
| Fusion | `type=none/representation/soft_token`, respectively required for clinical_only/fusion_a/fusion_b; `token_position=prefix`; `numeric_skip=false`, `uncertainty=false`; positive integer `fusion_dim=256`; `dropout=0.1` in [0,1) |
| Prediction | `type=regression`; positive integer `hidden_dim=256`; `dropout=0.1` in [0,1); `clip_range=[0,30]` or null; finite ordered bounds within [0,30]. Clipping is intended for inference only, not a sigmoid or training-time constraint |
| Loss | `primary=huber`; finite positive `huber_delta=1.0`; `auxiliary=false`. No other losses are currently accepted |
| Training | Positive integer `batch_size=1`, `gradient_accumulation=8`, `max_epochs=30`, `early_stopping_patience=5`; positive finite `lr_lora=0.00002`, `lr_new_modules=0.0001`, `gradient_clip_norm=1.0`; finite nonnegative `weight_decay=0.01`; Boolean `gradient_checkpointing=true`; `validation_fraction=0.2` strictly between 0 and 1; `optimizer=adamw` |
| Evaluation | `use_existing_splits=true`; distinct integer `seeds=[0,...,9]` selected from the existing protocol; `n_folds=5` must match Medley; distinct `folds=[1,...,5]` selects test folds for development; Boolean `save_per_case=true` |

Boolean values cannot masquerade as integers. Nonfinite numbers and unsupported component names are rejected. Hyperparameters are never automatically reduced to fit GPU memory.

Tuning/quantization combinations: QLoRA requires nf4; LoRA requires the string `"none"`; frozen permits none or nf4 (frozen quantized backbone). Quantization is a configuration declaration, not proof that a chosen model/runtime supports it. No LoRA target-module resolver is implemented.

PD clinical sets come from existing task YAML: C0 age/sex; C1 adds UPDRS III/disease duration/LEDD; C2 adds PIGD/TD. For Iowa only C0 matches the initially permitted set names; C1/C2 are unavailable, and `C1_iowa` is intentionally outside this initial schema. Selecting Iowa with the default C1 raises an explicit error; use `--set dataset=iowa_rest --set clinical_set=C0`. MoCA, CTT, UPDRS Total, target variables, subject/group and unmapped variables are rejected as inputs. Clinical value timing/units and any additional leakage require future private-data preflight.

S-CEReBrO requires both `--set eeg.encoder=scerebro` and `--set eeg.feature_dim=3420`; the loader does not silently adjust feature dimensions. Existing S-CEReBrO mean pooling (180 dimensions) is outside this initial schema and raises an integration error if supplied through local settings. Clinical-only retains EEG configuration/provenance fields but does not load EEG or assert cache availability.

## Fixed splits

The new loader never imports `medley.splits` or invokes `get_splits`, which can create missing files. `validate_split_selection(settings, medley)` reads CSV metadata only. It verifies required columns, nonempty assignments, unique (subject, seed), valid fold bounds, consistent patient coverage across seeds and availability of each selected seed/fold. It returns `validated` when the manifest is present and valid, or `deferred` when absent. Malformed or unreadable existing manifests fail, rather than being treated as absent. No patient IDs appear in errors or snapshots.

The four current repository manifests were read successfully without modification. Patient identity across visits, alignment to future datasets, train/validation separation and actual cache availability remain Task 2 or later preflight concerns. Missing manifests must fail before training in a future runner; `deferred` is not permission to generate replacements.

## Reproducible snapshots and authentication

`save_resolved_config(config, output_path, *, overwrite=False) -> Path` saves sorted, indented UTF-8 JSON with a trailing newline. A snapshot contains exactly `settings` and `medley`, including inherited and CLI settings and the captured Medley configuration. No timestamps, patient assignments or runtime status are embedded. Reload uses captured Medley values, not potentially changed `local.yaml`, and validates current split metadata when available. Snapshots therefore preserve model/training/preprocessing settings; they do not package source code, model weights or data and do not guarantee bitwise training reproduction.

The parent directory must exist. Output must end in `.json`; symbolic-link files and destinations within the application's existing config or configured split directories are rejected. Exclusive creation is the default; replacing an output requires `overwrite=True` or CLI `--overwrite`. New snapshot files are created with owner-only permissions. Keep snapshots with local paths in private experiment storage.

Credential fields (token/password/secret/api_key/access_token/auth_token/credentials), credential-bearing URLs and HF-token-shaped identifiers are rejected. Unknown settings also fail. Authenticate through environment/service facilities in later loaders. Do not place credentials in model IDs, revisions or local path strings; these checks cannot classify every arbitrary secret disguised as ordinary text.

`llm.model_id=null` is configuration-valid. A future model factory must call `require_model_id(config)` before loading, then separately verify revision, access, tokenizer, quantization, compute dtype, GPU and LoRA targets. No model ID is claimed verified here. Default revision null intentionally leaves model pinning to the future model-selection step.

## Executed CLI examples

Run from `/workspace/Medley_skkai/medley_pd`. The system Python can run the configuration CLI with existing PyYAML; the test environment can also run it.

```bash
python -m multimodal.config --config configs/multimodal/clinical_only.json
python -m multimodal.config --config configs/multimodal/fusion_a.json
python -m multimodal.config --config configs/multimodal/fusion_b.json
python -m multimodal.config --config configs/multimodal/fusion_b.json --set training.batch_size=2 --output /tmp/resolved-example.json
python -m multimodal.config --config /tmp/resolved-example.json
python -m multimodal.config --config configs/multimodal/fusion_b.json --set llm.tuning=frozen --set eeg.pooling.num_queries=8
python -m multimodal.config --config configs/multimodal/fusion_b.json --set llm.tuning=lora --set llm.quantization=none
```

These command forms were executed using `/tmp/medley-task1-tests/bin/python`, with snapshot destinations in a fresh temporary directory rather than the illustrative `/tmp/resolved-example.json`. Each returned 0 and valid resolved JSON. Repeated output creation without `--overwrite` returned 2 and preserved the existing file. CLI validation errors also return 2 with a configuration error message. The CLI validates/prints settings only; it does not train any model.

## Tests and dependency impact

Before installation, the plan and effect were reported: create `/tmp/medley-task1-tests` using `python -m venv --system-site-packages`, inspect pip dry-run and install only missing lightweight test dependencies. Installed in that venv: pytest 9.2.0, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1, and their required iniconfig/pluggy/joblib/narwhals/threadpoolctl/cloudpickle dependencies. Existing numpy 2.5.2, PyYAML and GPU packages are reused. No system dependency was installed or upgraded. `pip check` in the venv passed. This is a temporary Task 1 verification environment, not a project-wide historical experiment lock.

Executed from repository root:

```bash
PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_config.py
```

Tests cover shipped configurations, inheritance/merge, override precedence and type conversion, valid tuning pairs, invalid and missing fields, nonfinite numbers, leakage/dimension drift, read-only/deferred split validation, snapshots/round trips/output protection, CLI behavior and unchanged legacy configuration files/loader behavior. A fresh subprocess confirms that loading does not import torch, transformers, peft, bitsandbytes, pandas, sklearn or the Medley split module. Existing root tests and four legacy configuration cases are included.

Final result: **100 passed in 2.21s**. The initial 98-test run also passed; two additional edge cases cover scientific-notation overrides and Medley leakage/dimension mismatches. `compileall`, the repository scaffold check and whitespace checks passed. Fingerprints confirmed all 113 original files, including the Task 0 audit and four fixed splits, were unchanged; only the eight Task 1 files listed above were added. The original branch remained active. System torch remained `2.8.0+cu128` / CUDA `12.8` with CUDA available; pytest remained absent from the system interpreter and was installed only in the temporary test venv.

The synthetic `medley_pd/tests/smoke_test.py` was not run: it performs many baseline fits and permutation workflows and writes/deletes local.yaml, beyond configuration-only Task 1. No EEG/model compatibility or training behavior is claimed tested.

## Future consumers, not implemented

Task 2 may consume `config.medley` for existing paths/column mappings, `config.clinical_variables` for the allowed input list and `config.settings` for the selected cached-feature/architecture/evaluation policy. It must require fixed manifests, validate caches and patient alignment, and honor patient-level inner validation. A future model factory will consume LLM/EEG/fusion/prediction fields; a future trainer will consume training/loss settings and the computed single-GPU effective batch size. No such consumer was added in Task 1.
