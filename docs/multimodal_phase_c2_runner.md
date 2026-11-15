# Phase C2 — Unified multimodal experiment runner

## Scope and prerequisites

C2 adds planning, lazy model registration, bounded execution, validation-only
checkpoint selection, final test export and validated run-level reuse for M0–M3.
No actual full-grid, real-patient, pretrained 4B or GPU QLoRA training was run.
No C3 statistics or Phase D experiments are implemented.

Started at C1 `f022af6e6888985410bcec4b137f9dfd7a805596` on
`feat/multimodal-framework`, without uncommitted prior work. C1's **72 tests passed**
before source changes. Tasks 0–6/A1–A3/B/C1 documentation and actual config,
protocol, manifest, Dataset, trainer, checkpoint and legacy prediction contracts
were inspected. C1's initially unpushed commit was preserved; no amend/rebase/reset.
Shared changes are recorded in docs/requests/fusion.md. Original fixed split hashes
are unchanged; actual seeds/folds remain inherited from C1, not recreated.

## Architecture and APIs

| Module | Responsibility / API |
|---|---|
| experiment_runner.py | `plan_experiments`, `PlannedRun`, `execution_guard`, `main` CLI |
| model_registry.py | `REGISTRY`, `model_config`, `create_runtime`, `ModelRuntime`, `validate_runtime_config`, `check_readiness`, `RunnerError` |
| training_control.py | `EarlyStopping`, `train_epoch_with_validation`, `train_select_restore`, `verify_restoration` |
| runner_execution.py | `execute_plan`, `ExecutionResult` |
| run_state.py | `RunStore`, `inspect_completed`, `validate_completion`, atomic JSON/integrity helpers |

All above modules are under medley_pd/multimodal. Existing scientific model
forward definitions and trainer optimizer/accumulation behavior are unchanged.
Registry imports model/trainer/checkpoint modules only when execution needs them.
M1 uses Task 6 ClinicalLLMRegressor/ClinicalTrainer/clinical_checkpoint; M2 uses
Phase B representation fusion APIs; M3 uses Phase A soft-token APIs. The runner
never assembles EEG tokens, branch representations or model-specific forward inputs.
M0 has no model/trainer factory and is import-only.

`model_config` copies a Task 1 config and changes only architecture/fusion type
for M1/M2/M3. Clinical feature sets, adaptation modes, encoder, optimizer groups
and all training settings are preserved. Each run creates an independent supplied
Task 4 backbone/model/head, without copying fitted M1/M2/M3 weights.

```python
from multimodal.experiment_runner import plan_experiments
from multimodal.runner_execution import execute_plan

plans = plan_experiments(config, models=['M2'], seeds=[0], folds=[1],
                         output_dir='outputs/multimodal_runs')
# Planning alone reads no participants, EEG cache or pretrained model.
# Execute only after real-model/data preflight and explicit authorization:
results = execute_plan(plans, max_runs=1, existing='fail')
```

Explicit adapter/backbone factory injection supports synthetic testing and
embedding. Tests still use the actual registered M1/M2/M3 models and trainers.
Factory injection is not exposed as a CLI mode, not a QLoRA fallback, and is not
proof of compatibility with arbitrary injected pretrained bases. Callers supplying
factories must preserve their declared model identity and initialization policy.

## Planning and safe CLI

Default is dry-run. Planning uses configuration plus read-only C1 outer-source
metadata, never participant/cache/pretrained loading. Plan IDs bind model,
seed/fold, effective configuration, outer source hash and output root. Missing
sources/resources are listed as prerequisites; a plan is never a completed run.
Cohort/cache fingerprints remain unverified until execution prepares C1 records.
Actual C1/execution IDs and namespaces are finalized after cohort validation.
Default model comes from configured architecture. Configured grid is normally
50 runs per model; explicitly listing M0–M3 yields 200 plans, never execution.

Both entrypoints work with the existing test environment from repository root:

```bash
# One model/seed/fold dry-run; default is also planning without --dry-run.
/tmp/medley-task1-tests/bin/python -m medley_pd.multimodal.experiment_runner \
  --config medley_pd/configs/multimodal/fusion_b.json \
  --model M2 --seed 0 --fold 1 --dry-run

# Entire configured plan, no execution.
/tmp/medley-task1-tests/bin/python -m medley_pd.multimodal.experiment_runner \
  --config medley_pd/configs/multimodal/fusion_b.json --dry-run

# Equivalent application import style.
PYTHONPATH=medley_pd /tmp/medley-task1-tests/bin/python -m multimodal.experiment_runner \
  --config medley_pd/configs/multimodal/fusion_a.json --model M2 --seed 0 --fold 1

# Template only: requires an existing private, preflighted effective config.
/tmp/medley-task1-tests/bin/python -m medley_pd.multimodal.experiment_runner \
  --config /private/resolved_config.json --model M2 --seed 0 --fold 1 \
  --execute --max-runs 1 --output-dir outputs/multimodal_runs

# Validate completed artifacts without pretrained model loading.
/tmp/medley-task1-tests/bin/python -m medley_pd.multimodal.experiment_runner \
  --inspect-completed outputs/multimodal_runs/<C1-run-id>/executions/<execution-id>
```

`--model`, `--seed`, `--fold` accept explicit lists. Unsupported/duplicate models
or identifiers outside configured selections fail. Execution requires --execute
and --model plus explicit seed/fold selection, or an additional --ack-grid.
--max-runs defaults to 1 and must cover the complete selected count before any
adapter/model is constructed; it does not silently take the first N runs.
A larger grid needs explicit --ack-grid and a sufficient bound. No such real
grid was executed during implementation. Python API execution is likewise bounded.

--set reuses Task 1 dotted overrides; --output-dir sets C1's private manifest root.
--resume-completed, --restart-incomplete and --fail-on-existing are mutually
exclusive. Default is fail on existing execution. Dry-run does not create run
files, initialize torch/LLM modules or download anything. CLI reports count,
models, destinations, source/config identities and prerequisites without IDs/rows.
Configuration, file, dependency and backend errors are sanitized; arbitrary
exception text is not printed or stored.

## Training and validation semantics

C1 prepares exact source/model cohort, targets, clinical fields and cache content,
and provides the only train/validation/test partition. The runner creates actual
Task 3 Dataset/Collator loaders with configured subject batch size, drop_last=false,
num_workers=0 and configured train shuffling. Validation/test order follows C1's
source order. No outer-test label is used for training or selection. Inputs and
source are revalidated before final test and completion; changed cohort/targets/
cache/clinical content fail rather than producing Completed status.

Before model creation, Task 6 seed_training is called with C1 derive_seed of the
actual seed/fold and purpose `training:<clinical_regression.random_seed>`.
The independent CPU loader generator uses purpose
`dataloader:<clinical_regression.random_seed>`. Python/NumPy/torch/CUDA seeding is
therefore controlled without enabling expensive global deterministic algorithms.
Backend/kernel/GPU nondeterminism remains; bit-identical GPU behavior is not claimed.

Existing defaults are reused: Huber delta=1, AdamW, max_epochs=30, patience=5,
microbatch=1, accumulation=8, LoRA LR=2e-5 and new-module LR=1e-4. Each trainer
retains its model-specific groups and actual subject-count gradient normalization,
clipping and partial accumulation behavior. Runner does not impose a generic LR.

New optional Task 1 `runner` settings:

| Field | Default / support |
|---|---|
| selection_metric | MAE only |
| min_delta | 0.0, finite nonnegative |
| shuffle_train | true, Boolean |
| num_workers | 0 only; multiworker/distributed behavior deferred |
| baseline_path | null; explicit existing CSV/gzip source for M0 |
| baseline_model | null; explicit legacy model name, e.g. clinical:C1 |

All effective values are in the C1 resolved snapshot. Task 4's actual resolved
padding, qualified adapter targets, device and model revision are additionally
saved in runtime_config.json. Scientific/runner settings may not change in the
factory. Explicit revision/padding/adapter targets are checked against the request.
Orchestration-only runner options are excluded from inference checkpoint digests,
preserving old M1/M2/M3 inference files; they remain in full run/config identity.

Epochs are one-based. Each epoch trains only inner train, evaluates only inner
validation and requires finite nonnegative MAE/RMSE and correct subject count.
First finite epoch establishes best. Improvement is strictly
`MAE < best_MAE - min_delta`; equality is not improvement. On improvement reset
bad_epochs to zero and save a fresh checkpoint directory for that epoch. Otherwise
increment bad_epochs, then stop at bad_epochs >= patience. Max epochs is inclusive.
NaN/Inf/invalid validation metrics fail the run. No valid checkpoint means failure.

After the loop the best checkpoint is loaded through its original model-specific
loader. Epoch/counters and actual saved trainable tensors are checked for exact
restoration. Failure cannot fall back to final-epoch weights. The training selector
accepts no test loader. Only after restoration does execution call test evaluation
**once**, verify finite predictions/exact ID coverage and source order, export,
and compute durable completion. Controlled-score orchestration tests are separate
from real-trainer smoke acceptance; they prove an earlier best is restored.

## Checkpoints, exports and durable layout

Existing safetensors inference formats save only each model's documented new
modules/LoRA, not full pretrained bases. No optimizer/RNG/scheduler/sampler state
is saved, so full training or mid-epoch resumption is unsupported.

```text
<private-root>/<C1-run-id>/
  manifest.json                 # exact C1 schema, atomic exclusive publication
  .manifest.lock
  executions/<execution-id>/
    .lock
    completed.json              # final immutable completion marker
    attempts/<unique-attempt>/
      state.json
      runtime_config.json       # M1/M2/M3
      history.json
      selection.json
      checkpoints/epoch-0001/... # original model-specific inference format
      predictions.csv.gz
      metrics.json
      baseline_audit.json       # M0 only; no model checkpoint
```

Execution identity binds C1 run ID and M0 source-byte hash when applicable. C1
already binds full config, actual split/cohort/cache/clinical identity, code and
runtime provenance. Changing relevant inputs/config/code/environment yields a
new namespace; completed reuse is intentionally strict. Original prediction
columns stay `seed,fold,subject,subgroup,model,y,yhat,target`. M1/M2/M3 retain their
established distinct model strings. Extra metadata is stored separately, not in
additional patient CSV columns. Patient rows remain private and never auto-logged.

selection.json binds execution/C1 ID, model, seed/fold, best epoch and validation
MAE, relative checkpoint path, verified restoration and test_used_for_selection=false.
Completeness validation replays configured early stopping across sequential history,
checks patience/max-epoch boundary, counts/step counts and selected best. It checks
runtime versus planned config, checkpoint format/config/counters/weight checksum/
finite tensors, prediction schema/IDs/order/values, recomputed MAE/RMSE, state,
manifest linkage and complete artifact hashes before completion/reuse.
Hashes establish local integrity/self-consistency, not third-party authenticity
or scientific provenance of missing historical inputs.

## M0 import-only semantics

Provide runner.baseline_path and runner.baseline_model through config/--set.
C1 audits actual seed/fold, exact subject population, target/subgroup mapping,
finite predictions, model identity and duplicate/missing rows. Source bytes are
hashed and rechecked across read-only audit/import. Explicit selected rows are
copied to the new namespace in C1 test order; the source is never modified.
No baseline model, optimizer, early stopping or checkpoint is instantiated.
The completion status is **imported**, and baseline_audit.json explicitly records
training_provenance=not_verified. Population alignment does not establish equal
historical training/validation/cache/preprocessing; unmatched artifacts fail.
Changing baseline bytes creates a different execution identity, never stale reuse.

## Lifecycle, interruption and reuse

Planned exists only as returned/printed plan metadata. Running is an owned attempt.
Failed records sanitized fixed error categories. Incomplete means no valid final
completion marker; directory existence alone never means completed. Completed
(or imported for M0) requires validated artifacts and atomic completed.json.

Initial C1 manifest and final completion use 0600 temp files, file fsync, atomic
exclusive links and directory fsync. Active attempt state updates use atomic
replace inside its owned namespace. Namespace/attempt directories are private.
A short bootstrap flock protects C1 publication; a nonblocking execution flock
prevents parallel owners. Lock files remain to avoid unlink/recreate races.
Existing and new namespaces both enforce protected data/cache/split roots,
Git-ignored outputs/ or checkpoints/ inside the repo, and symlink rejection.
Explicit external private output roots are supported. No previous result is overwritten.

--resume-completed revalidates C1 current inputs, completion identity and all
required artifacts before returning aggregate results without pretrained loading
or any training/test pass. Safetensors integrity checking still reads small
inference weight files on CPU. If no previous execution exists, explicit execution
may create one; an existing incomplete/failed run requires --restart-incomplete.
Restart preserves every old attempt and creates a fresh attempt initialized with
the same configured seed. It is restart-from-initialization, not optimizer/RNG resume.

A crash before atomic bootstrap publication leaves recoverable orphan temporary
metadata; publishing the missing manifest never overwrites those files. Catchable
interruptions mark the active attempt failed. A hard kill may leave running state,
but absence of completed.json still means incomplete. Explicit restart after lock
release makes a new attempt. Corrupt manifests or completed artifacts are rejected,
not repaired or silently treated as cache hits. Choose a new explicit output
namespace when previous provenance cannot be established. An exception after completion publication cannot rewrite its hashed state; already
validated published artifacts remain reusable. A filesystem crash can lose partial
work; run-level recovery does not promise exact mid-epoch replay.

## Verification and independent review

Sequential gates: planning **9 passed**; actual single M1 epoch **10 passed**;
shared early-stopping gate **22 passed**; actual M1/M2/M3 checkpoint/export/reuse
**25 passed**; recovery/M0/LoRA expanded gate **40 passed**. Final tests add
completion semantics, source-root/symlink safety, ownership, metadata types,
configuration/dependencies, zero-torch dry-run and bootstrap interruption.

Acceptance smoke uses actual Task 2 adapter reading temporary authored participants,
NPZ caches and fixed CSV, actual Task 3 loaders and locally constructed tiny GPT2
backbones. Frozen and LoRA M1/M2/M3 execute bounded single-seed/fold runs, select/
restore checkpoints, export final test rows and validate/reuse completed outputs.
M0 imports a synthetic existing file byte-for-byte read-only. No 50-fold execution,
patient data, pretrained download or real clinical performance claim is involved.

Independent read-only source review and its own synthetic runs reproduced two
issues before completion:

1. Completion checked only one improved history row, permitting an earlier
   non-best epoch after rehashing. It now replays configured stopping and binds
   complete selection/config identity, with focused rehashed-metadata tests.
2. Existing C1 namespace bypassed new-namespace source-root validation. All
   existing/new paths now share protected-root/ignored/symlink checks, with
   precreated-manifest and namespace/execution redirection regressions.

Independent initial reviewer tests passed 38 cases. The follow-up reviewer turn
was unavailable due to the agent tool's usage limit. Corrections were verified
by focused primary-agent regression tests and source review; no successful
independent follow-up is claimed. Additional final review strengthened bootstrap
atomicity, ownership, metadata typing and sanitized CLI errors. Earlier tests
were not changed or weakened.

Exact commands from repository root, existing environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_experiment_runner.py medley_pd/tests/test_multimodal_training_control.py medley_pd/tests/test_multimodal_run_state.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_*.py
```

Final C2 result: **51 passed, 0 failed, 0 skipped** (6 retained warnings).
Final full relevant regression: **862 passed, 0 failed, 1 skipped** (115 retained warnings).
Existing opt-in actual NF4
GPU test remains skipped, not passed. Retained warnings concern PEFT GPT2 Conv1D
orientation and Transformers checkpointing deprecation. Source syntax, credential/
artifact scope, split hashes and staged whitespace checks accompany testing.

## Engineering principles, limitations and C3 handoff

| Principle | Concrete mechanism |
|---|---|
| Modularity | C1 retained; registry, training selection, durable state and CLI/execution separated |
| Configurability | Task 1 defaults/overrides/snapshots; per-model groups retained; explicit bounded selection |
| Backward compatibility | Existing models/trainers, source splits and baseline files unchanged; old inference identity normalization and full regression |
| Reproducibility | C1 identities/content/code/runtime provenance, purpose-derived seeds, actual runtime snapshot and validated reuse |
| Incremental development | Plan -> actual epoch -> stopping -> checkpoint/export -> recovery -> multi-model/regression gates |
| Ablation-first | Different resolved configs create distinct identities without fusion internals in runner |
| No redundant experiments | M0 import-only; valid completed reuse; no EEG extraction or prior baseline retraining |

New files: five runner modules above, three corresponding test modules, this doc.
Necessary shared changes: config.py optional runner settings, clinical_checkpoint.py
orchestration-only digest exclusion, docs/requests/fusion.md. Twelve files total.
No generated weights/predictions/cache/participant data are committed.

Real execution remains unverified: defaults still have no selected LLM ID/revision,
bitsandbytes is absent and configs/local.yaml is absent at environment recheck.
Default real factory requires local_files_only=true, model ID, immutable 40-hex
revision, dependencies and Task 4's genuine mode/backend checks, with no silent
QLoRA fallback. No packages/CUDA stack were installed or changed. Actual NF4/BF16,
pretrained forward/backward, peak VRAM, clinical units/sex coding, patient/visit
linkage, cache extraction provenance and persistent storage still need preflight.

C3 can consume validated completed.json, C1 manifest, selection/history, metrics,
private prediction CSV and M0 audit status. Reuse does not establish that different
models have matched capacity or historical training. C3 still must enforce matched
population/settings and implement statistics separately. Stop at C2.
