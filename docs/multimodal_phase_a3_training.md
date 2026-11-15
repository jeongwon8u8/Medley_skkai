# Phase A3 / Task 9 — Architecture B training and verification

## Scope and provenance

A3 makes A2's M3 forward trainable using explicit subject loaders and verifies
training, evaluation, private export and inference restoration on CPU. Started
on clean `feat/multimodal-framework` at A2 commit
`f7c269e` (A1 `608d2ef`, Task 6 `b40a7fe`). Reviewed Tasks 0–6, A1/A2,
ownership/contracts, Fusion instructions and actual source interfaces. A1/A2
prerequisite tests passed: **153 passed, 0 failed, 0 skipped**.

No Architecture A, pretrained training, baseline rerun, full CV runner, automatic
inner split, model selection, patience-based early stopping or ablation experiment
is implemented. No dependency, CUDA/PyTorch stack or existing fixed CSV changed.
Shared changes are recorded in `docs/requests/fusion.md` under the user's explicit
A3 integration authorization.

## APIs and composition

From repository root use `PYTHONPATH=medley_pd` for standalone application imports.

```python
from multimodal.fusion_b_trainer import FusionBTrainer, build_optimizer, seed_training
from multimodal.fusion_b_checkpoint import (
    export_predictions, save_inference_checkpoint, load_inference_checkpoint,
)

# Call seed_training(config.settings['clinical_regression']['random_seed'])
# BEFORE independently preparing Task 4 backbone and A2 model.
# Construct DataLoader with an explicitly seeded torch.Generator, batch_size from
# training.batch_size, and the Task 3 MedleySubjectCollator.
trainer = FusionBTrainer(model, split=adapter.get_split(seed, fold),
                        train_subjects=explicit_inner_train_ids,
                        validation_subjects=explicit_withheld_outer_train_ids)
info = trainer.train_epoch(train_loader)
validation = trainer.evaluate(validation_loader)  # role='validation'
test = trainer.evaluate(test_loader, role='test')  # explicit, no selection
export_predictions(test, private_output_path)     # exclusive create
save_inference_checkpoint(model, private_checkpoint_path,
                          epochs_completed=trainer.epochs_completed,
                          optimizer_steps=trainer.optimizer_steps)
# Independently reconstruct identical base/tokenizer/config and A2 model first.
load_inference_checkpoint(reconstructed_model, private_checkpoint_path)
```

`FusionBTrainer` inherits Task 6's execution engine through small optimizer,
model-identifier and input/forward hooks. Common `_batch_targets` validates IDs
and original-unit targets. M1 retains clinical-only serialization and never reads
EEG. M3 moves features to the pooling device/dtype, and boolean masks/flags to the
head device, without mutating source batches. Clinical serialization/tokenization
remain entirely in A2. Inputs to the model never include targets, IDs or subgroup.

No complete trainer copy was introduced. Existing Task 6 seed utility, Huber
loss, partition checks, evaluation result container, metric definitions and CSV
export are reused. `FusionBEvaluation` aliases `ClinicalEvaluation` (metrics and
private predictions DataFrame). `build_optimizer(model)` can be used independently.
The resolved configuration comes from the supplied backbone rather than an
independent potentially conflicting trainer configuration argument.

## Objective, optimizer and accumulation

Task 6 `moca_huber_loss` enforces finite identical `[B]` prediction/target shapes,
no broadcasting, and positive configured delta (default **1.0**). Targets stay in
original MoCA units. Predictions remain raw; the reserved inference clip option
is not applied during training or evaluation. Each row is one subject, regardless
of EEG window count. There is no LM generation objective or target normalization.

AdamW groups have deterministic order: EEG pooling, EEG projector, regression
head, feature adapter if parameterized, then LoRA. Base weights are excluded.
Every trainable parameter must occur exactly once; duplicate/omitted parameters,
unfrozen base weights, frozen new modules and missing trainable adapters fail.
Feature adapter ON remains rejected by A2/A1 until its separately scoped phase;
the optimizer does not imply a newly implemented feature adapter.

| Setting | Default / behavior |
|---|---|
| `training.optimizer` | `adamw` |
| `training.lr_lora` | `2e-5` |
| `training.lr_new_modules` | `1e-4` |
| `fusion_b_training.lr_pooling` | null: inherit new-module LR |
| `fusion_b_training.lr_projector` | null: inherit new-module LR |
| `fusion_b_training.lr_head` | null: inherit new-module LR |
| `fusion_b_training.lr_feature_adapter` | null: inherit new-module LR; reserved adapter remains OFF |
| `training.weight_decay` | 0.01, includes biases/norms as in Task 6 |
| `training.gradient_clip_norm` | 1.0, after subject normalization |
| `training.batch_size`, `gradient_accumulation` | 1 and 8 |
| `clinical_regression.mixed_precision` | none; CUDA BF16 requires supported device |
| `clinical_regression.random_seed` | 0, caller seeds before model/loader construction |

Group overrides are optional positive finite numbers; bool, nonnumeric, zero,
negative and nonfinite values fail. Old JSON/snapshots receive null defaults.
Inheritance, dotted overrides and resolved snapshot roundtrips are tested.

For each accumulation group, sum per-subject Huber losses and call backward for
each microbatch. Immediately before stepping, divide all accumulated gradients
by the **actual subject count**, then clip and AdamW-step. Final partial groups
are stepped, unequal microbatch sizes retain correct weighting, gradients are
cleared at boundaries/errors, and successful update counters survive a later
batch failure. A failed epoch is not rolled back: completed optimizer steps may
already have changed weights; `epochs_completed` increments only on full success.

Tests compare both optimizer-input normalized gradients and resulting parameters
against explicit effective batches in Frozen and LoRA, including final partial
groups. Dropout is zero in controlled comparisons. Gradient tolerance is
`rtol=2e-4, atol=2e-6`; parameter/prediction comparison uses
`rtol=2e-5, atol=5e-6`. Observed near-zero attention-bias Adam updates differed up
to 2.61e-6 due to FP32 batched operation order; this is why exact equality is not
required. Production dropout/backend/hardware can introduce additional differences.

## Gradients, precision and readiness

Actual Huber backward and AdamW updates were verified for local tiny random GPT2
Frozen and LoRA models using batch sizes 1 and 3, variable EEG lengths and missing
clinical values. All intended EEG learned queries, input/attention projections,
projector and head receive finite nonzero gradients. LoRA gradients are finite
and at least some adapters have nonzero gradients/updates; zero-initialized LoRA
factors need not all have nonzero gradients on the first step. Base weights stay
frozen and unchanged. A2's separate GPT2/Qwen2/Gemma gradient tests also pass.

Training does not wrap the frozen LLM in no_grad. Evaluation uses Task 6 no_grad
and restores prior train/eval mode. A2 performs differentiable EEG-to-actual-text
embedding dtype conversion. CPU tests use FP32 and no autocast; CUDA BF16 remains
an explicitly gated future runtime policy, not a verified GPU training claim.

**Real QLoRA execution: NOT VERIFIED.** CPU QLoRA requests fail through actual
Task 4 readiness checks without LoRA fallback. No bitsandbytes installation,
pretrained download/load, GPU NF4 forward/backward or peak VRAM measurement was
performed. No 4B compatibility or fit claim follows from these tiny CPU checks.

## Splits, cohort, evaluation and exports

Pass the Task 2 `SubjectSplit` from its read-only adapter. Explicit train and
validation IDs must be unique disjoint subsets of outer train; outer test is
forbidden for training/validation. Seed/fold must belong to configured selections.
No IDs are fabricated and no split is regenerated. Loader Dataset membership and
observed batches are checked; duplicate, unexpected and omitted subjects fail.
Custom iterable errors discovered after an earlier update cannot undo that update.
One subject yields one prediction. Validation does not change weights, optimizer
state or update counters; there is no automatic checkpoint selection.

MAE = mean(abs(yhat-y)); RMSE = sqrt(mean((yhat-y)^2)); n counts subjects.
These use existing `medley.metrics.regression_metrics`. Synthetic values test
plumbing and arithmetic only and are not clinical performance results.

Export columns exactly follow Task 6/Medley:
`seed,fold,subject,subgroup,model,y,yhat,target`, with original-unit `y == target`.
M3 identifier is `fusion_b_llm:C0|C1|C2`, distinct from M1 `clinical_llm:*`.
Order follows the explicit loader. Export is opt-in CSV or reproducible gzip and
exclusively creates the destination; existing M0/M1/baseline files cannot be
overwritten. In-application destinations are restricted to ignored `outputs/`
and `checkpoints/`; A3 adds application ignore entries for associated JSON
metadata too. Synthetic outputs use pytest temporary directories. Patient rows
are never automatically logged or committed.

M1 and M3 both use the actual Task 3 EEG-backed Dataset cohort: even M1's clinical
path requires its subjects' EEG caches to load through that Dataset. Missing
cache errors do not silently create a larger clinical-only cohort. Fair real
comparisons still require validating identical eligible subject lists/cache
availability and patient/visit identity before execution.

## Checkpoint format and restore

M3 saves `trainable.safetensors`, `resolved_config.json`, and last-written
`manifest.json` in an exclusive new directory. Scope is role-based even if the
caller inference-freezes modules: pooling, projector, head, feature adapter if
parameterized, and LoRA weights. Pretrained base weights are excluded. Metadata
contains effective settings/seed, model ID and pinned revision, tuning mode,
LLM config/tokenizer fingerprints, full resolved-config digest, progress counts,
Git HEAD when available, and checksum. Git HEAD alone does not attest a clean
working tree. Small injected synthetic frozen bases are fingerprinted; large
injected bases require explicit model identity. Pretrained revisions must be pinned.

Caller must reconstruct the matching base/tokenizer/config before loading.
Architecture, K, dimensions, mode, tokenizer, synthetic base, full tensor
keys/shapes/dtypes/finiteness, manifest fields and hashes are checked before any
weight copy. No pickle, dynamic import or model download occurs. Digests detect
mismatch/corruption, not authenticity of arbitrary untrusted artifacts. Exact
training configuration matching is intentionally strict for M3 restoration.

Checkpoints are **inference-only, not full training-resume**: optimizer, sampler,
RNG and scheduler state are absent. Restore predictions agree with
`rtol=1e-6, atol=1e-7`; inference freezing can select a different PyTorch MHA path
(observed 1.49e-8 difference). M1 config hashing excludes only the new M3-specific
optional settings, preserving pre-A3 M1 inference checkpoint compatibility.

## Incremental tests and independent review

First gate: 4 Frozen/LoRA training/gradient tests passed. Next gates added
subject-weighted accumulation, validation/export, checkpoint restore and end-to-end
CPU tests. In-memory synthetic Task 2 records use actual Task 3 Dataset/Collator.
Additional disk tests exercise the actual Task 2 adapter on temporary synthetic
participants TSV, EEG NPZ caches and fixed split CSV, then train, validate, test,
export, save and reload. Fixed split bytes remain unchanged. Synthetic sex mapping
is explicit and applies only to the authored fixture, never to private real data.

Independent read-only agent review reproduced and corrected two A3 defects:

1. Missing Git provenance field could previously raise after weight copying;
   manifest validation now happens before mutation, with malformed-manifest tests.
2. Adding optional M3 settings initially invalidated old M1 config digests;
   M1-specific normalization and a pre-A3-schema regression preserve compatibility.

Final artifact review also corrected application output/checkpoint ignore coverage.
An initial disk smoke fixture requested ten seeds while its authored split had two;
the test configuration now explicitly selects those two existing seeds. No split
or existing test was weakened. Independent follow-up confirmed the two fixes.

Exact commands from repository root, existing environment, no installation:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_fusion_b_trainer.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_*.py
```

Final A3 result: **58 passed, 0 failed, 0 skipped** (28 retained warnings).
Full relevant regression: **633 passed, 0 failed, 1 skipped** (63 retained
warnings). The single skip is the pre-existing opt-in actual GPU NF4 test,
requiring CUDA, bitsandbytes and an authorized cached model. Warnings concern
PEFT GPT2 Conv1D orientation and Transformers checkpointing deprecation.
Checks also include
`git diff --check`, scope/credential/artifact inspection, explicit staging and
`git diff --cached --check` before the requested commit.

## Seven principles and Phase A integration

| Principle | Evidence |
|---|---|
| Modularity | A1 modules, A2 model, Task 6 loss/engine/metrics/export, separate M3 checkpoint APIs |
| Configurability | Existing optimizer/precision/seed controls plus independently overrideable group rates |
| Backward compatibility | Full regression; old M1 checkpoints; unchanged caches/splits/baselines and M1 EEG independence |
| Reproducibility | Explicit Python/NumPy/torch seed utility, seeded loader generator, config/Git/model provenance |
| Incremental development | One step, then evaluation, checkpoint, disk integration, independent review and regression |
| Ablation-first | Separate role groups and reusable modules; no hidden architectural changes |
| No redundant experiments | Cached-feature contracts and synthetic fixtures; no encoder/baseline/CV reruns |

A1 supplies differentiable subject EEG pooling and LLM-space projection. A2 inserts
these prefixes and clinical text into one causal LLM with a fresh M3 regression
head. A3 adds subject-level Huber/AdamW execution, explicit evaluation, compatible
export and complete lightweight M3 inference restoration. Phase A's **synthetic
implementation/verification scope** is complete once final regression passes;
real clinical or pretrained performance is not established.

Before real execution or Phase B comparison, select model ID/pinned revision,
verify license/access/local assets, actual inputs_embeds backend/mask behavior,
NF4/PEFT/BF16/checkpointing gradient paths and measured peak VRAM. Verify persistent
private storage, patient/visit linkage, source clinical units/sex mapping,
cache provenance/dimensions, fixed assignments and common M0/M1/M3 cohort. Phase C
owns the inner-validation/model-selection protocol, patience=5 and 10-seed ×
5-fold runner. None of those later implementations is started here. No
cross-device bitwise reproducibility is promised.
