# Phase B — Architecture A (M2), representation-level EEG–clinical fusion

## Scope, prerequisite gate and milestones

Implements current subject-level PD MoCA regression, not longitudinal decline.
Began on clean `feat/multimodal-framework`, HEAD and actual remote both at A3
`ec3948c4890f1019b7a8ce6ddb49ba7d8b378703`. The earlier SSH failure did not remain
an unpushed prerequisite: a read-only HTTPS remote check confirmed A3 was present.
A1/A2/A3 modules/docs, Tasks 0–6, actual config/splits/Dataset/Collator,
M1 suffix selection and A3 optimizer/evaluation/checkpoint APIs were reviewed.
Before source changes, A3 tests passed: **58 passed, 0 failed, 0 skipped**.
No unresolved critical A3 defect was found.

Milestones were sequential:

1. B1 independent representations and M1 regression gate: **63 passed**.
2. B2 external fusion plus existing A2 regression gate: **126 passed**.
3. B3 initial M2/M1/M3 training gate: **103 passed**; expanded B1–B3 gate:
   **101 new Phase B tests passed** before final review additions.
4. B4 independent review, precision/feature-set boundaries and full regression.

No Architecture B redesign, Phase C/D runner, early stopping, sweep, real-model
training, EEG extraction, preprocessing, baseline rerun or split generation is
introduced. Original fixed split files and all earlier tests remain unchanged.
Shared integration intent is recorded in `docs/requests/fusion.md`.

## Changed-file manifest

New files:

- `medley_pd/multimodal/representations.py`
- `medley_pd/multimodal/fusion_a.py`
- `medley_pd/multimodal/fusion_a_trainer.py`
- `medley_pd/multimodal/fusion_a_checkpoint.py`
- `medley_pd/multimodal/training_utils.py`
- `medley_pd/multimodal/inference_checkpoint.py`
- `medley_pd/tests/test_multimodal_fusion_a.py`
- `medley_pd/tests/test_multimodal_fusion_a_trainer.py`
- `docs/multimodal_phase_b_fusion_a.md`

Necessary shared changes:

- `medley_pd/multimodal/config.py`: optional A settings and validation.
- `medley_pd/multimodal/clinical_regressor.py`: shared suffix-state helper.
- `medley_pd/multimodal/clinical_trainer.py`: overridable model device hook.
- `medley_pd/multimodal/clinical_checkpoint.py`: architecture-aware legacy digest normalization.
- `medley_pd/multimodal/fusion_b_trainer.py`: wrapper over extracted A3 helpers.
- `medley_pd/multimodal/fusion_b_checkpoint.py`: wrapper over extracted serialization engine.
- `docs/requests/fusion.md`: authorized integration request record.

## Architecture and APIs

```text
Task 3 cached EEG [B,T,D_e]             Approved clinical dictionaries
        |                                           |
A1 identity adapter + query pooling        Task 5 exact English serializer
        |                                           |
    [B,K,D_p]                          Task 6 tokenizer + suffix ONCE
        |                                           |
mean of K learned queries                     Task 4 text-only LLM
        |                                           |
     [B,D_p]                            final suffix hidden state [B,H]
        |                                           |
Linear(D_p,E)                                 Linear(H,C)
        |                                           |
      [B,E]                                      [B,C]
        +-------------------+-----------------------+
                            |
                    concat([EEG, clinical])
                         [B,E+C]
                            |
              LayerNorm(E+C) -> Linear(E+C,F)
                 -> GELU -> Dropout -> Linear(F,1)
                            |
                   raw float32 MoCA [B]
```

Defaults: D_e=3800 (CBraMod) or 3420 (S-CEReBrO), K=4, D_p=256,
E=C=F=256, four pooling heads and fusion dropout=0.1. H is derived from the
actual supplied LLM embedding, never guessed. Exact default fusion layers:
`LayerNorm(512) -> Linear(512,256) -> GELU -> Dropout(0.1) -> Linear(256,1)`.
The two preceding branch projections are separate and there is no extra head
after the final linear. No sigmoid, softmax, clipping or generation objective.

From repository root, standalone imports use `PYTHONPATH=medley_pd`.

| Module | Public API / purpose |
|---|---|
| `multimodal/representations.py` | `MeanQueryAggregation(kind='mean')`, `RepresentationProjection(input_dim, output_dim, device=None)` |
| same | `EEGRepresentationEncoder(config, device=None)`, `ClinicalRepresentationEncoder(backbone)` |
| `multimodal/clinical_regressor.py` | `clinical_prediction_state(backbone, input_ids=..., attention_mask=..., prediction_positions=...)` |
| `multimodal/fusion_a.py` | `RepresentationFusionHead(E,C,F,dropout,device=None)`, `FusionARepresentations(eeg,clinical)` |
| same | `EEGClinicalRepresentationFusionRegressor(backbone)` |
| `multimodal/fusion_a_trainer.py` | `FusionATrainer`, `FusionAEvaluation`, `build_optimizer`, reused `seed_training` |
| `multimodal/fusion_a_checkpoint.py` | `export_predictions`, `save_inference_checkpoint`, `load_inference_checkpoint` |

Model `extract_representations(eeg_features=..., eeg_mask=..., clinical=...,
artifact_flags=None)` returns the two independent branch vectors. Forward with
those same keyword arguments returns `[B]`. Targets/IDs/subgroups are never model
arguments. `clinical_encoder.encode_hidden(clinical)` exposes reusable `[B,H]`
representations independently of projection/fusion.

```python
from multimodal.fusion_a import EEGClinicalRepresentationFusionRegressor
from multimodal.fusion_a_trainer import FusionATrainer, seed_training
from multimodal.fusion_a_checkpoint import (
    export_predictions, save_inference_checkpoint, load_inference_checkpoint,
)

# Resolve fusion_a.json and call seed_training(configured_seed) BEFORE preparing
# an independent Task 4 backbone and creating the M2 model/DataLoader.
model = EEGClinicalRepresentationFusionRegressor(prepared_backbone)
trainer = FusionATrainer(model, split=adapter.get_split(seed, fold),
                        train_subjects=explicit_train_ids,
                        validation_subjects=explicit_withheld_outer_train_ids)
info = trainer.train_epoch(train_loader)
validation = trainer.evaluate(validation_loader)
test = trainer.evaluate(test_loader, role='test')
export_predictions(test, explicit_private_csv_path)
save_inference_checkpoint(model, new_private_checkpoint_directory,
                          epochs_completed=trainer.epochs_completed,
                          optimizer_steps=trainer.optimizer_steps)
# Reconstruct compatible Task 4 backbone/tokenizer and M2 model, then:
load_inference_checkpoint(reconstructed_model, private_checkpoint_directory)
```

The supplied backbone is used directly, not cloned or reloaded. Caller must
prepare independent backbones for M1/M2/M3 training; reusing the same object
would explicitly share weights. No fitted M1 or M3 head is reused.

## EEG and clinical boundaries

EEG reuses A1 `EEGFeatureAdapter` and `LearnedQueryAttentionPooling` without
changing masking. `eeg_mask=True` denotes a real window. Artifact flags remain
validated metadata, never padding or quality weighting. Pooling replaces padded
values before projection and masks their attention; padded EEG gradients are
zero, including when padding contains NaN. All K pooled queries are valid, even
T<K. Mean aggregation acts on K, never on raw windows and never using the original
window mask across K. K=1/4/7 and both feature widths are tested. M2 has its own
Linear(D_p,E); it does not reuse A1's EEG-to-LLM-width projector.

Clinical encoding reuses Task 5 serializer and Task 6 tokenizer unchanged. Its
LLM receives clinical token IDs only: EEG soft tokens and M3 combined-sequence
forward are absent. M1 suffix hidden selection was factored into one helper,
retaining Task 4 selection validation and final-real-token check. Left/right
padding, three-token suffix, EOS-as-PAD and all-missing/suffix-only descriptions
are tested on tiny GPT2/Qwen2/Gemma. Target/alias/ID/subgroup rejection is inherited
from the approved serializer. Source numeric-sex mappings remain explicitly
verified inputs; the disk fixture's mapping is synthetic, not a real-data claim.

## Training, gradients and precision

Task 6 `ClinicalTrainer` remains the shared explicit-epoch engine. A device hook
supports an MLP head without making M1 read EEG. `training_utils.py` extracts A3's
role optimizer and EEG placement code. M2 and M3 each supply their own roles;
neither depends on the other's model internals. Accumulation/evaluation are not
copied. M1 still reads only clinical fields for model input.

Huber loss uses original MoCA units, default delta=1, identical finite `[B]`
prediction/target shapes and no broadcasting. Each loss term is one subject,
regardless of window count. Training predictions are unrestricted and unnormalized.
AdamW groups are ordered EEG pooling, EEG projection, clinical projection,
fusion, optional parameterized feature adapter, then LoRA. Each trainable weight
must appear exactly once; missing/duplicate/unexpected base parameters and frozen
new modules fail. Default new-module rates are 1e-4, LoRA 2e-5, weight decay .01
(including biases/norms, preserving A3 conventions).

Frozen: EEG pooling/projection, clinical projection and fusion train; LLM base
stays frozen. LoRA additionally trains Task 4 adapters. Huber backward tests
observe finite gradients in every intended new module and finite adapter
gradients with nonzero adapter updates. Zero-initialized LoRA factors need not
all have nonzero gradients at the first step. A single actual EEG window has no
attention choice and can legitimately produce zero query gradients; nonzero
query tests use multiple distinct real windows. No global no_grad wraps training.

New modules are FP32 on the supplied embedding device, even under a changed
PyTorch global default dtype. Standalone branches require matching devices;
projection casts dtype differentiably without moving tensors. Clinical hidden
states can be reduced precision and are cast to clinical projection dtype.
Fusion casts both projected vectors to its norm dtype before concatenating, then
checks finiteness. Trainers explicitly place EEG/flags/masks/targets; forward
never moves or globally casts the LLM. Evaluation uses no_grad and restores mode.
Tiny CPU BF16/FP16 clinical hidden-to-projection casts are tested, not real GPU
QLoRA training. CUDA BF16 autocast remains Task 6's hardware-gated option.

Accumulation sums per-subject Huber losses and divides accumulated gradients by
actual contributing subject counts immediately before clipping/step. Unequal
microbatch sizes, final partial groups, update counts and cleanup are tested in
Frozen/LoRA against explicit effective batches. Gradient tolerances:
`rtol=2e-4, atol=2e-6`; parameter/eval prediction comparison:
`rtol=2e-5, atol=5e-6`. Controlled comparisons use zero dropout; FP32 batched kernel
order and Adam near-zero gradients preclude a bitwise guarantee. Failed epochs
are not transactional: earlier successful updates remain, are counted, and
remaining gradients are cleared. No optimizer/RNG training-resume state is saved.

## Configuration

Existing `medley_pd/configs/multimodal/fusion_a.json` still selects M2 through
Task 1 inheritance. Only missing optional fields were added to `config.py`:

| Setting | Default / supported scope |
|---|---|
| `fusion_a.query_aggregation` | `mean` only |
| `fusion_a.eeg_projection_dim` | 256, positive integer |
| `fusion_a.clinical_projection_dim` | 256, positive integer |
| `fusion_a.layers` | 2 only |
| `fusion_a.layer_norm` | true only |
| `fusion_a.activation` | gelu only |
| `fusion_a.lr_pooling` | null inherits `training.lr_new_modules` |
| `fusion_a.lr_eeg_projection` | same |
| `fusion_a.lr_clinical_projection` | same |
| `fusion_a.lr_fusion` | same |
| `fusion_a.lr_feature_adapter` | reserved OFF adapter; same LR inheritance |

Reuses `fusion.fusion_dim=256`, `fusion.dropout=.1`, existing EEG encoder/K/width/
heads, C1, QLoRA/NF4/BF16 configuration declarations, Huber delta, batch_size=1,
accumulation=8, lr_lora and clinical_regression seed/precision fields. Callers
control loader batch size/generator and epoch counts; trainer does not construct
a runner from max_epochs. Group LR overrides require positive finite numbers.
Inheritance, overrides and snapshots are tested, including different E/C widths.

Unsupported aggregation, layer count, activation, missing LayerNorm, numeric skip,
non-attention sequence pooling and adapter ON fail explicitly. Adapter ON remains
A1's deferred Phase D boundary. No alternate poolers/ablation loops were added.
M1/M3 digests exclude A-only options so adding optional defaults does not reject
older checkpoints; M1 supports prior raw and canonical snapshot digests while
still checking normalized settings and other identity fields. Earlier tests were
not modified or weakened.

## Splits, metrics and prediction export

Requires Task 2 `SubjectSplit`, normally from `adapter.get_split(seed,fold)`.
Train/validation are explicit unique disjoint outer-train subsets; test IDs cannot
enter either. No fabricated validation assignment or checkpoint selection exists.
Dataset membership and observed coverage reject duplicate/missing/unexpected
subjects; one row is one prediction. Synthetic disk tests verify exact fixed-split
bytes are unchanged. M1/M2/M3 through Task 3 share EEG-cache-backed eligibility;
missing caches error rather than silently enlarging the clinical-only cohort.
Real eligible IDs, patient/visit linkage and cache availability still need audit.

Existing regression metric definitions provide n, MAE=mean(abs(yhat-y)) and
RMSE=sqrt(mean((yhat-y)^2)). Validation changes neither weights nor optimizer
state. Explicit role='test' evaluates outer test, never automatically selects
checkpoints. Exports reuse columns
`seed,fold,subject,subgroup,model,y,yhat,target`, with original-unit y==target.
Identifier is `fusion_a_llm:C0|C1|C2`, distinct from M1/M3. Seeds/folds come from
the explicit split, order from the loader. CSV/gzip creation is exclusive, private
and opt-in; existing baseline/M1/M3 outputs cannot be overwritten. No patient rows
are logged or committed. All test artifacts use temporary directories.

## Inference checkpoint format

`inference_checkpoint.py` extracts A3's engine; public M3 wrappers/format/scope
remain intact. M2 format is `medley_m2_inference_v1`. Saved role scope includes
EEG pooling, EEG projection, clinical projection, fusion, parameterized adapter
if available, and LoRA; pretrained base is excluded even if the caller later
freezes all inference parameters.

Files: trainable.safetensors, resolved_config.json and last-written manifest.json.
Identity includes model ID/pinned revision, adaptation, full relevant effective
config (encoder/K/aggregation/E/C/F/layers/dropout/clinical set/seed), base LLM
configuration and tokenizer fingerprints, small synthetic base fingerprint,
Git HEAD when available, counters and tensor checksum. HEAD does not attest a
clean working tree. Large injected bases require explicit model identity and
pretrained saves require pinned revisions.

Caller reconstructs the same base/tokenizer/config before load. Format,
architecture, dimensions, K, tuning, tokenizer, base, hashes, counters, Git-field
types, tensor scope/shapes/dtypes/finiteness are checked before copying weights.
M3 checkpoint cannot silently load into M2. Bad metadata and tensors leave the
destination unchanged. No pickle, dynamic imports or downloads are used. Hashes
detect corruption/mismatch, not arbitrary artifact authenticity. Existing
checkpoint directories are never overwritten. Application artifacts must use
ignored outputs/ or checkpoints/, or explicitly supplied external private paths.

This is inference persistence, not full training resumption: optimizer/scheduler/
RNG/sampler state is absent. Eval predictions after restoration agree at
`rtol=1e-6, atol=1e-7`; inference freezing can switch PyTorch MHA execution paths.

## Verification, independent review and results

Tests use synthetic cached-feature tensors, missing clinical values, variable
lengths and artifact patterns with tiny local causal models, CPU and offline
mode. Disk smoke tests use actual Task 2 adapter reading authored temporary TSV,
NPZ and split CSV, actual Task 3 Dataset/Collator, then pooling/aggregation,
independent projections, fusion, Huber backward/AdamW, validation/test metrics,
export and checkpoint restoration. Synthetic metrics are not clinical evidence.

B4 independent read-only source review found no critical defect; a separate
reviewer run passed **37 tests, 64 deselected** on gradients, accumulation,
modality connectivity, mismatch rejection and pre-B M3 restore. Final boundary
checks also cover C0/C1/C2 and dtype-overflow/global-default-dtype behavior.

Development corrections: the initial batch-size-one nonzero query-gradient test
had one real EEG window; it now uses a subject with multiple real windows while
retaining batch-size-one coverage. Old checkpoint compatibility was preserved
proactively through digest normalization and focused pre-B M3/pre-A3 M1 tests.
No architecture specification change or weakened earlier test was required.

Exact commands from repository root in the existing environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_fusion_a.py medley_pd/tests/test_multimodal_fusion_a_trainer.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_*.py
```

Final Phase B result: **106 passed, 0 failed, 0 skipped** (46 retained warnings).
Final full relevant regression: **739 passed, 0 failed, 1 skipped** (109 retained warnings).
Retained warnings concern
PEFT GPT2 Conv1D orientation and Transformers checkpointing deprecation. The
existing opt-in actual NF4 GPU test remains separately skipped, not passed.
Whitespace/staged scope, private artifact/credential checks accompany testing.
Ruff/Black are absent; no lint dependency was installed.

## Comparison readiness and seven principles

M2 fuses projected subject vectors outside the LLM; M3 places K EEG tokens before
clinical text inside the LLM and uses one linear head. M2 uses L physical text
positions; M3 uses K+L and shared causal contextualization. Their clinical policy,
Task 4 backbone settings, A1 pooler, feature cache, fixed splits, Huber objective,
metrics/export and eligible subject population can be matched by configuration.
Separate model instances are required and no fitted weights are silently shared.

Default new-module counts, for D_e=3800, D_p=E=C=F=256 and K=4, excluding LoRA:
M2 = 1,435,905 + 256H; M3 = 1,237,249 + 260H. Actual measured local tiny GPT2
H=16 comparison (not a pretrained medical model):

| Mode | M2 trainable | M3 trainable | LoRA contribution per model |
|---|---:|---:|---:|
| Frozen | 1,440,001 | 1,241,409 | 0 |
| LoRA | 1,442,049 | 1,243,457 | 2,048 |

M2 adds branch projections and fusion capacity; M3 has a different projector and
causal sequence. Matched experiments cannot attribute differences solely to
fusion location without considering capacity, aggregation, dropout, initialization,
sequence length and optimization dynamics. No performance comparison was run.

| Principle | Concrete enforcement |
|---|---|
| Modularity | Separate aggregation, projections, text helper, branches, fusion head, loss, trainer and checkpoint scopes |
| Configurability | Single Task 1 schema/defaults/inheritance/overrides/snapshots, explicit unsupported-option errors |
| Backward compatibility | M1/M3 wrappers and behavior preserved, old checkpoint tests, full regression, unchanged splits/caches/baselines |
| Reproducibility | Explicit seed utility before construction, seeded loader generator, mode-controlled comparisons, config/model/Git provenance |
| Incremental development | B1 gate before fusion; B2 before training; B3 before B4 independent review and full suite |
| Ablation-first | Identifiable K/aggregation/projections/adaptation/fusion/feature-set/encoder/numeric-skip interfaces; alternatives deferred |
| No redundant experiments | Existing cache/adapter/split/metrics reused; synthetic checks only, no encoder or baseline/CV rerun |

## Real environment, limitations and Phase C dependencies

Read-only environment recheck: torch 2.8.0+cu128, CUDA runtime 12.8,
transformers 5.23.0, peft 0.21.3, accelerate 1.15.0, safetensors 0.8.0.
CUDA and BF16 report available on RTX 4090. **bitsandbytes is absent**;
application configs/local.yaml is absent; default model ID/revision remain null.
No large model downloaded or loaded, and no package/GPU stack change occurred.

**Real pretrained NF4/QLoRA forward/backward and peak training VRAM: NOT VERIFIED.**
CPU QLoRA requests fail Task 4 readiness checks with no LoRA fallback. Device
capability is not proof that 24GB suffices. Actual patient-data schema/units/sex
coding, cache provenance, subject/visit linkage and matched eligibility remain
unverified. No real patient data were read and no clinical improvement is claimed.

Synthetic implementation is ready for a separately scoped Phase C runner after
final regression. Real-model integration requires selecting ID/pinned revision,
license/access/local tokenizer assets and verifying genuine NF4/PEFT/BF16,
checkpointing, gradient behavior, supported backend and measured peak VRAM.
Real-data experiments require persistent private storage and the data/cohort/cache
checks above, then a specified inner-validation/model-selection protocol.
Full 10-seed × 5-fold execution, early stopping and Phase D alternatives are not
implemented here. CUDA/kernel/device/library nondeterminism remains; no cross-device
bitwise reproducibility is promised. Stop at Phase B.
