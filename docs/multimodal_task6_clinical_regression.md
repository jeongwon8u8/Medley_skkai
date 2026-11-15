# Task 6 — Clinical-only LLM current-MoCA regression (M1)

## Scope and prerequisites

M1 predicts one continuous current MoCA score per subject from Task 5 clinical
predictors. This is not future decline prediction, classification, EEG inference
or a clinical performance result. Task 6 adds a trainable model, explicit-epoch
training/evaluation utilities, tensor-only inference checkpoints and synthetic
tests. Tasks 7–11, fusion, nested CV and actual pretrained QLoRA training are not
implemented or executed.

Work began on `feat/multimodal-framework` at Task 4 commit `57e3e83`, with Task 5
uncommitted. Its serializer, tests and documentation remain byte-identical
(SHA-256 checked). `multimodal/config.py` is the only overlap: Task 6 adds optional
M1 defaults/schema/validation on top of the existing Task 5 changes. The initial
implementation made no commit/push/merge. A subsequent user-authorized final
review prepares one Task 6 commit without rewriting previous commits.

During final inspection, Task 5 was present as commit `065be16` in the shared
repository. That commit was not created by this Task 6 session; the three Task 5
file hashes remained unchanged. Before the Task 6 commit, pending files comprise only the Task 6
modules/tests/documentation and its 14-line configuration extension.

## Architecture and reused interfaces

```text
Task 2 SubjectRecord -> Task 3 Dataset -> Task 3 Collator
  batch['clinical'] -> Task 5 ClinicalTextSerializer
  -> exact clinical text + '\n\nPredicted MoCA:'
  -> Task 4 tokenizer -> IDs [B,L], mask [B,L], suffix positions [B]
  -> Task 4 LLMBackbone.forward_hidden -> [B,L,H]
  -> select_prediction_state -> [B,H]
  -> Linear(H,1) -> raw float32 predictions [B]
  batch['moca'] [B] ---------------------------------> Huber loss
```

`ClinicalLLMRegressor(backbone)` registers the already prepared Task 4 model;
it does not load another language model. It requires `architecture=clinical_only`
and validates Task 4 trainability. The head is exactly `torch.nn.Linear(H, 1)`,
with H read from the actual embedding width. Head parameters are float32 on the
backbone embedding device. Hidden states are cast to the head dtype without
detaching gradients. No sigmoid, softmax, normalization or clipping is applied.
The existing `prediction.hidden_dim`/`dropout` fields are not used to create an
MLP. Existing `prediction.clip_range` does not clip M1 training or evaluation;
both use raw predictions.

## Public API

From repository root set `PYTHONPATH=medley_pd` (tests manage their own imports).
The following is a manual workflow sketch, not an automatically executed job:

```python
from multimodal.config import CONFIG_DIR, load_multimodal_config
from multimodal.llm_backbone import load_llm_backbone
from multimodal.clinical_regressor import ClinicalLLMRegressor
from multimodal.clinical_trainer import ClinicalTrainer, seed_training
from multimodal.clinical_checkpoint import export_predictions, save_inference_checkpoint

cfg = load_multimodal_config(CONFIG_DIR / 'clinical_only.json')
seed_training(cfg.settings['clinical_regression']['random_seed'])  # before construction
# FIRST resolve actual model access, hardware and source-coding prerequisites.
# Default model_id=null deliberately prevents the following real-model load.
backbone = load_llm_backbone(cfg)  # local-only; no implicit model selection
model = ClinicalLLMRegressor(backbone)
# adapter is Task 2 MedleyDataAdapter; loaders are Task 3 subject loaders.
split = adapter.get_split(seed, fold)
trainer = ClinicalTrainer(model, split=split, train_subjects=explicit_inner_train_ids,
                          validation_subjects=explicit_inner_validation_ids)
for epoch in range(cfg.settings['training']['max_epochs']):
    train_info = trainer.train_epoch(train_loader)
    validation = trainer.evaluate(validation_loader)  # no automatic selection/stopping
# Explicit test evaluation is separate from model selection:
test_result = trainer.evaluate(test_loader, role='test')
export_predictions(test_result, 'outputs/m1/run-unique/predictions.csv.gz')
save_inference_checkpoint(model, 'checkpoints/m1/run-unique',
                          epochs_completed=trainer.epochs_completed,
                          optimizer_steps=trainer.optimizer_steps)
```

The caller supplies loaders and explicit partitions. `train_epoch()` performs
one epoch; there is no automatic training on import, epoch runner, early stopping,
inner split generation or search. Loader construction should use configured
microbatch size. Explicit seed initialization seeds Python, NumPy and torch;
call before model/DataLoader creation and supply seeded samplers/workers as
needed. It is not a guarantee of cross-device bitwise reproducibility.

## Tokenization and prediction position

`ClinicalTokenizer(backbone)(texts)` returns `ClinicalTokens` with CPU int64
`input_ids`, `attention_mask`, `prediction_positions`. `.to(device)` returns a
moved batch. `ClinicalLLMRegressor.forward` accepts only these three keyword
tensors. No EEG, artifact flags, subject metadata or labels are model arguments.

The fixed suffix is `Predicted MoCA:`. For nonempty text, append exactly two
newlines and this suffix. For all-missing clinical text (`''`), tokenize the
suffix alone, retaining the subject without inventing clinical facts. Tokenize
the whole prompt with `add_special_tokens=False, truncation=False`: no automatic
BOS/EOS, chat template, vocabulary additions or embedding resize. This explicit
plain-text policy needs review against the eventually selected pretrained model;
instruction/chat-model prompting is not claimed validated.

The suffix can contain multiple tokens. Its final actual token is identified
before padding and adjusted by the left-padding offset. No last-array-column,
EOS/PAD identity or single-token suffix assumption is used. Task 4's selector
validates positions against the mask, and M1 additionally requires the selected
position to be the final unmasked index. Task 4 supplies cumulative position IDs
so both padding sides work. Even when EOS is explicitly reused as padding, mask
positions distinguish padding from actual tokens.

Both `llm.max_text_tokens` (clinical text alone) and
`min(llm.max_sequence_tokens, model.max_position_embeddings)` (full prompt) are
checked. A suffix that cannot fit fails. Full text plus suffix exceeding budget
also fails; no predictor or suffix is silently truncated. Vocabulary and padding
setup remain owned by Task 4.

Tests use a genuine tiny randomly initialized GPT2 causal LM and a locally built
WordLevel tokenizer. They verify the three-token suffix, left/right padding,
batch/single prediction agreement, empty text, and nonzero gradients from the
suffix representation back to preceding clinical-token embeddings. Changing a
prefix clinical token changes the prediction: causal context is actually used.

## Adaptation, loss and optimizer

- Frozen: Task 4 freezes every pretrained parameter, no adapters; only the head
  is optimized. The shared backbone's autograd behavior remains unchanged.
- LoRA: Task 4/PEFT freezes the base and adds adapters. Both adapters and head
  receive finite gradients and update in CPU synthetic tests.
- QLoRA: uses Task 4's actual NF4 load/verification and PEFT k-bit preparation.
  No separate quantizer/LoRA implementation or mode fallback exists. Injecting a
  full-precision model as QLoRA fails. Actual NF4 execution is NOT VERIFIED.

`moca_huber_loss(predictions, targets, delta=1.0)` requires matching nonempty
floating `[B]` tensors on the same device, finite values, and targets in [0,30].
It prevents accidental broadcasting, retains original target units and supports
mean/sum/none reduction. Trainer delta comes from existing `loss.huber_delta`.

`build_optimizer(model)` constructs AdamW with explicit trainable groups:

| Group | Existing configuration | Default |
|---|---|---|
| Head weight and bias | `training.lr_new_modules` | 1e-4 |
| LoRA weights only | `training.lr_lora` | 2e-5 |
| Both groups' weight decay | `training.weight_decay` | 0.01 |

No frozen base parameter enters the optimizer. Weight decay applies to all
optimized parameters, including head bias; no implicit bias/norm exemption.
Frozen has no adapter group. Unexpected trainable base weights are rejected.

For accumulation, each microbatch backpropagates **summed** Huber loss. At each
configured boundary (default eight microbatches), divide accumulated gradients by
the actual contributing subject count, then clip using
`training.gradient_clip_norm`, check finiteness, step, and clear gradients. The
last partial group uses its own actual sample count. This handles unequal
microbatch sizes and matches a concatenated-batch reference when dropout is off.
Gradients are cleared on exceptions; an epoch is not a transactional rollback of
optimizer steps already completed before a later loader error.
The optimizer-step counter advances after each successful update, including
updates completed before a later batch fails; the epoch counter advances only
after a complete successful epoch.

`clinical_regression.mixed_precision=none` is the CPU default. Explicit
`bfloat16` enables CUDA BF16 autocast only on supported CUDA; CPU requests fail
clearly. No CPU FP16, automatic precision fallback, GradScaler, scheduler or
untested distributed/offload optimization is introduced. Task 4 owns optional
non-reentrant gradient checkpointing. Actual CUDA autocast training remains
unverified; CPU tests validate the explicit rejection path.

## Subject-level splits, evaluation and leakage

Pass a Task 2 `SubjectSplit`, normally `adapter.get_split(seed, fold)`. Its
seed/fold must belong to configured selections. Explicit training and validation
IDs must be unique, disjoint subsets of the **outer train** IDs. Outer test IDs
cannot be used for optimizer updates or validation. No split file is generated,
rewritten or inferred, and no EEG window becomes an independent clinical sample.

Dataset identity lists are checked before an epoch/pass when available. Every
batch is checked against its allowed role; duplicate IDs and missing expected
subjects fail instead of silently shrinking a cohort. Evaluation defaults to the
explicit inner validation IDs. `role='test'` is an explicit evaluation-only pass;
`role='train'` can evaluate assigned training IDs. None of these automatically
select checkpoints or stop training. A caller must not use outer test metrics for
model selection. Full inner-validation protocol belongs to Task 11.

The trainer reads `clinical`, `moca`, `subject_ids` and evaluation `subgroups`.
It never reads EEG tensors/masks/artifact flags. Tests delete those fields and
still train successfully. The unchanged Task 3 Dataset still obtains Task 2
records with EEG caches: therefore real M1's cohort remains restricted to the
same EEG-available subjects. No standalone clinical cohort path is introduced.

Task 5 rejects target/cognitive/metadata keys and aliases inside predictor
dictionaries. Target values travel only to Huber loss or evaluation output;
they are never appended after the suffix. Tests change labels while retaining
clinical predictors and confirm unchanged predictions in evaluation mode.

Evaluation uses `torch.no_grad()`, restores prior model train/eval state and
returns `ClinicalEvaluation(metrics, predictions)` without printing subject-level
data. It reuses `medley.metrics.regression_metrics`, retaining n/MAE/RMSE. Metrics
are calculated from exactly one raw prediction per expected subject; no clipping.

## Prediction export and inference checkpoints

`export_predictions(result, path)` is an explicit opt-in, exclusive-create CSV or
CSV.gz export. It preserves loader identity/order and Medley's actual schema:

`seed, fold, subject, subgroup, model, y, yhat, target`

The stable label is `clinical_llm:C0/C1/C2`. Seed/fold come from the supplied fixed
split. Under Medley's current untransformed regression convention, **target is
the original numeric target value**, equal to y; it is not the string `moca`.
Finite values and duplicate (seed, subject, model) keys are checked. Float output
uses 17 significant digits; gzip has fixed mtime and no embedded filename for
deterministic bytes. Existing files, including baseline files, are never
overwritten. In-repository outputs/checkpoints must lie under ignored `outputs/`
or `checkpoints/`; external private paths are explicitly caller supplied. Test
artifacts stay under pytest temporary directories. No automatic patient logging.

`save_inference_checkpoint(model, path, epochs_completed=0, optimizer_steps=0)`
saves only head/LoRA tensors in `trainable.safetensors`, effective
`resolved_config.json`, and a last-written manifest with progress counters,
model ID/revision, mode, hidden dimension and integrity hashes. No full pretrained
weights or executable pickle are saved. Existing checkpoint directories fail.
Tensor scope is selected by parameter role rather than temporary `requires_grad`
flags, so inference callers can freeze the entire model and still save/restore
the head and adapters without changing those flags.

`load_inference_checkpoint(model, path)` requires an explicitly reconstructed
identical Task 4 backbone/tokenizer, then validates configuration/base-config/
tokenizer identities, weight hash, tensor keys/shapes/dtypes/finiteness before
copying any parameters. It never downloads or reconstructs executable objects
from checkpoint data. Pretrained saves require a pinned revision. Null-ID tiny
synthetic backbones are supported with a frozen-weight fingerprint (limit one
million frozen parameters); the exact synthetic base must be reconstructed or
preserved separately. Tokenizer backend, vocabulary and special/padding policies
are fingerprinted; the pretrained tokenizer must be recoverable from its source.
The synthetic base fingerprint excludes LoRA by role, preserving compatibility
when adapters are frozen for inference.

This initial format is strict about effective configuration/device/model setup;
portable device remapping or migration requires a later explicit design.
Hashes detect corruption/mismatch, not authenticity of malicious files.
Only use authorized checkpoints. These are **lightweight inference checkpoints**,
not training-resume checkpoints: optimizer/scheduler/RNG/sampler states are not
saved/restored. Counters are provenance only. Full deterministic training
resumption is not claimed.

## Configuration integration

All four shipped JSON configurations, inheritance, dotted overrides and older
snapshots remain compatible without editing the JSON files. Existing Task 5
settings remain intact. Only this optional group is added:

```json
{
  "clinical_regression": {
    "prediction_suffix": "Predicted MoCA:",
    "head_type": "linear",
    "mixed_precision": "none",
    "random_seed": 0
  }
}
```

Suffix/head choices are fixed for this initial policy; mixed precision is none or
bfloat16, and seed is an integer in [0,2^32). All other fields are reused: C1,
QLoRA/NF4/BF16 model compute, 256 clinical/512 total token budgets, Huber delta 1,
microbatch 1, accumulation 8, maximum 30 epochs and configured AdamW rates/decay.
Defaults never launch a job. Output paths are explicit API arguments rather than
new configuration flags. The caller uses configured epochs/batch size/seed;
`train_epoch()` itself neither constructs a DataLoader nor launches 30 epochs.

## Synthetic tests and results

All tests use local tiny random weights, synthetic SubjectRecords and the actual
Task 3 Dataset/Collator. The in-memory adapter implements Task 2's public record
interface; this is not a real patient-file loading test. Training uses three or
more assigned subjects, including partial and all-missing clinical records,
explicit withheld validation IDs and a separate test ID. Tests cover Frozen and
LoRA updates, finite losses, unchanged base weights, evaluation/export, head and
adapter save/load equality, split/contamination/coverage rejection, accumulation
equivalence and final partial groups. No meaningful medical learning is claimed.

Exact commands from repository root (existing temporary venv; no installation):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_clinical_regressor.py medley_pd/tests/test_multimodal_clinical_trainer.py
```

Task 6 after final review: **56 passed, 0 failed, 0 skipped** (20 retained dependency warnings).

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_config.py medley_pd/tests/test_multimodal_data_adapter.py medley_pd/tests/test_multimodal_dataset.py medley_pd/tests/test_multimodal_llm_backbone.py medley_pd/tests/test_multimodal_llm_preflight.py medley_pd/tests/test_multimodal_clinical_serializer.py medley_pd/tests/test_multimodal_clinical_regressor.py medley_pd/tests/test_multimodal_clinical_trainer.py
```

Integrated result after final review: **422 passed, 0 failed, 1 skipped**. The skip is the existing
opt-in actual GPU NF4 test requiring CUDA/bitsandbytes/an authorized cached model.
The 27 retained warnings concern PEFT GPT2 Conv1D orientation and a Transformers
checkpointing property deprecation. CPU tests do not claim actual QLoRA training.
`git diff --check` and changed-file scope are checked separately.

Final review reproduced two checkpoint failures (Frozen and LoRA) when the
destination model had `requires_grad_(False)`. Role-based weight selection and
base fingerprinting now allow both save and restore under inference freezing.
The step counter also now records successful updates if a subsequent batch
raises, while the incomplete epoch remains uncounted. Nine additional cases
cover those paths, EOS-as-PAD on both sides, and LoRA accumulation equivalence
including unequal batch sizes and final partial groups. The tiny GPT2 fixture
now sets a valid BOS ID within its vocabulary. No test was removed or weakened.

Review confirmed no target input, EEG consumption, split writes, prediction
clipping or frozen-base optimizer entries. Inference checkpoints remain distinct
from training-resume files. Phase A and subsequent work have not been started.

## Actual environment and real-run readiness

Environment-only Task 4 preflight was run without model inspection/loading:

- Python 3.12.3; torch 2.8.0+cu128; torch CUDA runtime 12.8.
- transformers 5.23.0, peft 0.21.3, accelerate 1.15.0, safetensors 0.8.0.
- bitsandbytes absent; model ID/revision null, no model selected or access tested.
- RTX 4090, 25,252,724,736 total / 24,838,406,144 free VRAM bytes at inspection;
  CUDA and BF16 available. This is available-memory inspection, not a peak-memory
  measurement or a model-fit guarantee.
- Cache filesystem had 49,905,246,208 free bytes; persistent storage remains
  unconfirmed. Disk/VRAM figures are point-in-time observations.

No GPU training, actual NF4 load, pretrained forward/backward, model download,
bitsandbytes installation, CUDA/PyTorch changes, real patient prediction or
medical performance validation occurred.

Before a separately authorized real run: confirm model ID and pinned revision,
license/access and local tokenizer assets; install compatible bitsandbytes under
a reviewed environment plan; verify actual NF4 load and QLoRA forward/backward,
BF16/checkpointing behavior and measured peak VRAM. Verify patient/visit IDs,
clinical units/source sex coding and explicit mapping, fixed-split/cohort/cache
consistency and persistent private storage. Task 5's neutral units and ambiguous
numeric-sex rejection remain in force. No fusion or later experiment protocol
has been started; full validation selection and training-resume machinery remain
outside this Task 6 implementation.
