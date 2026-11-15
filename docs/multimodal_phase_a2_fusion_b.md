# Phase A2 / Task 8 — Architecture B EEG–clinical soft-token regression

## Scope, provenance and prerequisites

A2 implements M3's model forward: EEG soft-token prefixes and clinical/suffix
embeddings enter one shared causal LLM, followed by one linear regression head.
It does not implement Architecture A, an optimizer, full Architecture B trainer,
split generation, cross-validation, checkpoint/export or actual pretrained
training. Those later workflows are not implied by a synthetic backward test.

Started on `feat/multimodal-framework`, clean at A1 commit
`608d2ef76ef69a6a20789835b19de70bbb9cab02`, with the local origin tracking reference
matching. Task 6 and A1 were present. Tasks 0–6/A1 documentation and actual
configuration/data/backbone/serializer/tokenizer/pooling/projector contracts were
reviewed. A1's 78 CPU tests passed before A2 development. Earlier commits are
preserved. Only three new files are introduced:

- `medley_pd/multimodal/fusion_b.py`
- `medley_pd/tests/test_multimodal_fusion_b.py`
- This document.

No previous source, JSON configuration, preprocessing, frozen feature cache,
split, baseline output or evaluation convention is modified.

## Public APIs and composition

From the repository root set `PYTHONPATH=medley_pd` for standalone use.

```python
from multimodal.fusion_b import (
    EEGClinicalSoftTokenRegressor, assemble_soft_token_inputs, FusedInputs,
)
```

| API | Actual behavior |
|---|---|
| `assemble_soft_token_inputs(backbone, eeg_tokens, text_tokens)` | Independent sequence construction, no LLM forward; returns FusedInputs |
| `FusedInputs` | `inputs_embeds [B,K+L,H]`, `attention_mask [B,K+L]`, `prediction_positions [B]` |
| `EEGClinicalSoftTokenRegressor(backbone, pooling=None, projector=None, feature_adapter=None)` | Compose a supplied Task 4 backbone, A1 components and a fresh head |
| `model.prepare_inputs(eeg_features=..., eeg_mask=..., clinical=..., artifact_flags=None)` | Task 5 serialization, Task 6 tokenization, A1 pooling/projection, assembly; no LLM forward |
| `model.forward(...)` / `model(...)` with the same keyword arguments | Exactly one shared Task 4 forward using inputs_embeds, then raw float32 prediction [B] |

The optional injected modules must be implemented A1 component types matching
configured input width, K, internal width, heads, normalization, actual LLM H and
device. Defaults construct independent A1 modules. Unsupported adapter ON or
other configurations fail rather than silently changing the experiment.

Example wiring (not an automatically launched job):

```python
from multimodal.config import CONFIG_DIR, load_multimodal_config
from multimodal.llm_backbone import load_llm_backbone
from multimodal.eeg_pooling import seed_eeg_modules

cfg = load_multimodal_config(CONFIG_DIR / 'fusion_b.json')
# Obtain separate M3 model weights through Task 4 only after authorized access /
# real-model preflight. Default model_id=null intentionally prevents this load.
backbone = load_llm_backbone(cfg)
seed_eeg_modules(backbone.resolved_config)
model = EEGClinicalSoftTokenRegressor(backbone)
# A future trainer must explicitly move these Task 3 EEG tensors to model device.
predictions = model(
    eeg_features=batch['eeg_features'], eeg_mask=batch['eeg_mask'],
    artifact_flags=batch['artifact_flags'], clinical=batch['clinical'],
)
```

`batch['moca']`, subject/visit IDs and subgroup metadata are not model arguments.
No batch sorting, deduplication or patient aggregation occurs in this model;
output row order follows the paired EEG/clinical input rows. A3 must maintain
their subject association and fixed-split eligibility.

M3 always allocates a **fresh Linear(H,1) head**, equivalent to M1's independent
`nn.Linear` construction; there is no fitted M1 head argument or hidden cache.
Use a separately prepared backbone for M1 and M3. The supplied backbone is used
directly, not cloned/reloaded; deliberately passing an already shared object is
caller-requested sharing, not an automatic independent-weight guarantee. Tests
construct separate backbones and verify different parameter storage for models
and heads. Injecting fitted A1 components is likewise explicit, never default.

## Sequence, mask and prediction representation

```text
Task 3 EEG [B,T,D_e] + eeg_mask [B,T]
 -> A1 identity adapter -> learned-query pooling [B,K,D_p]
 -> A1 projector [B,K,H] ------------------------------+
                                                     |
Task 3 clinical dictionaries -> Task 5 exact text      |
 -> Task 6 ClinicalTokenizer (adds suffix ONCE)        |
 -> token IDs/mask/positions -> actual LLM embeddings  |
 -> [B,L,H] -----------------------------------------+
                                                     v
          concat([EEG soft tokens, text embeddings], dim=1)
                  [B,K+L,H], unified mask
                              |
          ONE Task 4 LLM inputs_embeds causal forward
                              |
       final actual suffix token at K + text_position
                              |
                    Linear(H,1) -> [B]
```

There is no external EEG/clinical concatenation MLP. EEG is never converted to
discrete IDs, and no vocabulary token is added. The regression head is the only
post-LLM projection; predictions are continuous, unclipped and unsquashed.

Task 5 is reused verbatim for C0/C1/C2, fixed templates, neutral units, missing
values and target/alias rejection. Task 6 is reused verbatim for
`clinical_text + '\n\nPredicted MoCA:'`, without a second suffix. An all-missing
clinical description stays eligible and yields suffix-only text, still preceded
by valid EEG soft tokens. True targets are never appended to the prompt.

Task 6 contributes its existing no-BOS/EOS-auto-insertion, no-chat-template,
no-truncation policy. Clinical-only text stays within `llm.max_text_tokens`.
The full fused physical length K+L must also fit
`min(llm.max_sequence_tokens, model.max_position_embeddings)`. Exceeding either
limit fails; neither EEG nor text/suffix is silently truncated.

Three mask concepts remain separate:

1. Original `eeg_mask=True` is a real cached window and controls A1 pooling.
2. `artifact_flags` remains validated optional metadata; flagged real windows
   are retained and flag changes do not change predictions.
3. The unified LLM mask is `concat([ones(B,K), text_attention_mask], dim=1)`.
   All K pooled tokens are valid even when T<K.

Text left-padding remains between prefix and actual text, producing masked holes
after valid EEG positions. Right-padding stays after actual text. Task 4 already
derives position IDs by cumulative real-token counts minus one, assigning real
tokens consecutive logical positions despite those holes. The LLM owns the
causal mask and positional encoding; A2 implements neither a custom attention
kernel nor a family-specific position scheme.

Text prediction positions are checked through Task 4's selector and must select
the last real text token. Fused indices are exactly **K + text positions**.
Tests verify the actual hidden vector entering the head against the captured
last-layer states at those indices, including a multi-token suffix, both padding
sides and EOS-as-PAD. Neither PAD nor automatically appended EOS is selected.

## Precision, devices and autograd

New EEG modules and the head are explicitly placed on the supplied backbone
embedding device in FP32. The LLM itself is never moved or cast. Explicitly
injected A1 modules retain their dtype and must already be on the correct
device. EEG features/masks are not implicitly moved: the future trainer must
match their device/dtype or use a supported autocast policy. Newly constructed
Task 6 token tensors are explicitly moved to the embedding device.

Assembly obtains text embeddings from `backbone.embedding_layer(input_ids)`.
It explicitly converts only projected EEG dtype to the **actual text embedding
output dtype**, preserving differentiability, then concatenates. It checks
finiteness both before and after conversion to catch reduced-precision overflow.
Hidden states are converted to the head's FP32 dtype without detaching. No
model-wide dtype conversion, detached prefix or global no_grad wraps the frozen
backbone. Frozen base parameters remain frozen while operations propagate
gradients back to prefix-producing EEG modules.

Genuine CPU tiny-model tests verify:

- GPT2, Qwen2 and Gemma text-only causal families, each Frozen/LoRA and both
  text padding sides: finite nonzero query, input projection, attention,
  projector and head gradients; no base-weight gradients; LoRA gradients present.
- Prefix embedding gradients and preceding clinical token gradients reach the
  selected suffix output, proving actual causal paths inside the shared LLM.
- Original EEG padding and text padding receive zero gradients; genuine EEG
  changes can change predictions. Batch/single predictions agree within stated
  floating tolerances despite left-padding holes.
- Frozen non-reentrant gradient checkpointing preserves EEG gradients.
- Actual tiny GPT2 BF16 and FP16 CPU weights accept FP32 EEG projections through
  explicit casts and backward remains finite. This is not GPU quantization.

These family checks use the eager attention backend and tiny random local
configurations, not pretrained checkpoints or every release of each family.
Composite MedGemma and unsupported inputs_embeds architectures retain Task 4's
rejection/validation boundaries. A real model's backend, tokenizer conventions,
mask-hole behavior and precision must be tested before claiming compatibility.

## Modes, configuration and reproducibility

Task 4 remains the sole Frozen/LoRA/QLoRA preparation path. Constructor checks
base freezing and adapter trainability. Frozen and LoRA were actually tested
with synthetic backward passes. QLoRA must use a genuinely NF4-loaded model via
Task 4's backend/PEFT checks; CPU NF4 requests fail without fallback. No
bitsandbytes installation, pretrained download or actual GPU NF4 forward/backward
was performed. No 4B/medical model compatibility or performance claim follows.

No A2 configuration extension was necessary. `fusion_b.json` already selects
`architecture=fusion_b`, `fusion.type=soft_token` and prefix placement. Existing
fields control encoder (3800/3420), K, pooling width/heads, identity-adapter flag,
projector normalization/output-width constraint, clinical set, LLM mode/model ID,
suffix/head conventions and seed. Architecture A, alternate fusion placement,
unsupported pooling/projectors and adapter ON are not silently supported.

Default K=4, D_p=256, heads=4, adapter OFF and linear projector with the existing
normalization flag are inherited. Actual H comes from the supplied embedding
layer and must match a non-null configured projector width. No dimension is
guessed from model branding. All earlier JSON/config inheritance/CLI/snapshot
behavior is retained because no old source/config is edited.

For reproducibility, seed Python/NumPy/torch before Task 4 preparation as needed
(including LoRA initialization), then call the existing `seed_eeg_modules` before
the deterministic M3 construction sequence. Constructors/forwards do not reset
RNG internally. Effective resolved snapshots remain supported; tests verify
C0/C1/C2 snapshot roundtrips and fresh seeded M3 initialization. Forward dropout,
backend/hardware and mixed-precision rounding can affect repeatability; inference
comparisons use eval mode, and cross-device bit equality is not claimed.

## Incremental verification and independent code review

Development gates were sequential: A1 prerequisites (78 passed), A2.1 assembly
(18 passed), A2.2 model/clinical boundary (36 passed), A2.3 genuine backward and
precision (58 passed), then final review cases and the full regression suite.

Code review followed data from raw module arguments through serialization,
pooling, dtype conversion, concatenation, unified mask, the sole inputs_embeds
forward and actual indexed head input. Review tightened explicit text-position
validation and made new EEG modules FP32 even if a caller changed torch's global
default dtype. Added regression cases cover that dtype context and casting
overflow before LLM execution. No failing critical issue remains in the tested
synthetic contracts; no earlier tests were weakened.

Tests use only CPU, synthetic records/tensors, local tiny random models and the
existing temporary environment. No dependencies, PyTorch/CUDA stack or private
data were changed. Exact commands from repository root:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_fusion_b.py
```

A2: **75 passed, 0 failed, 0 skipped** (8 retained dependency warnings).

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_config.py medley_pd/tests/test_multimodal_data_adapter.py medley_pd/tests/test_multimodal_dataset.py medley_pd/tests/test_multimodal_llm_backbone.py medley_pd/tests/test_multimodal_llm_preflight.py medley_pd/tests/test_multimodal_clinical_serializer.py medley_pd/tests/test_multimodal_clinical_regressor.py medley_pd/tests/test_multimodal_clinical_trainer.py medley_pd/tests/test_multimodal_eeg_pooling.py medley_pd/tests/test_multimodal_eeg_projection.py medley_pd/tests/test_multimodal_fusion_b.py
```

Full suite: **575 passed, 0 failed, 1 skipped**, with 35 retained PEFT Conv1D
orientation / Transformers checkpointing deprecation warnings. The skip is the
existing actual GPU NF4 opt-in test requiring CUDA, bitsandbytes and an explicitly
authorized cached model; it is separate from all A2 CPU checks. Final checks also include
`git diff --check`, exact candidate/staged scope, credential/artifact inspection
and `git diff --cached --check` before the authorized commit.

## Seven development principles

| Principle | Concrete application |
|---|---|
| Modularity | Separate assembly helper; reused A1 EEG modules/Task 4 LLM; fresh linear head; loss remains Task 6; trainer deferred |
| Configurability | Existing resolved configuration only, no new parallel options |
| Backward compatibility | Three new files only; existing tasks, splits, preprocessing/cache/evaluation stay unchanged |
| Reproducibility | Deterministic suffix/assembly, explicit seeds/construction order, effective snapshots |
| Incremental development | Assembly -> forward -> genuine backward -> full review/regression gates |
| Ablation-first | A1 components injectable with matching config; independent K/width/head/clinical-set choices; no Architecture A hardwiring |
| No redundant experiments | Synthetic cached-feature-shaped tensors; no encoder/baseline/CV reruns |

## Real-data status and Phase A3 dependencies

Actual patient data, clinical coding/units, patient/visit linkage, real cache
provenance/dimensions and fixed-split consistency remain unverified. Tests use
actual SubjectRecord/Dataset/Collator objects through an in-memory synthetic
adapter, not real participant/cache loading. M3 inherits Task 2/3's EEG-available
subject cohort; no alternate clinical-only cohort is introduced.

Production model ID/revision remain unselected, bitsandbytes remains absent and
actual NF4 load, QLoRA training forward/backward and peak training VRAM are
unverified. No model weights were downloaded. Numeric sex still requires Task 5's
source-verified mapping, and neutral clinical unit labels remain unchanged.

The model forward is ready for A3's separately scoped trainer work, subject to
those real-run prerequisites. A3 must implement model-specific optimizer groups,
fixed-split subject validation, accumulation/precision/evaluation policies and
appropriate M3 checkpoints. Task 6's M1 checkpoint scope does not save EEG
pooling/projector weights and must not be treated as a complete M3 checkpoint.
No optimizer step, training loop, M3 checkpoint format, model selection or final
experiment runner is implemented here. Stop after A2; A3/Phase B are not started.
