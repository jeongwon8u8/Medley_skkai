# Phase A1 / Task 7 — Modular EEG pooling and LLM-space projection

## Scope and inspected prerequisites

A1 provides cached-EEG feature adapter, learned-query pooling and LLM-width
projection interfaces. No LLM fusion/forward, Architecture B class, clinical
combination, MoCA loss, trainer, EEG encoder execution or later phase is added.
Synthetic scalar objectives verify autograd, not medical predictive performance.

Started on `feat/multimodal-framework` at Task 6 commit
`b40a7fe124e0c984f7cac721334d03433c08850b`, with a clean working tree and matching
local origin tracking reference. Reviewed Tasks 0–6 documentation/contracts,
resolved configuration, adapter/Dataset/Collator, backbone and clinical-only
model/trainer interfaces, encoder YAML/source and `fusion_b.json`. No earlier
validated source, preprocessing, cache, split, baseline or evaluation output is
changed. The only existing file modified is `multimodal/config.py`.

## Components and public APIs

All paths below are under `medley_pd`. From repository root set
`PYTHONPATH=medley_pd` for standalone use.

| Component | API | Responsibility |
|---|---|---|
| `multimodal/eeg_pooling.py` | `EEGFeatureAdapter(input_dim, enabled=False)` / `.from_config(config)` | Exact identity while OFF; ON fails clearly |
| Same module | `LearnedQueryAttentionPooling(input_dim, num_queries=4, hidden_dim=256, num_heads=4)` / `.from_config(config)` | Learned-query cross-attention over real cached segments |
| Same module | `seed_eeg_modules(config)` | Explicit torch seed before module construction; returns seed |
| `multimodal/eeg_projection.py` | `EEGToLLMProjector(input_dim, output_dim, layer_norm=True)` / `.from_config(config, hidden_dim=None)` | Independent linear projection to actual/explicit LLM width |

Modules remain independent `nn.Module` components; there is no combined fusion
model or automatic model loader. Their ordinary `state_dict()` contains module
weights; no new checkpoint format or training runner is introduced.

```python
from multimodal.config import CONFIG_DIR, load_multimodal_config
from multimodal.eeg_pooling import (
    EEGFeatureAdapter, LearnedQueryAttentionPooling, seed_eeg_modules,
)
from multimodal.eeg_projection import EEGToLLMProjector

config = load_multimodal_config(CONFIG_DIR / 'fusion_b.json')
seed_eeg_modules(config)  # once before the deterministic construction sequence
adapter = EEGFeatureAdapter.from_config(config)
pool = LearnedQueryAttentionPooling.from_config(config)
# Later callers pass the actual Task 4 backbone.hidden_dim here. A1 never loads it.
projector = EEGToLLMProjector.from_config(config, hidden_dim=actual_llm_hidden_dim)

pooled = pool(adapter(batch['eeg_features']), batch['eeg_mask'],
              artifact_flags=batch['artifact_flags'])
eeg_tokens = projector(pooled)
```

## Input, architecture and output contracts

Task 3 emits `eeg_features: float32 [B,T_max,D_e]` and boolean
`eeg_mask`, `artifact_flags: [B,T_max]` on CPU. Subject order is preserved. Other
batch fields (`clinical`, `moca`, IDs/subgroups) are not consumed by these
components. No EEG averaging or truncation precedes learned-query pooling.

Existing encoder source/configuration confirms:

- CBraMod: patch averaging and flattening 19 channels × 200 = **3800** features.
- S-CEReBrO channel pooling: 19 × 180 = **3420** features. Its alternative
  180-wide mean mode remains outside the existing Task 2 multimodal cache contract.

```text
Frozen cached EEG [B,T,D_e]
 -> feature adapter OFF: identity, same tensor object
 -> trainable input Linear(D_e,D_p), D_p=256 by default
 -> K learned query parameters [K,D_p], K=4 by default
 -> MultiheadAttention(query, projected EEG keys/values), heads=4
 -> pooled EEG [B,K,D_p]
 -> separate Linear(D_p,H)
 -> optional LayerNorm(H), honoring existing projector.layer_norm=true
 -> EEG representations in LLM width [B,K,H]
```

There are no extra residual/MLP/clinical embedding layers. Standard PyTorch
MultiheadAttention owns its trainable Q/K/V/output projections. Attention dropout
is fixed at zero in this initial policy; default module train/eval behavior does
not add stochastic attention dropout. Queries initialize normally with standard
deviation `D_p**-0.5`. K may exceed T. No temporal positional encoding is added:
this is set pooling, invariant to permutations that preserve feature/mask pairs.

Projector type `linear` includes an optional final normalization, not an MLP.
With `eeg.projector.layer_norm=false` it is exactly Linear(D_p,H). The flag was
already present and true in the shipped base, and is not silently ignored.

## Mask polarity and artifact semantics

`eeg_mask=True` means a real segment exists, not artifact-free EEG. PyTorch
MultiheadAttention's boolean `key_padding_mask=True` means ignore, so pooling
passes **`key_padding_mask=~eeg_mask`**.

Validation rejects empty batch/time/feature axes, wrong widths/ranks, non-floating
features, non-boolean/misaligned masks, device mismatch and any all-masked row
before attention. Real segment values must be finite. All-zero real features
remain real; values never define mask membership.

Before any trainable operation, padding positions are replaced with zero using
differentiable `masked_fill`. Thus arbitrary padded values, including very large
numbers/NaN/Inf, do not affect output or poison backward. This robustness does
not weaken Task 3's existing finite-data validation: its own padding is still
zero. Padded inputs receive exactly zero gradient; keys/values for padding are
excluded from attention. Appending padding and batching different real lengths
leave each subject's result equivalent within tested floating tolerances.

`artifact_flags` is optional, separately validated boolean [B,T] metadata. Both
true and false real flags are retained; flag changes have no effect on attention
or output. Padded flag values are meaningless. There is no artifact exclusion,
quality score, uncertainty weighting or flag-to-padding conversion.

## Adapter, dimensions, dtype and device

Feature adapter OFF is `nn.Identity`: no parameters, cloning, detach or hidden
MLP. Existing `feature_adapter.hidden_dim=128` and `dropout=0.1` remain reserved
for later experiments. ON is deferred to Phase D: the adapter constructor and
both EEG factory entry paths reject it with `NotImplementedError`. The old
configuration schema still accepts its reserved Boolean for backward
compatibility; accepting a snapshot is not a claim of executable ON behavior.

All constructor dimensions/K/heads must be positive integer values (not bool);
pooling dimension must divide evenly by head count. Projector input width is the
configured pooling width. H is never inferred from a model brand or hardcoded:

- Pass `hidden_dim=backbone.hidden_dim` from the actual Task 4 metadata; or
- explicitly configure positive `eeg.projector.output_dim` for synthetic work or
  a verified external hidden-width contract.
- If both are supplied they must agree. If neither is available, construction
  fails. A2 must cross-check an explicitly configured width with the actual LLM.

No implicit device movement or dtype conversion occurs. Call `.to(device,dtype)`
explicitly for both modules and inputs, or use supported autocast. Outside
autocast, mismatched input/parameter dtypes fail clearly. Outputs must be finite.
Tests cover float32, explicitly converted CPU BF16 modules/input, and CPU BF16
autocast. This is not actual CUDA/QLoRA verification. A2 must explicitly match
the projected output to the chosen LLM embedding dtype; `.to(dtype=...)` preserves
gradient flow. There is no detach or global no_grad in either forward.

## Configuration, ablations and reproducibility

All four existing JSON files are unchanged and load successfully. The existing
configuration controls encoder/width, attention type, K, internal width/heads,
adapter flag and linear-projector/normalization choices. Only two optional
fields are added to `config.py`, injected before override validation and into
old snapshots:

| Setting | Default / supported behavior |
|---|---|
| `eeg.encoder`, `eeg.feature_dim` | cbramod/3800 or scerebro/3420, existing validated pairing |
| `eeg.pooling.type` | `attention` only; mean/max alternatives fail |
| `eeg.pooling.num_queries` | 4; positive K, independent of T |
| `eeg.pooling.hidden_dim` | 256; independently configurable positive dimension |
| `eeg.pooling.num_heads` | 4; positive divisor of hidden dimension |
| `eeg.feature_adapter.enabled` | false identity; ON explicitly deferred |
| `eeg.projector.type` | linear only; MLP alternatives fail |
| `eeg.projector.layer_norm` | true; supports false for pure Linear |
| **`eeg.projector.output_dim`** | **null**; use actual LLM metadata, or positive explicit H |
| **`eeg.seed`** | **0**; integer in [0,2^32), used by explicit initialization helper |

Inheritance, CLI overrides and resolved snapshot roundtrips are tested. Example:

```python
config = load_multimodal_config(CONFIG_DIR / 'fusion_b.json', [
    'eeg.encoder=scerebro', 'eeg.feature_dim=3420',
    'eeg.pooling.num_queries=7', 'eeg.pooling.hidden_dim=128',
    'eeg.pooling.num_heads=8', 'eeg.projector.output_dim=64', 'eeg.seed=13',
])
```

Call `seed_eeg_modules(config)` once before construction. Constructors never
reset the RNG themselves; constructing successive modules consumes its stream.
Identical seed + configuration + construction order produces identical tested
CPU initialization/output. Save the resolved configuration and module weights
for later reproducibility. GPU kernels, hardware/library versions, autocast
rounding, thread settings and operation order may affect numerical results;
cross-device bitwise equality is not claimed. No split or sampling RNG is changed
by module forward calls.

## Tests, smoke integration and internal review

No dependency installation, patient data, EEG extraction, model weights, network
access or GPU is required. Used existing `/tmp/medley-task1-tests` environment;
PyTorch/CUDA and dependency files remain unchanged.

Exact commands from repository root:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_eeg_pooling.py medley_pd/tests/test_multimodal_eeg_projection.py
```

A1 result: **78 passed, 0 failed, 0 skipped**.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_config.py medley_pd/tests/test_multimodal_data_adapter.py medley_pd/tests/test_multimodal_dataset.py medley_pd/tests/test_multimodal_llm_backbone.py medley_pd/tests/test_multimodal_llm_preflight.py medley_pd/tests/test_multimodal_clinical_serializer.py medley_pd/tests/test_multimodal_clinical_regressor.py medley_pd/tests/test_multimodal_clinical_trainer.py medley_pd/tests/test_multimodal_eeg_pooling.py medley_pd/tests/test_multimodal_eeg_projection.py
```

Full regression result: **500 passed, 0 failed, 1 skipped** (27 retained warnings
from existing PEFT Conv1D handling and Transformers checkpointing deprecation).
The skip is the existing opt-in actual GPU NF4 test requiring CUDA,
bitsandbytes and an explicitly authorized local pretrained model. It is not an
A1 test or a claimed quantized execution result.

Smoke tests use three synthetic Task 2 SubjectRecords through the real Task 3
Dataset/Collator for **both encoder widths**: lengths [1,2,3], all real segments
artifact-flagged, [3,4,256] pooling -> [3,4,32] projection. A weighted synthetic
scalar objective backpropagates nonzero finite gradients into learned queries,
input projection, attention, EEG projector and real input positions; padded input
gradients are exactly zero. No target or clinical tensor is read by the modules,
and no LLM is loaded or forwarded. A subprocess verifies importing/constructing
the modules does not import transformers/PEFT/tokenizers.

Internal review reinspected polarity, mask application before trainable layers,
query expansion, feature/metadata separation, output width, normalization, dtype/
device checks, factory/config support and gradient paths. Additional focused
tests cover device mismatch without implicit movement and LLM independence.
No confirmed failing A1 defect remained after review; no old tests were weakened.
`git diff --check`, exact file scope and pre-commit credential/artifact inspection
are required before the authorized commit.

## Seven engineering principles

| Principle | Applied behavior |
|---|---|
| Modularity | Identity adapter, pooling, projector and existing LLM stay separate |
| Configurability | Existing resolved config plus only seed/output_dim extensions |
| Backward compatibility | Four JSONs/old snapshots work; no preprocessing/cache/split/baseline changes |
| Reproducibility | Explicit seed/order, saved resolved settings, documented numerical caveats |
| Incremental development | Component checks followed by Collator smoke and full regression |
| Ablation-first | Encoder, K, pool width/heads, normalization vary; unsupported adapter/pooling/projector options fail |
| No redundant experiments | Synthetic tensors/frozen-cache interface only; no encoder, baseline or full experiment reruns |

## Real-data limits and next interfaces

Only synthetic input tensors/records were tested. Actual cache dimensions/specs,
subject/visit identity, clinical source coding/units, fixed split consistency and
persistent private storage remain unverified. No actual pretrained LLM hidden
width or embedding-input integration is validated in A1; an explicit synthetic H
does not establish compatibility with a real model. Existing model ID/revision
selection, access, bitsandbytes and actual NF4/QLoRA memory/backward checks remain
separate prerequisites for real execution.

A2 can consume the [B,K,H] output and Task 4 backbone metadata after explicitly
checking width/device/dtype and defining its combined token/mask policy. Future
Architecture A may consume pooled [B,K,D_p] through its own representation
contract. Neither integration, trainer nor attention-weight interpretation is
implemented here. There is no A1 synthetic-test blocker to preparing A2; real
model/data readiness is not claimed. Stop at A1; A2/A3 have not been started.
