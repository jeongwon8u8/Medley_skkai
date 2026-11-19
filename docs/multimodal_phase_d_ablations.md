# Phase D — Modular controlled ablations

## Scope and prerequisite gate

Phase D implements controlled variants for **current PD MoCA regression**, one
subject per prediction. It is software and synthetic correctness validation,
not evidence for clinical benefit or future cognitive decline prediction. No real
patient ablation, pretrained-model download, EEG extraction, baseline fitting,
full CV or unrestricted sweep was performed.

Work continues on `feat/multimodal-framework` after locally committed C3
`ed8649353871b88cd1353f44bd34d9ce2383fd4c`, C2 `34baf34`, C1 `f022af6`,
Phase B `4368569` and the existing A1–A3/Tasks 0–6. C3 was already documented,
reviewed and tested. The mandatory pre-edit C1/C2/C3 gate passed **164 tests**.
No earlier commits or tests were rewritten. The existing PD MoCA fixed split
uses seeds 0–9, folds 1–5; SHA-256 remains
`5ef85c2056831d71cc04b603b06001739e5d69671a3a57a646597aebb2af6602`.

## D1: effective configuration and OFAT registry

`ablation_registry.resolve_variant(config, model=..., family=..., value=...,
candidate=None)` returns `AblationVariant` with reference/candidate resolved
Task 1 snapshots, semantic differences, model/family, scope, stable ID and
capacity/provenance limitations. `semantic_diff` compares effective nested
configuration values, including Medley's encoder specification, not JSON text.
Registry imports do not load torch, transformers or model weights.

The reference uses the explicitly selected model's C2 configuration. Both sides
receive `settings.ablation.enabled=true` as an implementation-version opt-in.
That common opt-in is not the tested factor. It enables new Phase D structures
while preserving legacy configuration rejection behavior without changing old
model defaults. Named family/value choices and existing `extends` / `--set`
overrides constitute the configuration interface; no second config resolver or
factor Cartesian product is introduced.

A candidate must equal the intended resolved configuration exactly. Undeclared
extra changes, no-op candidates and incompatible model/factor combinations fail.
A caller-supplied candidate snapshot is checked against the exact declared bundle.
The stable `ablation-<sha256>` identity includes both resolved snapshots, model,
family, semantic changes, explicit dependent fields, seed/fold scope and metadata.
`assert_valid()` detects subsequent mutations. Each config/data/model provenance
record remains part of C1/C2 identity at actual execution; planning does not
pretend that unknown input provenance is verified.

| Family | Models | Values and declared bundle | Interpretation |
|---|---|---|---|
| pooling | M2/M3 | attention, mean, max; `eeg.pooling.type` | K=1 reference required; architecture/capacity change |
| feature_adapter | M2/M3 | OFF/ON; `eeg.feature_adapter.enabled` | Adds trainable linear capacity |
| soft_tokens | M3 | K=1/4/8; `eeg.pooling.num_queries` | Learned-query attention only |
| adaptation | M1/M2/M3 | frozen/lora/qlora; tuning + quantization | Explicit training/precision bundle |
| numeric_skip | M2 | OFF/ON; `fusion.numeric_skip` | Explicit additional clinical branch and head capacity |
| clinical_set | M1/M2/M3 | C0/C1/C2; `clinical_set` | Input representation; identical cohort required |
| encoder | M2/M3 | cbramod/scerebro; encoder + width + Medley encoder spec | Input representation/capacity bundle |

The intended baseline is a chosen config, not an empirically established best
model. Existing base configs retain QLoRA/NF4, C1, adapter OFF, numeric skip OFF
and attention K=4. Synthetic tests explicitly select CPU Frozen/LoRA configurations;
there is no automatic substitution for QLoRA. K changes are M3 soft-token
experiments. An M2 attention query-count experiment would require a separately
named family and is intentionally not included in this initial registry.

## D2: pooling and adapter contracts

`eeg_pooling.build_eeg_pooling` selects existing learned-query attention or
`MaskedReductionPooling`. The typed `LearnedQueryAttentionPooling.from_config`
rejects mean/max rather than ignoring a requested alternative.

Attention preserves the original `[B,T,D_e]` plus `True=real [B,T]` mask to
`[B,K,D_p]` contract, including learned query shape, original segment membership,
no pooling positional encoding, separate artifact metadata and finite-input checks.
For K=1/4/8 the existing projector and M3 sequence assembler preserve the complete
clinical suffix, add K valid prefix mask entries and offset prediction positions
by K. No segment policy changes when K changes.

Mean/max reduce **raw valid segment features**, then apply `Linear(D_e,D_p)`,
returning `[B,1,D_p]`. Mean divides by valid segment count. Max masks padded
positions with negative infinity, so zero padding cannot beat negative real
features. Padded NaN/Inf values are sanitized before differentiable operations;
all-masked subjects and nonfinite real features fail. Appending padding or
changing artifact flags cannot change outputs. Gradients reach the projection and
valid inputs. Artifact flags remain metadata, never reliability weights.

No vector is replicated into fictitious informative K tokens. Config validation
requires K=1 for reduction pooling, and OFAT pooling requires K=1 in the reference
as well. Default K=4 attention to K=1 mean is rejected as a pooling-only comparison.
A combined K/pooling study is intentionally unsupported. M2's downstream query
aggregation still produces one EEG representation; M3 reduction produces one
actual LLM EEG prefix token. Attention and reductions have different parameter
capacity, explicitly recorded rather than assumed equal.

`EEGFeatureAdapter` defaults to OFF. It returns the same tensor and has zero
parameters in that mode, preserving the adapter state of existing OFF checkpoints.
ON requires Phase D opt-in. Its residual branch is LayerNorm → Linear down →
GELU → Dropout → Linear up, with a zero-initialized final weight and bias.
The branch uses `eeg.feature_adapter.hidden_dim` (128) and `dropout` (0.1).
Input padding is sanitized before the branch, and padded outputs are zeroed
after residual addition. New modules run in float32 on the LLM device and
consume cached features from frozen EEG encoders.

| Encoder | Previous linear ON | Bottleneck ON (128) | OFF |
|---|---:|---:|---:|
| CBraMod | 14,443,800 | 984,328 | 0 |
| S-CEReBrO | 11,699,820 | 885,908 | 0 |

ON counts are `(2h + 3)D_e + h`. The adapter logs its trainable parameter count
at INFO level. M2/M3 checkpoints from the previous full linear ON adapter
require the previous code to load. Checkpoint loading rejects the legacy
parameter keys before copying any tensors.

## D2: LLM modes and clinical/encoder inputs

Task 4 owns Frozen, LoRA and genuine NF4 QLoRA. Frozen base weights remain frozen;
LoRA adds trainable PEFT parameters; QLoRA requires its existing genuine 4-bit
NF4/CUDA/dependency path and has no silent fallback. New EEG/fusion/head modules
remain trainable. Adapter grouping is delegated to the existing role optimizer,
which verifies exact once-only coverage and frozen base parameters. Adaptation
changes numerical precision/training dynamics as well as parameter count; it is
not interpreted as a capacity-only causal effect. Real NF4 was not exercised.

Task 5 serialization is reused unchanged: C0=(age, sex), C1 adds UPDRS Part III,
disease duration and LEDD, C2 adds PIGD and TD. Field order, numeric formatting,
explicit missingness, verified sex mapping and target/leakage allowlists remain.
MoCA/CTT/UPDRS Total/IDs/subgroup are not predictors. Cohort eligibility and target
alignment are checked using actual C1 sources, not inferred from feature labels.

Encoder switching resolves the existing Medley encoder specification and declared
feature width together: CBraMod=3800, S-CEReBrO=3420. Input processing/channel/spec
validation remains the existing cache adapter's responsibility. Only existing
frozen caches are read. Missing/mismatched caches fail; no extraction is attempted.
Equal subject/target/subgroup population, unchanged saved splits and identical
C1 inner partitions are mandatory before either model is loaded. The encoder
factor includes width-induced parameter differences and representation/preprocessing
changes. Actual extraction history is not certified by byte/decoded-record hashes.

## D2: M2 numeric skip

Numeric skip is supported **only for M2**. M1/M3 requests fail. M3's EEG soft-token
fusion is not bypassed through an undocumented numeric output shortcut.
`numeric_branch.NumericClinicalBranch` projects approved structured values and is
concatenated with clinical and EEG representations before the existing M2 fusion
LayerNorm/MLP. `ablation.numeric_projection_dim` defaults to 16. The head input
width grows by that projection width. OFF creates no branch, consumes no branch
initialization RNG and leaves the original state_dict/output architecture unchanged.

The shared torch-free `numeric_encoding.encode_numeric_record` validates through
the existing serializer before encoding selected fields in the same fixed order:

- Each non-sex field: `[raw_value_or_zero, missing_indicator]`.
- Sex: `[female_indicator, male_indicator, missing_indicator]`, using the verified
  existing mapping rather than treating arbitrary categorical codes as magnitudes.

Zero is an encoding accompanied by an explicit missingness flag, not an invented
clinical measurement. The dimensions are C0=5, C1=11, C2=15. Numeric scaling is
explicitly `ablation.numeric_scaling=none`; no fitted preprocessing or test/cohort
statistics exist. Other scaling policies are rejected. Finite range is checked
again after float32 conversion. This simple raw-scale intervention is not a claim
that raw magnitudes are optimal; a future train-only scaling family would need
new fit artifacts and split provenance.

For **all Phase D-enabled configs, skip OFF and ON**, C1's clinical fingerprint
includes both serialized text and the approved raw numeric encoding. Thus a raw
value change hidden by text significant-digit rounding invalidates reuse. OFF
still does not feed those values numerically into the model. The stronger hash
attests input source equivalence; legacy opt-in-disabled C1 text-only fingerprints
remain unchanged. Older sources may consequently lack sufficient provenance for
strict Phase D reuse/comparison; C3 still reads them, but does not promote them to
verified raw-input equivalence.

## D3: planning, bounded execution and reuse

`ablation_runner.plan_ablation` uses C2 `plan_experiments` for reference and
candidate. It requires explicit seed/fold/output selections, identifies one
family/value and exposes planned run count, both effective configs, changed bundle,
C1 split hash, model/revision, namespace and existing-source status. Default action
is read-only planning: no models, training, weights or report writes.

```bash
PYTHONPATH=src:medley_pd python -m multimodal.ablation_runner \
  --config medley_pd/configs/multimodal/fusion_b.json \
  --model M3 --family soft_tokens --value 8 \
  --seed 0 --fold 1 --output-dir outputs/phase-d --dry-run
```

Pooling-only planning requires `--set eeg.pooling.num_queries=1`. Boolean values
use JSON `true`/`false`; family names and strings may be unquoted CLI text. Existing
Task 1 inheritance/snapshot and repeated `--set` options are reused.

Creating results requires `--execute`, explicit `--role reference`, `--role
candidate` or both, and a sufficient explicit `--max-runs` (default 1). Selection
and count guards run before adapter/model construction. The planner cannot launch
a factor grid. Before execution both sides validate their complete C1 populations,
metadata and inner/outer partitions. An unavailable/different cohort raises an
error before model construction. Valid individual runs delegate to C2
`execute_plan`, retaining readiness checks, safe ignored output roots, locks,
exclusive artifacts, early stopping, best-checkpoint restoration, one held-out test
pass and interrupted-run protection. `--restart-incomplete` explicitly requests
C2's new-attempt behavior; it is not optimizer/RNG training continuation.

`find_completed` searches only selected output/configured manifest roots and
optional explicit `--search-root` registries. It validates native C1/C2 completion,
exact effective architecture/training/input config, seed/fold/split, source bytes
and runtime Python/library/application-source fingerprints. Operational output,
analysis scope and inactive Phase D opt-in differences are normalized; architecture,
training, input and parameter settings are not waived. A matching directory alone
is not completion. Ambiguous matches or corrupt relevant completions fail closed.
Changed code/environment causes `reuse_provenance`, not automatic retraining.
Git commit/dirty metadata alone does not reject equal application source bytes.

Planning labels a valid source `completed_bytes_valid_current_inputs_pending`.
Before actual reuse, fresh C1 validates the current cohort/cache/clinical hashes
and exact train/validation/test membership against the source. Valid results return
without constructing any model. Missing baselines remain missing until the caller
explicitly authorizes their execution role. M0 has no ablation model factory and
is never trained. No source prediction/checkpoint is overwritten. Older-source
compatibility is constrained by recorded evidence, not asserted from equal labels.

## D3: C3 comparisons and report interfaces

`compare_ablation(variant, reference_paths=..., candidate_paths=...,
analysis_config=..., output_dir=None)` attests each completed config against its
exact declared side, then calls C3 ingestion, matching, MAE/RMSE, deltas, clustered
bootstrap and export. Only declared applicable setting paths are allowed as C3
ablation differences. Numeric skip does not waive clinical-source matching.
Unknown/missing provenance remains unavailable. Arbitrary factor results are never
joined on model labels alone.

The report reuses C3's distinct pooled-row versus unweighted mean-run estimands,
`candidate - reference` signed errors, subject-cluster multiplicity and descriptive
percentile intervals. Repeated predictions are not independent patients. The C3
minimum-subject and unsupported repeated-run mean-metric CI policies remain.
There are no p-values or synthetic clinical-superiority claims.

C3 CSV/JSON/Markdown artifacts additionally contain family/variant identifiers,
reference/candidate semantic config changes, measured trainable parameter counts
per completed run, explicit unmeasured computational cost and comparison limitations.
Counts are optional checkpoint top-level metadata, measured at save time. Older
checkpoints without counts report unavailable (`null`), never invented numbers.
GPU timing, VRAM and throughput are not measured. Variant and Phase D evaluator
module hashes are included in the final analysis identity; completion markers
hash every output. Existing C3 exclusive ignored report namespace and source
protection rules remain in force.

Frozen/LoRA synthetic comparisons additionally attest canonical frozen tensor
content, normalizing PEFT wrapper/base-layer parameter names while excluding LoRA
weights. Only explicit Phase D synthetic checkpoints record this extra identity;
legacy inference identities remain unchanged. Unknown bases are not waived.
This is not real NF4/base precision attestation, which still needs genuine preflight.

## Checkpoint compatibility and parameter registration

Optional Phase D settings default to disabled. Disabled optional defaults are
excluded from inference config digests, preserving older checkpoint reconstruction.
Phase D-enabled config, pooling type/K, adapter state and numeric head dimensions
participate in checkpoint/config identity. ON/OFF or incompatible shapes fail
before parameter mutation. New modules use existing M2/M3 checkpoint role selectors
and exact optimizer group validation. Numeric branch weights are included in M2
checkpoints and a separate optimizer role using the new-module learning rate.
No optimizer groups contain repeated or missing trainable parameters.

The old default models produce unchanged outputs when loaded with identical
weights and evaluation settings. Prior tests are retained intact, including legacy
unsupported-option behavior without Phase D opt-in. No extraction, baseline,
clinical serializer, loss, selection or outer-split algorithm is replaced.

## D4: tests and independent review

Development gates were sequential:

1. C1/C2/C3 prerequisite gate: **164 passed**.
2. D1 registry plus existing config: **110 passed** before D2.
3. Mean/max plus existing pooling: **42 passed** before adapter ON.
4. Adapter plus M2/M3 and pooling regressions: **176 passed**.
5. Numeric/K alternatives plus registry/trainers: **147 passed**.
6. Initial D3 synthetic end-to-end integration: **11 passed**.

Final targeted and full regression commands:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  /tmp/medley-task1-tests/bin/python -m pytest -q \
  medley_pd/tests/test_multimodal_ablation_registry.py \
  medley_pd/tests/test_multimodal_ablation_modules.py \
  medley_pd/tests/test_multimodal_ablation_runner.py

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  /tmp/medley-task1-tests/bin/python -m pytest -q tests \
  medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_*.py
```

Tests use tiny local randomly initialized backbones, no pretrained 4B model. Nine
bounded paired smoke cases cover M2 attention reference and mean/max alternatives,
adapter ON, M3 K=1 reference with K=4/8 alternatives, M1 Frozen/LoRA, C1/C0,
M2 numeric skip and both feature widths. Each uses one seed/fold and one epoch,
C2 optimizer steps, checkpoint save/restore, original eight-column export, C3 strict
comparison and protected reports. Extra tests cover K=1/4/8 Frozen/LoRA masks,
suffix positions, backward paths, numeric missingness/leakage/coding, exact OFF
behavior, parameter grouping, checkpoint incompatibility, deterministic OFAT,
unsupported/confounded options, source/split preservation, invalid completions,
cohort mismatch, changed code provenance, and numeric values hidden by text rounding.
A fresh subprocess verifies default planner imports neither torch nor transformers.
Reuse smoke paths replace the model factory with one that raises if called.

An independent read-only reviewer inspected D1–D3, separately reran architecture
and runner tests, and confirmed three fixes: typed attention factory now rejects
reduction requests; the C3 fully controlled flag includes new ablation settings;
constructed trainable counts are persisted rather than always missing. A further
raw-numeric provenance issue was identified during self-review and resolved with
the shared pure encoder and OFF/ON source hashes. Runtime provenance reuse and
D-specific report identity were hardened following review. No reviewer edits were
mixed into implementation. The final independent targeted run passed **54 tests** (24.76s).

## Principles and remaining scientific limits

P1: registry/orchestration, input encoding, pooling, adapter, numeric projection,
existing models/trainers and C3 statistics remain separate. P2: all supported
variants use Task 1 configuration and the family/value interface. P3: opt-in,
legacy digest normalization, unchanged defaults and retained tests preserve prior
contracts. P4: resolved configs, variants, C1/C2 hashes, run seeds and report
identities are recorded. P5: the gates above enforce sequential development.
P6: exact semantic OFAT checks reject confounded changes; dependent adaptation and
encoder fields are explicit bundles. P7: frozen caches, validated completed runs,
C2 training and C3 evaluation are reused; no baseline or EEG extractor is invoked.

Implementation/synthetic correctness does not establish real-model readiness or
actual ablation results. Real execution still requires explicit LLM ID/immutable
revision, access/license eligibility, local assets, actual NF4 forward/backward,
GPU memory measurement, verified EEG cache/spec/extraction provenance, clinical
units/coding and subject/visit linkage, common comparison cohort, fixed-split
compatibility and baseline prediction/training provenance. None is inferred from
CPU smoke success. No actual QLoRA results, VRAM or clinical improvement are claimed.

Recommended separately authorized real execution order: first validate one frozen
baseline and its provenance; then one compatible pooling or K factor; then adapter,
clinical set and encoder factors with common populations; then numeric skip; then
Frozen/LoRA/genuine QLoRA under verified precision/hardware. Reuse each verified
baseline. Expand seed/fold scope only after bounded preflight and a prespecified
C3 aggregation/uncertainty protocol. Do not run an initial Cartesian product.

## Final verification record

Final targeted command above: **54 passed**, 54 warnings, 23.78s.
Final full regression command above: **957 passed, 1 skipped**, 286 warnings,
73.98s. There were no failures. The skip is the explicitly opt-in real NF4 GPU
integration test (requires CUDA, bitsandbytes and explicit `MEDLEY_TEST_CACHED_LLM`
local-model opt-in): actual bitsandbytes/NF4 CUDA execution was not authorized or
exercised by this CPU synthetic task. It is not counted as a pass. Warnings are
existing PEFT Conv1D orientation, Transformers gradient-checkpointing deprecation
and pandas round-trip CSV parser deprecation messages. `git diff --check` and
AST syntax checks passed; the original PD MoCA split hash was rechecked unchanged.

Changed implementation files are the new registry, ablation runner, pure numeric
encoder and numeric branch; minimal existing config, pooling, representation,
M2/M3 construction, optimizer/checkpoint and C1/C3 provenance/report integration.
Three new test modules and this document accompany them. No patient-level data,
cache, model weights, generated predictions or result directories are committed.
C3 itself remains in its earlier separate commit. The requested Phase D commit
and SSH push outcome are reported after execution, outside this artifact.
