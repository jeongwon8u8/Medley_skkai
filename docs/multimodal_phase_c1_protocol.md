# Phase C1 — Reproducible multimodal experiment protocol

## Scope and prerequisite confirmation

C1 supplies read-only outer split access, explicit subject/cohort checks,
deterministic inner validation, seed helpers and private experiment manifests.
It does not fit M0/M1/M2/M3, extract EEG, generate outer folds, run CV, select
checkpoints or compute statistical comparisons. C2/C3/Phase D are not implemented.
The task is current original-unit PD MoCA regression, one subject per observation,
not longitudinal cognitive decline forecasting.

Started on clean `feat/multimodal-framework` at Phase B
`436856932e57033fc130d3da31e179ce568d00b0`. Existing Task 0–6, A1–A3/B docs and
actual config/adapter/Dataset/trainer/evaluation/checkpoint contracts were
reviewed. Phase B prerequisite gate: **106 passed, 0 failed, 0 skipped**. No
critical prerequisite was absent. Prior commits and earlier tests are preserved.
Shared changes are recorded in `docs/requests/fusion.md`.

## Original Medley source and baseline semantics

Fixed source: `medley_pd/splits/<dataset>__<task>.csv`, columns
`subject,seed,fold`. For PD current MoCA the actual source is
`pd_eeg__moca.csv`: 113 subjects, seeds **0–9**, folds **1–5**. Each subject has
one test-fold assignment per seed; all other folds are outer train. The ten seeds
have the same source subject population. No zero-based fold remapping occurs.

Read-only SHA256 of the existing PD MoCA source remains
`5ef85c2056831d71cc04b603b06001739e5d69671a3a57a646597aebb2af6602`.
Existing Iowa MoCA source has 100 subjects and the same seed/fold identifiers,
SHA256 `7c406cec43032efef64153fd6f4fe05642d57dd5f8d6cd913dbb79641c53d8ba`.
These are source-assignment checks, not verification of patient identity/labels.

Original `medley.splits.get_splits` may regenerate/write if the file is absent;
C1 never calls it. It reuses Task 2 `adapter.get_split`, cross-checked against a
separately pinned parse of the actual source CSV. Original `medley.evaluate.prepare`
aligns participant rows to available averaged EEG features and can exclude missing
cache subjects. C1 never invokes that baseline preparation/training path and does
not assume its final historical population equals the source manifest.

Original `medley.evaluate.cv_predict` indexes long rows by
`seed,fold,subject,model` with `subgroup,y,yhat,target`. Model names include mean,
noise, eeg, clinical:<set> and fusion_<name>:<set>. For current MoCA y and target
are the same original-unit score. Existing predictions.csv.gz can be consumed
read-only; no actual historical prediction artifact or real cohort was audited
in this implementation. No baseline is rerun.

Existing `_inner_scores`/`inner_oof` and classifier CV serve stacking or estimator
fitting inside outer train; they do not define a common persistent 20% validation
partition for M1/M2/M3. C1's holdout policy is newly explicit and distinct from
those OOF folds. It does not alter any original baseline training protocol.

## Public components and examples

From repository root use `PYTHONPATH=medley_pd` for standalone imports.

| Module | API |
|---|---|
| `multimodal/experiment_protocol.py` | `ReadOnlyOuterSplits`, `ExperimentProtocol`, `PreparedProtocol`, `InnerPartition` |
| same | `select_inner_validation`, `derive_seed`, `seed_run`, `dataloader_generator`, `ProtocolError` |
| `multimodal/experiment_cohort.py` | `validate_cohort`, `CohortSnapshot`, `SubjectMetadata`, `CohortMatch`, `compare_cohorts` |
| same | `validate_predictions`, `audit_prediction_file` |
| `multimodal/experiment_manifest.py` | `build_manifest`, `ExperimentManifest`, `validate_manifest`, `save_manifest`, `read_manifest` |

```python
from multimodal.experiment_protocol import ExperimentProtocol
from multimodal.experiment_manifest import build_manifest, save_manifest
from multimodal.experiment_cohort import audit_prediction_file

protocol = ExperimentProtocol(config, adapter)
outer = protocol.get_outer_split(seed=selected_seed, fold=selected_fold)
prepared = protocol.prepare(seed=selected_seed, fold=selected_fold)
# Future C2 passes these explicit IDs to existing Dataset/Trainer APIs:
train_ids = prepared.partition.train_subjects
validation_ids = prepared.partition.validation_subjects
test_ids = prepared.partition.test_subjects

manifest = build_manifest(protocol, seed=selected_seed, fold=selected_fold)
public_summary = manifest.public_metadata()  # counts/hashes only, no IDs/paths
private_path = save_manifest(manifest)       # exclusive private directory + JSON

# Existing baseline alignment only, not historical training equivalence:
audit = audit_prediction_file(existing_prediction_path, prepared.cohort,
                              prepared.outer, model='clinical:C1',
                              target_atol=config.settings['protocol']['target_atol'])
```

`prepare()` revalidates all source subjects and the unchanged outer file before
returning a private cohort snapshot and partition. `build_manifest()` prepares
again, intentionally recording currently validated inputs. Direct dataclass
construction is not independent evidence that external records were validated.
`read_manifest()` verifies a saved document's self-consistency, not live source
or extraction authenticity; C2 must revalidate source/cohort before execution.
No runner CLI or automatic artifact discovery is added.

## C1.1 — Outer split invariants

`ReadOnlyOuterSplits` pins actual source bytes, SHA256, ordered assignments and a
canonical semantic assignment fingerprint. Missing/duplicate/malformed rows,
invalid IDs, out-of-range folds, inconsistent seed cohorts and unavailable
configured selections fail. Source bytes are rechecked before and after every
Task 2 split retrieval; changes, disappearance and same-assignment byte edits are
detected. Adapter output must equal exact source train/test assignments and order.
Train/test must be nonempty, unique and disjoint; source subjects must occur in
the adapter cohort. No replacement, exclusion or outer regeneration occurs.

Seed/fold selections use existing config identifiers, not range iteration.
Synthetic tests select noncontiguous seeds [1,7] and folds [2,5] without remapping;
these remain actual Medley-supported identifiers. Existing Task 1's protocol
validation is not relaxed to invent new seeds or change n_folds.

## C1.2 — Source, model-eligible and matched populations

Source cohort is the subject set in the pinned outer CSV. Model-eligible cohort
is explicit adapter.subject_ids plus successful per-subject record validation.
The implemented `exact_source` policy requires the sets to be identical. Missing,
extra or duplicate subjects fail, with aggregate membership mismatch counts in
`ProtocolError.details`; no implicit intersection or eligibility adjustment exists.

For every source subject, `validate_cohort` reads Task 2 SubjectRecord and checks
subject alignment, nonempty finite EEG of configured width, Boolean aligned
artifact flags, finite original-unit MoCA in [0,30], subgroup string-or-None and
Task 5 predictor serialization. Every selected clinical key must exist; None and
all-missing predictors remain valid. Missing columns or ambiguous numeric sex
mapping fail, rather than excluding the subject. No preprocessing is fitted.

A private `CohortSnapshot` retains IDs and subject/target/subgroup metadata with
repr suppression. Fingerprints cover sorted subject/target/subgroup mappings,
actual decoded EEG/flag arrays including shape/dtype/content, and serialized
selected clinical descriptions. These verify the content observed through the
adapter, not EEG extraction history, source clinical correctness, raw recordings,
visit linkage or scientific cache provenance. Cache spec status is reported
separately, including absent/unverified states.

`compare_cohorts({model_name: snapshot, ...})` checks exact subject/target/subgroup
alignment across explicit snapshots. It returns counts/status or raises with a
`CohortMatch` report in strict mode. The scope is population/target/subgroup,
**not training equivalence, clinical feature equivalence or cache equivalence**.
Fingerprint fields expose separate observed cache/clinical inputs for later checks.
Same population under different clinical sets/encoders is not by itself a matched
experimental configuration. Different membership is never silently intersected.

`strict_comparability=false` permits reporting a mismatch; it does not permit
`prepare()` to train a restricted cohort or label mismatched cohorts comparable.
An explicitly restricted matched-cohort policy is not implemented in C1.

Baseline prediction audits require the exact outer-test subject population,
correct target/subgroup mapping and one row per model/seed/subject across folds.
They reject duplicate/missing/extra subjects, invalid non-Boolean integer seed/
fold metadata, nonfinite values and wrong mappings. The explicitly requested
model/seed/fold is selected from a long file without pooling other runs. y and
target are compared against the source within recorded absolute tolerance, default
1e-5 original units to allow existing float32 export rounding; relative tolerance
is zero. No target normalization, label-derived selection or metric computation.

`audit_prediction_file` hashes source bytes before/after a read-only CSV/gzip load.
It preserves string IDs and normalizes only empty subgroup cells. Result reports
`population_aligned=true` and **`training_provenance=not_verified`**. Historical
M0 training/cache/preprocessing and inner validation are not certified by matching
prediction rows. Any real M0/M1/M2/M3 cohort equivalence still requires real data
and relevant artifact provenance, which were not available/validated here.

## C1.3 — Inner validation and seed derivation

For N outer-training subjects, validation size is **ceil(N × fraction)**, default
fraction 0.20 from existing `training.validation_fraction`. Ceiling is computed
exactly from the decimal string using Fraction, independent of global Decimal
context or binary floating multiplication. No clamping: if train or validation
would be empty, fail with `small_cohort`. For N=8/f=.2, validation=2 and train=6;
for N=11/f=.2, validation=3. Fractions outside (0,1) fail.

`derive_seed(seed,fold,purpose='inner_validation')` hashes canonical UTF-8 JSON
of version='medley_seed_v1', actual seed, actual fold and purpose. The first eight
hex digits are interpreted as a big-endian 32-bit integer. No built-in hash(),
global RNG state, labels, subgroup, architecture or feature configuration enters.

Subjects are ranked by SHA256 of canonical JSON with version='medley_inner_rank_v1',
derived_seed and canonical subject ID; ties break by subject string. Select the
lowest ceil(N*f) ranks. Selection membership is independent of source iteration
order; returned partition lists retain original source order. The original outer
train tuple is preserved separately. Test IDs remain exactly the original outer
test tuple, and train+validation exhaust outer train with pairwise disjointness.
There is no target-dependent stratification or other hidden selection.

M1/M2/M3 with the same validated population/seed/fold/fraction therefore receive
identical IDs. Different eligibility is an error, not merely a different RNG
stream. Protocol tests verify global Python/NumPy/torch RNG independence, Decimal
context independence, different actual folds and architecture-independent IDs.

Explicit `seed_run(seed,fold,purpose='training')` reuses Task 6's Python/NumPy/
torch.manual_seed utility (including torch's CUDA seeding). It is not called by
split selection. `dataloader_generator(seed,fold)` returns an independent seeded
CPU torch generator for later loaders without resetting global RNG. Worker seed
initializers/distributed execution and the full training runner remain C2 work.
No expensive deterministic-algorithm mode is enabled. CUDA/library/kernel
nondeterminism and non-bitwise cross-device behavior remain possible.

## C1.4 — Manifest schema and reproducibility

Detailed schema: `medley_experiment_manifest_v1`, top-level fields:
`schema,run_id,identity,subjects,resolved_config,output_namespace`.
Only train/validation/test subject IDs are stored as individual records; **no raw
MoCA labels, clinical values, EEG embeddings or prediction rows** are stored.
The resolved config can contain local private paths and source column mappings,
so detailed manifests stay private. Target values contribute only to a fingerprint.
Hashes are not anonymization or a proof of artifact authenticity.

Identity `medley_c1_identity_v1` records model/architecture, explicit baseline name
when M0, actual seed/fold, current-MoCA target/unit, original split byte hash and
assignment hash/reference, source/eligible sizes and exact-source policy,
partition counts/hash and source-ID hash, cohort/cache/clinical hashes and cache
spec status, verification scope/limits, inner policy/seed/fraction/rounding,
EEG encoder/width, clinical set, LLM ID/revision/tuning, full resolved config hash,
Git revision/dirty status, Python/library versions, application source content
fingerprint and output root. No timestamps, hostname, current random draws or
Python randomized hashes affect identity.

The run ID is `medley-` plus full SHA256 of canonical identity JSON. Canonical
serialization sorts keys, uses fixed separators/UTF-8 and rejects NaN/Inf.
Changing material config/source/content/code/environment yields a different
identity. Same captured inputs in the same environment yield the same identity.
Output namespace is `<configured_root>/<run_id>`. Relative output and protected
source roots are bound when ExperimentProtocol is constructed; changing cwd
later does not redirect them or alter the manifest.

Code provenance records Git HEAD/dirty status plus actual Python source content
under medley_pd/multimodal and medley_pd/medley. This covers the inspected
application source, not all external dependency implementation details. Library
versions are recorded separately. No model is loaded to collect metadata.
LLM revision is explicitly unspecified, mutable-reference supplied/not verified,
or immutable 40-hex SHA supplied/not verified. Supplying a revision does not
establish model access, resolved weights or successful pretrained integration.

M0 requires an explicit name from actual Medley baseline/clinical/fusion options.
Its manifest describes an audit/proposed comparison partition and explicitly
marks historical baseline training and inner validation as **not verified**.
M1/M2/M3 identifiers must agree with clinical_only/fusion_a/fusion_b config.

`validate_manifest`/`read_manifest` check the identity field whitelist, digests,
counts, ID uniqueness/disjointness, model/architecture, actual selected seed/fold,
clinical/EEG/LLM config consistency and revision status, derived seed/fraction/
rounding and deterministic validation membership, source-ID completeness and
namespace structure. Recomputing a hash cannot make contradictory metadata valid.
JSON duplicate keys are rejected. Self-consistency is separate from revalidating
live source files or proving scientific provenance.

`ExperimentManifest.to_dict()` returns a deep copy; public_metadata() is a
whitelist of schema/run ID, model/architecture, seed/fold, counts and nonidentifying
hash/scope fields, without subject IDs, values, config or local paths.
`save_manifest()` exclusively creates its namespace, with mode 0700 and manifest
JSON mode 0600. In-repository paths must be Git-ignored; data/cache/split roots
are rejected even if outside the repo. Existing namespaces are never overwritten.
Explicit external private destinations are supported. No source artifact writes.

## Configuration and compatibility

Existing split source `medley.paths.split_root`, `evaluation.seeds/folds/n_folds`
and `training.validation_fraction` are reused. Optional Task 1 `protocol` fields:

| Field | Default / validation |
|---|---|
| inner_policy | sha256_rank_v1 only |
| cohort_policy | exact_source only |
| strict_comparability | true; Boolean; false reports only |
| manifest_dir | outputs/multimodal_runs; nonempty private path |
| target_atol | 1e-5; finite 0 <= tolerance < .001, original MoCA units |

Old snapshots/presets receive optional defaults. base.json, clinical_only.json,
fusion_a.json and fusion_b.json remain valid. Inheritance, dotted overrides,
invalid settings and snapshots are tested; no new runner CLI exists. No duplicate
validation-fraction or split-source setting was introduced.

Protocol-only settings are excluded from inference checkpoint identity for
M1/M2/M3. Shared checkpoint loading validates raw/canonical historical config
hashes before normalization, preserving earlier checkpoint formats and models.
No model/scientific forward behavior, training/evaluation metric or prior test
was changed. The full existing regression suite remains a gate.

## Tests, independent review and verification

Milestones: C1.1 **8 passed**, C1.2 expanded **28 passed**, C1.3 expanded **43 passed**,
initial C1.4 **57 passed** before independent-review regression additions.
Synthetic temporary files exercise actual Task 2 participants/cache/split reading;
no patient records, historical prediction files, pretrained weights or network/GPU
are required. Separate authored prediction frames/CSV test legacy M0 alignment.

Independent read-only review reproduced two issues, fixed before completion:

1. Caller Decimal context changed ceil counts; exact Fraction-based ceiling and
   a low-precision ROUND_DOWN regression remove that dependence.
2. Rehashed manifests could retain inconsistent model/seed/feature metadata;
   strict semantic cross-validation and unknown-field rejection now catch these.

Additional focused tests reject boolean/float prediction identifiers, preserve
relative-path binding across cwd changes and legacy checkpoint compatibility.
Reviewer follow-up passed **22 focused tests** and found no remaining critical
issue in the reviewed boundaries. Earlier tests were not edited or weakened.

Exact final commands, repository root, existing temporary environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_experiment_protocol.py medley_pd/tests/test_multimodal_experiment_manifest.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_*.py
```

Final C1 result: **72 passed, 0 failed, 0 skipped**.
Final full relevant regression: **811 passed, 0 failed, 1 skipped** (109 retained warnings).
The existing opt-in genuine NF4 GPU
test remains separately skipped, never counted as a pass. Dependency warnings
are existing PEFT GPT2 Conv1D orientation/Transformers checkpointing deprecation.
Whitespace, source syntax, staged scope, credentials/artifact inspection and fixed
split hash checks accompany tests. No dependency installation or environment
change is needed for C1.

## Changed-file manifest

New: experiment_protocol.py, experiment_cohort.py and experiment_manifest.py under
medley_pd/multimodal; test_multimodal_experiment_protocol.py and
test_multimodal_experiment_manifest.py under medley_pd/tests; this document.
Necessary shared edits: multimodal/config.py (optional policies),
multimodal/clinical_checkpoint.py and multimodal/inference_checkpoint.py (legacy
checkpoint compatibility), docs/requests/fusion.md (integration record).
Ten files total; no earlier test, split CSV or generated artifact is included.

## Engineering principles and C2/C3 handoff

| Principle | Implementation evidence |
|---|---|
| Modularity | Outer loader, cohort/prediction audit, inner selector, seed helpers and manifest APIs are separate |
| Configurability | Existing Task 1 schema/overrides, seed/fold selection and fraction; minimal optional policies |
| Backward compatibility | Source files read only, no legacy split writer/encoder/baseline invocation; old checkpoint normalization and full regression |
| Reproducibility | SHA256 seed/rank and canonical fingerprints, exact ceiling, captured source bytes/content/code/runtime, no incidental RNG |
| Incremental development | Outer gate -> cohort gate -> inner gate -> manifest gate -> independent review/regression |
| Ablation-first | Architecture-independent population/partition, config-bound feature/cache/model identities, unsupported policies fail explicitly |
| No redundant experiments | Reads existing adapters/cache/predictions; no training, preprocessing, extraction, baseline rerun or statistical engine |

C2 may consume PreparedProtocol.partition for explicit Dataset/Trainer contexts,
revalidate cohort/source, seed before model/loader construction, and save the
manifest alongside ignored private outputs. It must implement the actual runner,
validation/checkpoint selection, early stopping/resume policies and runtime model
readiness separately. C3 may consume read_manifest plus read-only baseline audits
and explicit CohortMatch reports before any matched statistics; no significance
or baseline comparability is inferred automatically by C1.

Synthetic protocol readiness is verified separately from real-data readiness.
Real runs still require participant/visit identity reconciliation, source coding
and clinical units, validated cache extraction history and persistent private
storage, matching historical baseline subjects/targets/subgroups plus experiment
provenance. Actual pretrained/NF4/QLoRA, model access and peak VRAM remain outside
C1; no real cohort equivalence or clinical performance is claimed. Stop at C1.
