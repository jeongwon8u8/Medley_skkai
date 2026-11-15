# Task 2 — Medley data adapter

Implemented 2026-11-15 (Asia/Seoul) on `feat/multimodal-framework`, after the completed Task 0/1 commits. Scope: read existing files into validated **current-MoCA regression** subject records and retrieve saved outer train/test assignments. No prediction of future cognitive decline is claimed.

New files only:

- `medley_pd/multimodal/data_adapter.py`
- `medley_pd/tests/test_multimodal_data_adapter.py`
- `docs/multimodal_task2_data_adapter.md`

Existing EEG preprocessing, feature extraction, fixed splits, baseline predictions, evaluation and Task 1 configuration files are unchanged. No dependency installation or GPU-stack change was needed; the temporary Task 1 test environment was reused. No model/data download or training was performed. The final review authorizes one Task 2 commit and a development-branch SSH push attempt; the actual Git outcome is reported separately after verification.

## Existing interfaces reused

| Existing API | Adapter use |
|---|---|
| `medley.features.feature_dir(cfg)` | Existing `feature_root/dataset.name/encoder.name` cache directory |
| `medley.features.load_segments(cfg, sid)` | Return complete `feat` and Boolean `flag` arrays from `<sid>.npz` |
| `medley.participants.load_participants(cfg)` | Existing source-column mappings, missing tokens, sex encoding, patient regex and subgroup extraction |
| `medley.participants.clinical_sets(cfg, cross_site=False)` | Resolve C0/C1/C2 composition from the existing dataset/task settings |
| `medley.participants.task_cohort(cfg, d=None)` | Reuse patient-only and nonmissing-target cohort definition, passing the already loaded table |
| `medley.config.preprocess_spec(cfg)` / `spec_hash(spec)` | Compare an existing cache `spec.json` against resolved encoder/preprocessing settings, without creating a spec |
| Task 1 `validate_config(settings)` | Revalidate the supplied resolved settings, including selected encoder/dimension, task and clinical-set names |
| Task 1 `validate_split_selection(settings, medley)` | Read-only existing manifest validation; an unavailable file becomes an adapter error, never a generated replacement |

Actual schemas in `medley/evaluate.py` and `metrics.py` confirm subject-level original-unit regression. The adapter does **not** call `evaluate.prepare()` or `load_features()`, which average windows, nor does it call baseline/task model factories. Current MoCA regression uses the original scalar `moca` column after `task_cohort`, with additional finite/range validation; no transformed or classification label is introduced.

Existing `medley.splits.get_splits(cfg, d, regenerate=False)` can create a CSV when missing and can drop unavailable participants. There is no existing read-only function returning a train/test ID partition. The adapter therefore reuses Task 1's read-only validation and reads the **same saved CSV**, interpreting the same seed/fold convention. It never calls `get_splits()` or `make_splits()` and implements no competing/random split generation system.

## Public API and data flow

`MedleyDataAdapter(config: ResolvedConfig)` copies the Task 1 configuration, checks integration and source-column safety, loads the existing participant table once and records eligibility outcomes. EEG arrays are loaded lazily by subject. Configuration extensions were unnecessary: the existing resolved `medley` paths, dataset mappings, task, encoder and preprocessing definitions contain all required adapter inputs.

The adapter follows Medley's path conventions: `data_root/dataset.dir/participants` (TSV by default), `feature_root/dataset.name/encoder.name/<subject>.npz`, and `split_root/<dataset>__moca.csv`. Relative roots are interpreted against the construction-time working directory, then bound to absolute paths in a private copy so a later cwd change cannot redirect reads. The caller's configuration is unchanged. No RunPod-specific paths are embedded in the implementation.

| Interface | Return/behavior |
|---|---|
| `adapter.subject_ids` | Tuple of eligible patient IDs, in participant-table order. Does not assert all EEG caches exist |
| `adapter.clinical_variables` | Tuple of selected, approved Medley standard clinical names |
| `adapter.exclusions` | Tuple of `SubjectExclusion(subject_id, reason)` with `not_patient`, `missing_moca`, or `invalid_moca`. Held in memory, never automatically saved/logged |
| `adapter.metadata_notices` | Generic notice tuple when legacy participant normalization printed a warning; source values are suppressed from terminal output |
| `adapter.cache_spec_status` | `validated` when existing spec matches, otherwise `absent`; malformed/mismatched existing specs raise errors |
| `adapter.get_subject(subject_id)` | `SubjectRecord` or a `DataAdapterError`; no silent cache failures or dropped windows |
| `adapter.get_split(seed, fold)` | `SubjectSplit(seed, fold, train_subjects, test_subjects)` from the fixed manifest, with tuple ID lists |
| `validate_subject_partition(train, test)` | Checks disjointness, duplicates and nonempty sides; no randomness or file access |

`DataAdapterError` is a `ValueError` with a stable `.code`. Adapter messages omit clinical values/subject identifiers. Do not publish records, eligibility IDs, split IDs or traceback/context objects; these remain private research artifacts.

### SubjectRecord

| Field | Type and contract |
|---|---|
| `subject_id` | Canonical string, shared across mapped participants and cache filename; blank, missing-string sentinels and path-traversal identifiers rejected |
| `eeg_features` | Real floating NumPy `[T_i,D_e]`, T_i > 0, finite; original float dtype retained (normally float32, float64 accepted) |
| `artifact_flags` | Boolean NumPy `[T_i]`; same number/order of windows as EEG features |
| `clinical` | Read-only mapping from approved standard variable names to finite Python floats or `None` |
| `moca` | Finite Python float in [0,30], one current score per subject |
| `subgroup` | Existing Medley subgroup converted to string, or `None` if its mapped value is missing. The legacy literal `"NA"` subgroup is preserved |

Dataclasses are frozen; subject/array/clinical fields are excluded from repr to limit accidental logging. EEG arrays are independent copies and remain writable for future consumers; frozen does not make array contents immutable. The clinical mapping cannot be extended with a target field.

## EEG cache behavior

The segment loader preserves **all** windows in stored order, including every flagged window and all-flagged subjects. It does not average, reorder, truncate using legacy `evaluation.max_segments`, remove artifacts, pad, build batch masks or recompute features. Source check: `medley/eeg.py:preprocess` sets `flag=True` when any absolute EEG sample in the window exceeds `preprocess.flag_uv` (default 100 microvolts), before encoder amplitude scaling. True denotes a potential artifact, not a valid-window mask; neither True nor False proves clinical usability. The adapter preserves original ordering, without independently claiming timestamps establish chronology.

CBraMod's configured 19 channels x 200 embedding dimensions give D_e=3800. S-CEReBrO channel pooling uses 19 x 180=3420. Both the Task 1 dimension declaration and the resolved Medley channel/embedding/pooling configuration are checked; actual NPZ widths must agree. S-CEReBrO global mean pooling (180 dimensions) is outside the initial Task 1 schema. Cached vectors already pool patches inside each five-second window; the adapter cannot recover the raw within-window tokens.

Checks cover missing/unreadable/malformed NPZ, absent feat/flag entries, incorrect rank, zero windows, wrong width, nonfloating/nonfinite features and flag length. The existing loader converts flag values to bool, so the adapter additionally reads only the stored flag entry and verifies Boolean or numeric 0/1 values; NaN/strings/other numbers cannot silently turn into valid flags. This second flag read does not duplicate EEG preprocessing or the feature loader.

An existing `spec.json` must match the existing Medley spec hash. Absence is explicitly reported, not repaired. Legacy spec hashing does not include every preprocessing threshold, checkpoint contents or source-data provenance (see Task 0); a match is not a cryptographic/data-provenance guarantee. Real cache dimensions and historical provenance still need private-data preflight.

## Clinical sets, missing values and leakage

For PD-EEG the existing YAML maps these standard names to same-named raw columns:

| Set | Selected clinical inputs |
|---|---|
| C0 | `age`, `sex` |
| C1 | C0 + `updrs_part_iii`, `disease_duration`, `ledd` |
| C2 | C1 + `pigd_score`, `td_score` |

Set membership is retrieved from Medley, not duplicated in JSON or an alternative configuration loader. The seven listed standard variables are the explicit adapter **safety allowlist**. Selected names must be distinct, mapped and approved. Task 1 permits Iowa C0 with existing AGE/GENDER mappings; its C1_iowa name remains outside the initial C0/C1/C2 schema. Tests here use synthetic PD-EEG tables, not real Iowa data.

Missing **values** remain `None` for later serialization; there is no imputation, scaling or English text generation. Missing required source **columns** fail, even if optional values in those columns would have been allowed. Existing `load_participants` requires all configured mapped raw columns (including excluded label-related columns and configured subgroup source), not just selected predictors; the adapter preserves this contract. Extra unrelated raw columns are never selected as inputs.

Sex follows Medley's existing M=1/F=0 aliases or configured `sex_map`; numeric source codes are inherited, not reinterpreted. Its normalization warning is replaced with a generic metadata notice. Nonmissing values remaining nonnumeric or infinite after legacy loading raise errors. Existing numeric coercion may already convert malformed strings to missing; the adapter does not recover their original strings or invent replacements.

MoCA is only `record.moca`. It is absent from `record.clinical`; the same is true of CTT, UPDRS Total, IDs and group/subgroup labels. Leakage guards check:

1. Existing Medley task exclusions/target checks.
2. The adapter's seven-name predictor allowlist.
3. Configured raw source aliases against excluded/target source columns, plus normalized MoCA/CTT/assessment names anywhere in the source name and UPDRS+Total in either order. This includes transformed/reordered aliases such as score_MoCA, z_CTT and total_UPDRS, not just prefixes.
4. Duplicate source mappings among selected predictors.

These rules prevent broad numeric-column selection and known aliases. Arbitrary semantic aliases or proxies cannot be proven safe merely from values/correlation; private data dictionaries and acquisition timing remain required. No blanket exclusion based on correlation is used. PIGD/TD definitions, disease-duration/LEDD units, education availability and source sex coding require real-data review; no clinical meaning or unit is invented here.

### Label eligibility

Patient regex/subgroup/missing-target selection is inherited from Medley. Duplicate subject rows, including controls, fail rather than being aggregated across visits. Current MoCA must be numeric, finite and within its 0–30 score domain; fractional values are accepted. This label-domain check is independent of Task 1 inference `prediction.clip_range` and never clips a label.

Controls and missing/invalid labels are excluded from `subject_ids` with an explicit `SubjectExclusion`. Direct requests raise a corresponding error. No labels are imputed; no participants are silently removed. An unparseable raw MoCA may become missing during existing Medley numeric coercion and is then recorded as `missing_moca`. A fixed split containing an excluded subject raises an error, preserving cohort comparisons instead of silently shrinking the manifest.

### Subject and visit identity boundary

The join/cache/split key is the existing mapped `subject` (PD-EEG source `participant_id`). Raw `subject_id` is used by Medley only to extract a subgroup prefix, not exposed as a validated cross-visit patient linkage. There is no inspected, general patient-to-session registry in these APIs. The adapter must not guess that a session suffix is the same person, strip suffixes, or merge records. A synthetic mismatch test confirms that changing a participant key to a session-style ID does not silently join an old subject's cache/split. Duplicate identical keys fail, and windows under a single key cannot cross outer partitions.

**Different keys belonging to the same real person cannot be detected from these APIs alone.** Before real training, verify one-to-one patient identity across participant_id, any internal subject_id, visits/sessions, cache filenames and manifest rows, using an authorized private linkage table. If two keys refer to one patient and cross train/test, stop for protocol review; do not repair or regenerate official splits automatically. No real identity mismatch has been observed because actual tables/caches are absent. This is an unverified data-contract assumption, not a claim of implemented multi-visit grouping.

## Fixed splits

`get_split(seed,fold)` requires the requested seed/fold to be in the Task 1 selection. The original `subject,seed,fold` CSV is validated read-only: unique headers/assignments, matching row/header widths, finite integer seed/fold parsing, fold bounds, consistent patient coverage across seeds and selected assignments present. Duplicate headers are rejected before DictReader can silently replace one fold field with another; extra/missing row values also fail. For the selected seed, rows with fold == requested fold are test; all remaining original folds are train. The original one-based folds (1–5), zero-based seeds (0–9), file order and patient identities are retained. Development selection of test folds does not remove the other folds from training.

Every referenced ID for the requested seed must occur in loaded participants and be label-eligible. Train/test must be disjoint, individually unique and nonempty. Extra eligible participants outside the fixed manifest are never inserted. Missing caches do not silently alter partition IDs; loading such a record fails explicitly. Missing/malformed/duplicate/overlapping/unknown-subject manifests fail without creating directories/files or regenerating splits. A missing Task 1 manifest may be deferred at configuration time, but cannot be deferred past adapter split retrieval.

This is only the existing **outer train/test** split; no inner validation, early stopping assignment, shuffled windows or CV runner is implemented.

## Usage with Task 1

From `medley_pd/`, with authorized existing participant/cache paths configured through Medley's private `configs/local.yaml` or a resolved Task 1 snapshot:

```python
from multimodal.config import load_multimodal_config
from multimodal.data_adapter import MedleyDataAdapter

config = load_multimodal_config(
    "configs/multimodal/fusion_b.json",
    overrides=["clinical_set=C1", "evaluation.seeds=[0]"],
)
adapter = MedleyDataAdapter(config)
partition = adapter.get_split(seed=0, fold=1)
record = adapter.get_subject(partition.test_subjects[0])
# record.eeg_features: [T_i,3800]; record.clinical: selected structured values
# record.moca: current score label. Do not log or publish patient records/IDs.
```

For S-CEReBrO, supply `eeg.encoder=scerebro` and `eeg.feature_dim=3420` through the same Task 1 loader, with corresponding existing cache files. No new adapter-specific configuration fields were needed. This usage was exercised against synthetic disk fixtures; the example cannot run on absent real data without authorized private paths.

## Tests and actual validation

Executed from repository root using the existing temporary environment:

```bash
PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_config.py medley_pd/tests/test_multimodal_data_adapter.py
```

Final review result: **170 passed, 0 failed, 0 skipped (4.22s)**, including all 100 prior lightweight tests plus 70 adapter cases. The initial implementation had 161 passing tests. Review first added six failing regression cases to reproduce three defect classes (three alias cases, two malformed CSV cases, one NumPy partition case), then fixed their causes without disabling tests. Three additional passing coverage cases exercise all 50 seed/fold combinations, exact session-style key mismatch rejection and Task 1 resolved snapshot integration. `compileall` for new Python files and the repository scaffold check also passed.

Review commands: the reproduction run used `PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_data_adapter.py -k review` (6 failed, 61 deselected before correction); the full command above passed after correction. Fixes remain inside the three Task 2 files: non-prefix cognitive alias checks, duplicate-header/row-width checks, and length-based empty-partition checks compatible with NumPy ID arrays. No Task 1 schema extension or existing Medley edit was needed.

Fixtures write only synthetic tables, feature arrays and split manifests to pytest temporary directories. They exercise actual Medley loading, mapping and cohort APIs, with a spy test confirming reuse and prohibiting the averaging/extraction paths. Checks include multiple windows, order/flags, variable lengths, both encoder dimensions, all-flagged sequences, malformed shapes/flags/caches, NaN/Inf, C0/C1/C2, optional missing values versus missing columns, source alias leakage, valid/invalid labels, duplicates/unknown/unsafe IDs, immutable clinical mappings, read-only exact split retrieval, overlap/missing/malformed/unknown/ineligible split records, spec compatibility, copied config/array behavior and cwd-bound path resolution. No actual model is loaded or fit.

At inspection: **0 workspace NPZ caches and 0 participants.tsv files**; default `medley_pd/data`, `medley_pd/features` and private `configs/local.yaml` were absent. Existing fixed CSVs are available, but real subject-record loading/integration was **not executed**. Passing synthetic file tests is not evidence that private patient tables, clinical schema semantics or real encoder caches are compatible.

## Before Task 3

Confirm persistent storage and authorized private paths; verify actual dataset columns, subject/visit identities, source coding/units and cache provenance; obtain matching frozen cache sequences and confirm manifest/cohort consistency without regenerating splits. Real-data access failures should be addressed through preflight, not automatic downloading or preprocessing.

No PyTorch Dataset/Collator, padding/batch masks, English serializer, tokenizer, LLM, LoRA/QLoRA, pooling/projector/fusion/regression head, training loop or experiment runner was implemented. Task 3 is not started. Final-review authorization covers one Task 2 commit and push to `feat/multimodal-framework` only, with no merge or pull request. A passphrase-protected SSH key may require manual terminal authentication; no passphrase will be requested, logged or stored.
