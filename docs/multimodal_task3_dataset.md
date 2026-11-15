# Task 3 — Subject Dataset and variable-length EEG Collator

Implemented 2026-11-15 (Asia/Seoul), on `feat/multimodal-framework` over the finalized Task 2 commit `8b19c6c67a3da66a4d2d61586ebc69cc392e0c0a`. Final review authorizes one Task 3 commit and a development-branch SSH push attempt; its actual Git outcome is reported separately. This task adds patient-level CPU tensors and right-padded batches for **current MoCA regression**, not model execution or future cognitive decline prediction.

New files:

- `medley_pd/multimodal/dataset.py`
- `medley_pd/tests/test_multimodal_dataset.py`
- `docs/multimodal_task3_dataset.md`

No Task 1 configuration extension, Task 2 modification, baseline change, split write, dependency installation or PyTorch/CUDA change was needed. No earlier commit is rewritten, and no merge or pull request is part of this review.

## Relationship to Task 2

The Dataset calls the public `MedleyDataAdapter.get_subject(subject_id)` interface and consumes its actual `SubjectRecord` dataclass fields: subject_id, eeg_features, artifact_flags, clinical, moca and subgroup. It checks ID membership against `adapter.subject_ids` and clinical keys against `adapter.clinical_variables`. It reuses the exported `APPROVED_PREDICTORS` safety allowlist for defensive validation; it does not recreate raw column mappings, cache paths, eligibility rules, clinical composition or split logic.

Task 2 remains responsible for interpreting the Task 1 resolved configuration, participant tables, cached features, encoder widths, source aliases, MoCA eligibility and saved split metadata. `SubjectSplit.train_subjects` and `test_subjects` supply explicit Dataset membership. There is no patient discovery from EEG windows and no replacement split manager.

## Dataset API

```python
from multimodal.dataset import MedleySubjectDataset

data = MedleySubjectDataset(adapter=adapter, subject_ids=split.train_subjects)
sample = data[0]
```

`MedleySubjectDataset` subclasses `torch.utils.data.Dataset`. Constructor input is a nonempty Python sequence of eligible canonical subject IDs (normally Task 2's tuple or an explicit list). A bare string, empty sequence, duplicate IDs, non-string/blank/whitespace-altered IDs and IDs absent from the adapter's eligible list fail immediately. Exact membership inherits Task 2 canonicalization; the Dataset does not strip IDs or infer patient identity from visit suffixes. The Dataset stores an immutable tuple copy exposed through `data.subject_ids`, preserving the supplied order even if the caller changes its original list.

`len(data)` counts subjects, never EEG segments. `data[index]` calls the adapter lazily and returns exactly one patient. Standard Python negative indexing is supported; Boolean indices, slices and noninteger indices are rejected. Out-of-range indexing raises `IndexError`. Dataset-level input errors use `SubjectDataError`, a `ValueError` subclass. Existing adapter exceptions propagate unchanged instead of being replaced with generic errors.

Repeated indexing produces equivalent values while source data remain unchanged. Each access returns independent tensors and a fresh clinical dictionary; mutating one sample cannot modify the adapter record or later samples. No cache files are loaded at Dataset construction. Missing caches and other record failures surface at access rather than silently dropping the patient.

### SubjectSample schema

`SubjectSample` is a TypedDict (ordinary dictionary at runtime):

| Field | Shape / type | Meaning |
|---|---|---|
| subject_id | str | Exact requested patient key |
| eeg_features | `[T_i,D_e]`, torch.float32, CPU | All cached EEG segments, unchanged order |
| artifact_flags | `[T_i]`, torch.bool, CPU | Original per-segment potential-artifact status |
| clinical | dict[str, float or None] | Same selected Task 2 clinical keys/values, separate from label |
| moca | `[]`, torch.float32, CPU | One original-unit current MoCA score |
| subgroup | str or None | Task 2 metadata, including literal `"NA"` when provided |

Task 2 normally yields float32 or float64 NumPy EEG. The Dataset copies it to contiguous float32 storage, handling read-only and negative-stride arrays. Normal float32 rounding is expected when the source is float64; finite values outside float32's representable range are explicitly rejected rather than overflowing into a valid-looking tensor. Integer/complex/nonfinite EEG, empty arrays, wrong shapes, misaligned/non-Boolean flags, missing/malformed SubjectRecord objects, identity mismatches and invalid scalar labels fail clearly. No averaging, truncation, artifact filtering, normalization or augmentation is performed.

NumPy masked EEG/flag arrays are explicitly rejected. They are not the ordinary arrays returned by Task 2's NPZ loader: converting them to a plain ndarray would discard missingness and expose hidden values (even hidden NaN) as observed data. This defensive boundary does not fill masks, drop windows or change Task 2's data contract.

The actual adapter validates D_e against its configured encoder: CBraMod 3800, S-CEReBrO 3420. Task 3 does not hardcode these dimensions, so small synthetic records can test generic padding. A batch must still have one common feature width.

## Collator API and batch schema

```python
from multimodal.dataset import MedleySubjectCollator

batch = MedleySubjectCollator()([data[0], data[1]])
```

`MedleySubjectCollator` is a stateless callable. It accepts a nonempty sequence of SubjectSample mappings and validates all required fields, tensor dtype/rank/device, finite EEG and targets, metadata types, flag alignment and clinical allowlisting. It rejects duplicate subject IDs in a batch, incompatible EEG widths and different selected clinical key sets. It does not silently cast malformed input samples. The Dataset performs the intentional NumPy-to-tensor conversion; the Collator expects its float32/bool CPU tensor contract.

| Batch field | Shape / type |
|---|---|
| subject_ids | list[str], length B, input order |
| eeg_features | `[B,T_max,D_e]`, torch.float32, CPU |
| eeg_mask | `[B,T_max]`, torch.bool, CPU |
| artifact_flags | `[B,T_max]`, torch.bool, CPU |
| clinical | list[dict[str, float or None]], length B, input order |
| moca | `[B]`, torch.float32, CPU |
| subgroups | list[str or None], length B, input order |

`T_max` is the longest sequence in this batch, not a global configured truncation length. `torch.nn.utils.rnn.pad_sequence` right-pads EEG with zeros and flags with False, without sorting the patients or windows. Target scalars are stacked in patient order. Clinical dictionaries are copied independently, not flattened/merged; subgroup and subject-ID lists remain aligned. Input samples and underlying arrays are never mutated. Memory grows with B × T_max × D_e; no automatic hyperparameter changes or silent truncation is introduced.

### Three distinct mask concepts

1. **eeg_mask:** True means a real segment exists; False means padding. It is based only on sequence length. A real all-zero embedding remains True.
2. **artifact_flags:** True means a potential artifact under Task 2/Medley's existing convention (default source threshold: any absolute EEG sample >100 microvolts). A flagged real segment remains present and has eeg_mask=True. No flags are inverted. False used to pad artifact_flags is **meaningless wherever eeg_mask=False**; it is not an observed clean segment. Always consult eeg_mask before interpreting flags.
3. **LLM attention mask:** Not implemented. EEG padding masks are not a tokenized-text or soft-token LLM mask; later tasks must explicitly construct those.

Example lengths 3 and 5 produce:

```text
eeg_mask = [[True, True, True, False, False],
            [True, True, True, True,  True]]
```

## Clinical data and label separation

Task 2's selected clinical dictionary is copied with its existing numeric-or-None representation. C0/C1/C2 selection, raw clinical aliases, units, sex encoding and missing-value interpretation remain Task 2 responsibilities. A defensive check rejects keys outside its seven-name allowlist, including MoCA, CTT, UPDRS Total or unexpected metadata keys. Sample clinical keys must match `adapter.clinical_variables`, and samples in one batch must share a selected key set.

No sentence generation, numeric embedding, scaling, imputation or categorical recoding occurs here. Existing numeric sex/phenotype values are not reinterpreted; missing None remains None. Nonmissing clinical values must meet the inherited finite numeric contract; text such as `"M"` is rejected rather than arbitrarily encoded. MoCA is validated as a scalar in [0,30] and remains in the separate moca field. Future model-input builders should explicitly select predictor fields and keep moca for loss/evaluation, rather than forwarding the full sample dictionary as predictors.

## Fixed split and DataLoader usage

From `medley_pd/`, after authorized private paths have been configured through the existing Medley configuration mechanism:

```python
from torch.utils.data import DataLoader
from multimodal.config import load_multimodal_config
from multimodal.data_adapter import MedleyDataAdapter
from multimodal.dataset import MedleySubjectDataset, MedleySubjectCollator

config = load_multimodal_config("configs/multimodal/fusion_b.json")
adapter = MedleyDataAdapter(config)
split = adapter.get_split(seed=0, fold=1)
train_data = MedleySubjectDataset(adapter, split.train_subjects)
test_data = MedleySubjectDataset(adapter, split.test_subjects)
assert not set(train_data.subject_ids) & set(test_data.subject_ids)

train_loader = DataLoader(
    train_data,
    batch_size=config.settings["training"]["batch_size"],
    shuffle=False,
    num_workers=0,
    collate_fn=MedleySubjectCollator(),
)
batch = next(iter(train_loader))
```

The Dataset/Collator do not read or write split files or create inner validation partitions. They consume exactly the provided ID tuple. Standard DataLoader shuffling may be requested later to change iteration order, but cannot change Dataset membership. A separate DataLoader helper was unnecessary. The example is exercised with synthetic disk-backed participants/NPZ/split fixtures, not actual private patient files. Only CPU, num_workers=0 behavior is claimed tested; multi-worker scheduling, worker spawning and GPU transfer are not validated here. Batch tensors are deliberately produced on CPU so a future trainer can move them explicitly.

## Verification and synthetic smoke test

Executed from repository root, with GPUs hidden from the test process:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_config.py medley_pd/tests/test_multimodal_data_adapter.py medley_pd/tests/test_multimodal_dataset.py
```

Final review result: **233 passed, 0 failed, 0 skipped (5.92s)**. This includes all 170 prior lightweight/Task 1/Task 2 cases and 63 Task 3 cases. The initial implementation had 229 passing tests. Final review added three regression cases reproducing silent unmasking of EEG/flags and one passing extended-precision NumPy target conversion case; only the confirmed masked-array defect required a source change.

The reproduction command was `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_dataset.py -k review`, producing 3 failed, 1 passed, 59 deselected before correction. The full command above passed after correction. The DataLoader smoke test also explicitly checks artifact values, subject IDs and subgroup alignment.

`compileall`, the repository scaffold check and whitespace checks passed during implementation/review. Pre-commit integrity verification checks all 124 existing tracked files, including Task 0–2 implementations/configuration and fixed splits. Only the three Task 3 files listed above are eligible for the new commit.

Coverage includes patient ordering/membership, lazy loading, scalar/feature/flag dtypes, deterministic repeat access, independent output storage, clinical None/subgroup preservation, malformed/missing records and sample fields, unknown/duplicate/invalid IDs, bad shapes/dtypes/labels/values, float32 overflow, negative strides/read-only arrays, equal/unequal/single-patient batches, incompatible widths, right-padding, separate artifact/existence semantics, input nonmutation, direct prohibited clinical fields, CPU DataLoader execution and saved SubjectSplit integration.

The CPU synthetic smoke path is `SyntheticAdapter → MedleySubjectDataset → DataLoader → MedleySubjectCollator → SubjectBatch`. Three patients have lengths 3,5,2 and width 4:

- Output EEG `[3,5,4]`, float32; B=3 patients, not 10 windows.
- eeg_mask `[3,5]` has 10 True entries with right padding only.
- Original alternating artifact flags are retained at real positions; padded entries are False and ignored by the mask.
- Labels `[20,21,22]` and clinical ages `[60,61,62]` remain aligned; sex values remain None and subgroup None is retained for the third subject.

Additional integration tests use the **actual Task 2 adapter and actual Medley file loaders** over temporary synthetic TSV/NPZ/CSV files, separately for D_e=3800 and 3420. They create train/test Datasets from its real SubjectSplit API, verify disjoint patient sets and ordered batches, and confirm the synthetic split bytes are unchanged. This demonstrates interface integration, not private-data compatibility. No full legacy training/permutation smoke workflow was run.

## Limitations and later-task interfaces

Actual patient/cache integration remains unverified: workspace participant tables and caches were unavailable. Source contracts cannot establish real clinical units/coding, cache provenance or whether different subject/session identifiers refer to the same person. Task 2's one-key-per-subject assumption and private linkage-table preflight remain required before real training. Confirm persistent storage, private paths, complete cache availability and immutable manifest/cohort alignment; do not regenerate official splits or infer patient identity by stripping session suffixes.

Task 4's future LLM factory will consume the existing Task 1 LLM settings; Task 3 does not load a model or tokenizer. Task 5 can consume each clinical dictionary with None intact, separately from moca. Task 7 can consume eeg_features `[B,T_max,D_e]` plus eeg_mask and the separate artifact_flags to implement its own explicitly documented modeling policy. No assumption about attention pooling or artifact exclusion is implemented here.

No LLM loading/tokenization, LoRA/QLoRA, feature extraction, EEG pooling, text/numeric clinical embedding, fusion, regression head, optimizer/loss/training loop, early stopping, validation splitting or cross-validation execution is provided. Task 4 and later tasks are not started.
