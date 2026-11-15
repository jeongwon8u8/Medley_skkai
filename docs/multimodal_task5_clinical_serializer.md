# Task 5 — Deterministic Clinical Text Serializer

## Purpose and scope

Convert allowlisted tabular clinical predictors into stable English inputs for
future M1/M2/M3 experiments. This is explicit template serialization: no generative
LLM, tokenizer, embedding computation, MoCA regression model, training or fusion.
There is no evidence here that converting values into text improves prediction.
Task 6 has not been started.

Tasks 0–4 documentation, existing participant normalization, configured clinical
sets, resolved configuration, adapter, Dataset/Collator and backbone interfaces
were inspected. Task 4 commit `57e3e835ffaf9e09675476f3420b5d4f0815dfc7` is
present. Work stays on `feat/multimodal-framework`; no commit/push/merge is part
of Task 5. Existing changes were preserved.

## Files and API

- New `medley_pd/multimodal/clinical_serializer.py`.
- Modified `medley_pd/multimodal/config.py`: optional Task 5 settings and validation.
- New `medley_pd/tests/test_multimodal_clinical_serializer.py`.
- New documentation: this file.

From the repository root use `PYTHONPATH=medley_pd` for the following API:

```python
from multimodal.config import CONFIG_DIR, load_multimodal_config
from multimodal.clinical_serializer import ClinicalTextSerializer

config = load_multimodal_config(CONFIG_DIR / "clinical_only.json")
serializer = ClinicalTextSerializer(config)
text = serializer.serialize(record.clinical)       # Task 2 SubjectRecord
texts = serializer.serialize_batch(batch["clinical"])  # Task 3 Collator
```

`ClinicalSerializationError(ValueError)` reports invalid predictor structures
and values without including supplied measurements or labels in messages. Whole
SubjectRecords, participant rows and batch dictionaries are not accepted. The
serializer never accesses labels, EEG, IDs or other subject metadata.

Batch input is a sequence of predictor mappings. Order, length and duplicate
entries are preserved. Empty batches return `[]`; a contaminated entry raises
without returning a partial list. Neither records nor batch tensors are mutated.

## Feature sets and exact templates

The serializer reuses Task 2's `APPROVED_PREDICTORS` and checks the selected
`ResolvedConfig.clinical_variables` against the versioned sequence below. That
property resolves existing Medley task/dataset definitions; no participant table
is read and no new fields are retrieved.

| Order | Canonical key | Sets | Exact `medley_neutral_v1` template |
|---|---|---|---|
| 1 | `age` | C0/C1/C2 | `The recorded age value is {}.` |
| 2 | `sex` | C0/C1/C2 | `The patient's recorded sex is {}.` |
| 3 | `updrs_part_iii` | C1/C2 | `The recorded UPDRS Part III score is {}.` |
| 4 | `disease_duration` | C1/C2 | `The recorded disease duration value is {}.` |
| 5 | `ledd` | C1/C2 | `The recorded LEDD value is {}.` |
| 6 | `pigd_score` | C2 | `The recorded PIGD score is {}.` |
| 7 | `td_score` | C2 | `The recorded TD score is {}.` |

C1 is the existing default. Fixed punctuation and one ASCII space between
sentences are used. Input dictionary insertion order cannot change output.

All selected keys must exist, with `None` representing missing values. Passing
C0 records to C1 fails instead of silently changing the experiment. Extra
approved keys fail by default. Explicit
`clinical_text.extra_approved_policy=ignore` permits larger approved sets, such
as C2 records for C1, excluding the additional predictors. Even excluded values
must be valid. Unknown/prohibited keys always fail, regardless of this policy.
Selection, version and formatting policy are saved in resolved snapshots.

## Verified source facts and unresolved units

Reviewed `medley/participants.py`, dataset YAML mappings and
`configs/tasks/moca.yaml`, plus Tasks 0/2 audits. PD canonical predictors map to
same-named raw columns. Iowa uses AGE/GENDER and the shared C0 definitions.
Names alone do not verify age/disease-duration units, LEDD dose units or PIGD/TD
score definitions. Private data dictionaries were unavailable. Templates therefore
preserve neutral names and numbers without years/months/mg/day, unit conversion,
assumed ranges, motor phenotype classification or diagnostic interpretation.
UPDRS Part III is named by its canonical key; scale/version is not newly inferred.

Medley's `_encode_sex` maps textual M/F aliases to 1/0 but preserves numeric source
codes without verifying their meaning when `sex_map` is null. Task 2 retains
these codes, and Task 3 keeps the numeric clinical contract unchanged. Therefore
the serializer does not enable a default numeric sex mapping.

- Direct strings `female`, `male`, `F`, `M` normalize by strip/lowercase. This is
  direct-API convenience, not a change to Task 2/3 numeric loading.
- Missing sex (`None`) is skipped.
- Numeric sex requires an explicit source-verified `sex_code_mapping`. Keys are
  canonical integer strings, values exactly `female` or `male`. Finite integral
  numeric scalars match without rounding. Unsupported/fractional codes,
  booleans, arbitrary strings, numeric strings such as `"0"`, and nonfinite
  values fail.
- The configuration asserts investigator verification; the code cannot prove
  source coding. Do not apply a synthetic mapping to unverified numeric sources.

For synthetic codes known to come from Medley's textual M/F normalization:

```python
config = load_multimodal_config(CONFIG_DIR / "base.json", [
    'clinical_text.sex_code_mapping={"0":"female","1":"male"}'
])
```

Real runs with numeric sex require source-coding verification and a recorded
mapping first. This is an explicit real-data prerequisite, not a guessed label.

## Numeric and missing-value policy

Accept Python int/float and NumPy integer/floating scalars. Reject booleans,
numeric strings, arrays/containers, complex numbers and unsupported types.
NaN and infinity are invalid, not silently treated as missing. Only `None`
denotes missing optional values.

The versioned precision policy is:

1. Integers render exactly, without significant-digit rounding.
2. Floats start from their type's shortest decimal `str` representation, avoiding
   float32-to-float64 widening artifacts.
3. Decimal half-even rounding uses 15 significant digits by default, configurable
   from 1 through 34. Thus `6.500000000000001` becomes `6.5` without a fixed
   decimal-place truncation. Decimal precision, rounding, exponent limits and
   traps are isolated from caller settings.
4. Fixed decimal notation, trailing fractional-zero removal and negative-zero
   normalization render `68`, `68.0` and NumPy equivalents as `68`; `6.5` stays
   `6.5`, and `0.00000012` stays `0.00000012`.

This is a representation policy, not source measurement precision certification.
Measurements needing more than 15 meaningful digits require a reviewed override.
No normalization, imputation or unit conversion occurs. Changing precision is an
explicit experiment configuration change.

Skip-missing is fixed by this template version: omit sentences for `None`, without
dropping subjects. If all selected fields are `None`, return the intentional empty
string `""` and retain its batch slot. No missingness-indicator sentence is added.
Later tokenizer integration must explicitly decide how to consume empty text.

## Leakage protection

Exact canonical allowlisting rejects all aliases outside the seven predictors,
including MoCA/MOCA_score/Montreal cognitive assessment, target-derived names,
CTT/Color Trails Test, UPDRS Total, subject/visit IDs, group/split metadata and EEG.
They fail even alongside otherwise valid predictors or with extra-ignore enabled.
No whole-row stringification, arbitrary metadata concatenation, label-dependent
wording or patient-specific logging occurs. Tests verify rejection rather than
silent omission and verify that label changes leave clinical text unchanged.
Approved fields that are semantically mislabeled upstream still require source
schema/observation-time review; allowlisting cannot establish their true origin.

## Task 1 configuration compatibility

Optional defaults are injected before overrides and when validating/saving older
configs or snapshots:

```json
{
  "clinical_text": {
    "template_version": "medley_neutral_v1",
    "numeric_significant_digits": 15,
    "extra_approved_policy": "reject",
    "sex_code_mapping": null
  }
}
```

All four existing JSON files load unchanged. Existing inheritance, CLI dotted
overrides and snapshot roundtrips work; unknown options, invalid versions,
precision/policy values and malformed mappings raise `ConfigError`. The serializer
copies configuration options so later caller mutation cannot alter its behavior.
No Task 2 loading or Task 3 Dataset/Collator contract changed. No Task 4 backbone,
QLoRA, EEG preprocessing/extraction, fixed split, baseline or evaluation file was
modified. No dependency installation or PyTorch/CUDA changes were required.

## Synthetic example and integration

Input:

```python
{"age": 68, "sex": "female", "updrs_part_iii": 24,
 "disease_duration": 6.5, "ledd": None}
```

Exact output:

```text
The recorded age value is 68. The patient's recorded sex is female. The recorded UPDRS Part III score is 24. The recorded disease duration value is 6.5.
```

The integration test constructs three actual Task 2 `SubjectRecord` objects via
a synthetic in-memory adapter implementing its public interface, followed by the
real Task 3 Dataset and Collator, then Task 5 serialization. It is not a real-data
`MedleyDataAdapter` file-loading run. Cases include a complete C1 record, partial
missingness and different numeric values, variable EEG lengths and explicitly
mapped synthetic numeric sex. Assertions cover identity/order, one text per
subject, no input mutation, deterministic repeat output, absence of synthetic
IDs/metadata/EEG/labels, and unchanged text after replacing every MoCA label.

## Verification commands and results

From the repository root, using the existing temporary test environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_clinical_serializer.py
```

Task 5 result: **88 passed, 0 failed, 0 skipped**.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_config.py medley_pd/tests/test_multimodal_data_adapter.py medley_pd/tests/test_multimodal_dataset.py medley_pd/tests/test_multimodal_llm_backbone.py medley_pd/tests/test_multimodal_llm_preflight.py medley_pd/tests/test_multimodal_clinical_serializer.py
```

Full regression result: **366 passed, 0 failed, 1 skipped**. The skipped existing
Task 4 test is the opt-in actual GPU/QLoRA check, unavailable in this CPU/offline
run. Existing PEFT Conv1D and Transformers deprecation warnings are retained.
A subprocess test verifies that serializer import/config load/use do not import
torch, transformers, tokenizers or PEFT. No model download is needed. The final
checks also include `git diff --check` and verification of the changed file list.

## Real-data status and prerequisites before Task 6

Only synthetic data were tested; actual patient-data compatibility is unverified.
Verify clinical units/coding/precision, observation timing, subject/visit linkage,
authorized persistent private data/cache paths and storage persistence before real
use. Record a verified numeric sex mapping where needed. Define empty-text
handling in the later tokenizer integration. Task 4's real-model/GPU/QLoRA access
and hardware requirements remain separate. No clinical efficacy or predictive
performance result is claimed. No patient data, caches, checkpoints or credentials
were added to these deliverables. All Task 5 changes remain uncommitted.
