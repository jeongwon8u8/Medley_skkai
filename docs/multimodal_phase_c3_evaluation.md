# Phase C3 — Matched multimodal evaluation

## Scope and prerequisites

C3 analyzes **stored current-MoCA predictions in original 0–30 units**, one
subject per model/seed/fold. It never constructs a model, extracts EEG, runs
forward passes, downloads weights or starts training. No real patient comparison,
full cross-validation or Phase D ablation was executed for this implementation.
Synthetic artifact-schema verification is not trained-model performance evidence.

The implementation follows Phase B (`4368569`), C1 (`f022af6`) and C2
(`34baf34`) on `feat/multimodal-framework`. Before implementation the C1/C2
protocol, manifest, runner, training-control and run-state tests passed: **123**.
The actual fixed protocol uses seeds **0–9**, folds **1–5**, not zero-based folds.
The original PD MoCA split SHA-256 remains
`5ef85c2056831d71cc04b603b06001739e5d69671a3a57a646597aebb2af6602`.
Existing prior artifacts and commits are preserved.

## Modules and public interfaces

- `experiment_evaluation.py`: `validate_prediction_frame`, `load_completed_run`,
  `load_series`, `load_legacy_predictions`, `PredictionSeries`, `AnalysisError`.
- `prediction_matching.py`: `comparison_conditions`, `match_predictions`,
  `MatchedPair`; distinct candidate/reference labels can represent future ablations.
- `evaluation_metrics.py`: `core_metrics`, `summarize_series`, `paired_metrics`.
- `paired_bootstrap.py`: `cluster_row_indices`, `paired_bootstrap`, `BootstrapResult`.
- `evaluation_reports.py`: `analysis_plan`, `analyze_predictions`, `export_analysis`.
- `evaluation_cli.py`: `main`, the prediction-only CLI.

Optional `settings.analysis` defaults are added through the existing config
resolver, preserving old snapshots. Analysis settings are excluded from inference
checkpoint identity. The existing digest is now a pure config helper; C2's default
completion validation still checks finite tensors. C3 explicitly selects
`validate_completion(..., verify_tensors=False)`: completion-state semantics,
selection replay, runtime/config identity, metrics, manifests and every artifact's
byte hash remain checked, without importing torch or loading checkpoint tensors.
This verifies recorded completion and bytes; it does not independently rerun the
model or certify clinical extraction provenance.

## Source schema and integrity

The exact original eight columns are
`seed,fold,subject,subgroup,model,y,yhat,target`. `y` must equal `target` exactly.
Subject/model strings are retained, including leading zeros; target/prediction
values must be finite numeric values. Targets must be original-unit MoCA scores.
Seed/fold must be integers with valid bounds. Duplicate model/seed/fold/subject
keys and multiple folds for one subject/model/seed are rejected. Repeated subject
targets and subgroup metadata must remain equal. Empty subgroup CSV fields are
represented as absent metadata, rather than fabricated categories.

Native ingestion accepts a C2 execution directory or its `completed.json`.
It checks C1/C2 manifests, exact test membership/order, run/model/seed/fold,
completion artifacts, selection/history and config identities. All source files
are hashed, and rechecked before/after analysis/export. DataFrame access returns
copies. `load_series` rejects duplicate runs and materially mixed variants under
one label; canonical key sorting makes input argument order immaterial.

M0 normally uses C2's read-only original baseline import. The optional legacy
loader selects an explicit original `model` from an eight-column CSV/CSV.GZ;
its provenance remains `legacy_rows_only_unverified_provenance`. Neither matching
rows nor a valid imported M0 completion proves its historical training protocol.
Strict comparisons to M0 are therefore unavailable when those conditions cannot
be verified. Explicit subset mode can describe matched predictions with the
unverified conditions visible. No baseline training is attempted.

## Matching and fair-comparison levels

The intended join is `(seed,fold,subject,target)`: target and subgroup consistency
are checked before joining by the first three keys, then verified after joining.
A changed target never becomes an excluded row through a silent inner join.
Changed fold assignments are rejected in strict mode. Explicit analysis scope is
the configured `evaluation.seeds × evaluation.folds`; missing runs are reported.

Strict mode requires exact matched keys and recorded comparable provenance:
cohort/source population, target/unit, split bytes/assignments, EEG cache content,
clinical serialized content, LLM/base/tokenizer identity, inner training partitions,
training/adaptation settings, selection settings and code/environment identities.
A declared immutable LLM revision is recorded as supplied, not independently
verified by downloading it. Checkpoint metadata plus supplied identity is the
available evidence. Synthetic base identity is accepted only under the explicit
synthetic analysis flag.

The report distinguishes matched rows, recorded protocol comparability and the
stronger `fully_controlled_configuration`. Different architectures/capacities
can share a comparable evaluation protocol without receiving the stronger flag.
The stronger flag also requires equal intrinsic fusion/projector/training config.
It is a statement about recorded config, not proof of clinical truth.

`matching=matched_subset` explicitly uses key intersection within scope. Private
analysis manifests retain original/scoped counts, matched rows/unique subjects,
missing runs, excluded keys and outside-scope rows. Public plan summaries omit
excluded subject keys. Every such result is labeled **matched subset**, and the
fully controlled flag is false. It is never whole-cohort performance.

`allowed_differences` accepts exact permitted setting paths for declared future
ablations, not wildcard provenance waivers. Shared setting differences otherwise
block strict comparability. Target/cohort identities cannot be waived. Each
material variant requires a distinct label, even if its underlying M1–M3 model
identifier is the same. Phase D experiments themselves are not run here.

## Metrics and estimands

MAE is `mean(abs(yhat-y))`; RMSE is `sqrt(mean((yhat-y)**2))`, using validated
float64 arrays and the original Medley definitions. Overflow fails explicitly.
Every summary includes prediction-row count, unique subjects, seeds, folds and
runs. These denominators distinguish repeated observations from independent
patients.

- `metrics_by_run`: within each seed/fold, prediction-row MAE/RMSE.
- `metrics_by_seed`: pooled prediction rows within each seed.
- `metrics_by_fold`: unweighted means of per-run metrics for that fold across seeds.
- `metrics_summary`: separate `pooled_prediction_rows` and
  `unweighted_mean_run_metrics` rows. The latter averages per-run RMSE, rather than
  taking a square root after averaging run metrics.

Pair comparisons use the common configured `pooled_rows` (default) or `mean_runs`
estimand on the **same matched rows for both models**. The convention is
`delta_MAE=MAE(candidate)-MAE(reference)`, likewise RMSE: negative means lower
stored-prediction error. The six default pairs are M1/M0, M2/M0, M3/M0, M2/M1,
M3/M1, M3/M2. Explicit pairs accept arbitrary labels. Missing/incompatible pairs
receive unavailable status, never fabricated scores. Partial standalone summaries
are labeled `partial_available_runs`.

No subject-mean-prediction estimand is implemented. Optional subgroup summaries
are reported unavailable because C1/C2 do not attest category-definition provenance;
row-level subgroup equality alone is insufficient.

## Subject-cluster paired bootstrap

The default is 2,000 draws, RNG seed 42, 95% confidence, minimum five unique
subjects (configurable with an absolute floor of two). Tests use small draws and
explicitly lower the policy threshold for tiny synthetic examples.

1. Sort the unique matched subject IDs and index their complete paired records.
2. Use a local `numpy.random.default_rng(seed)` to draw N subject indices with
   replacement, where N is the number of unique matched subjects.
3. Concatenate all records for every selected cluster across relevant seeds/folds.
   A subject drawn twice contributes its complete cluster **twice**.
4. Use identical row indices for candidate and reference; compute signed deltas
   for the selected estimand.
5. Take linear percentile quantiles at `(1-confidence)/2` and its complement.

`cluster_row_indices` exposes exact multiplicity for testing. No independent
prediction-row resampling or independent-fold significance test exists. Fixed
single-run scope is distinguished from pooled repeated-CV scope. Pooled repeated
CV is supported for `pooled_rows`. Repeated-run `mean_runs` has point estimates
but returns interval unavailable (`bootstrap_estimand`): resampling subjects can
empty individual runs, and C3 does not silently change the run denominator.
A single fixed run supports both equivalent choices. Too few unique subjects
returns unavailable (`insufficient_subjects`), rather than a narrow interval.

Intervals are descriptive and conditional on stored fitted predictions. Clustering
preserves repeated patient dependence but incompletely represents overlapping
training sets, fitted-model reuse, hyperparameter selection, dataset shift and
small/nonrepresentative cohorts. These are not universal significance tests.
No p-values, probability-of-superiority estimates or clinical-superiority claims
are generated.

## CLI and configuration

From the repository root, use the existing config/snapshot resolver:

```bash
PYTHONPATH=src:medley_pd python -m multimodal.evaluation_cli \
  --config <resolved-config.json> \
  --input-manifests variant_A=<completed-run-directory> variant_B=<completed.json> \
  --reference-model variant_A \
  --set 'evaluation.seeds=[0]' --set 'evaluation.folds=[1]' \
  --bootstrap-replicates 2000 --bootstrap-seed 42 --confidence 0.95 \
  --output-dir outputs/c3-analysis --dry-run
```

Repeat a label for multiple distinct seed/fold executions. Omit `--dry-run` and
add `--execute` to create reports; default operation only validates and prints the
plan. `--execute` requires an explicit output directory. Optional switches:
`--models`, `--strict-matching` (default), `--matched-subset`, `--aggregation
pooled_rows|mean_runs`, `--synthetic-test-data`, `--legacy-input
LABEL:original_model=path`. Synthetic reports explicitly say **synthetic test data**.
Use authored synthetic fixture inputs only with that flag; it does not turn
synthetic data into patient evidence.

`settings.analysis` also supports `pairs`, `allowed_differences`, `min_subjects`,
`subgroups`, `output_format=csv_json_markdown|json` and bootstrap options.
Independent row bootstrap and unsupported aggregation values are rejected.
An unavailable comparison is a reported analysis outcome, not an execution error
that is silently replaced by an easier cohort.

## Reports and reproducibility

`export_analysis` writes a new `analysis-<sha256>` namespace beneath the explicit
output root. Existing namespaces are never overwritten. Reports cannot be written
inside native experiment, split, data or feature-cache roots. In the repository,
only Git-ignored `outputs/` or `checkpoints/` destinations are accepted. Namespace
permissions are 0700; artifact permissions are 0600. No raw patient prediction
table is copied. Exact exclusion keys in private manifests must stay untracked.

Default artifacts are `metrics_by_run.csv`, `metrics_by_seed.csv`,
`metrics_by_fold.csv`, `metrics_summary.csv`, `pairwise_comparison.csv`,
`bootstrap_intervals.csv`, `analysis_manifest.json`, `analysis_summary.md` and
`analysis_complete.json`. JSON mode replaces CSV/Markdown with
`analysis_result.json`. The completion marker records hashes of all reports.

The analysis identity includes source paths/hashes/manifests/config identities,
explicit run scope, analysis options, six evaluator module byte hashes and
Python/numpy/pandas/scipy versions. No timestamp changes the ID. The manifest
records target/unit, models, cohort matching diagnostics, effective config,
missing conditions, signed direction, bootstrap method and limitations. Bootstrap
metadata includes draw fingerprint, subject/row counts, seed/draws/confidence,
scope and quantile method. This supports deterministic numerical reproduction
under the same files, config and runtime; relocating sources changes the recorded
identity. Small floating-point differences across platforms remain possible.

## Verification and independent review

The staged implementation gates passed ingestion/matching/metrics (20 tests),
then numerical bootstrap (35 combined tests), then reports/CLI and the review
regression (**41 C3 tests**). Fixtures author synthetic C1/C2 artifact schemas,
including one static NumPy safetensors array; they do not fit or load a model.
No actual clinical performance numbers appear in this document.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  /tmp/medley-task1-tests/bin/python -m pytest -q \
  medley_pd/tests/test_multimodal_experiment_evaluation.py \
  medley_pd/tests/test_multimodal_paired_bootstrap.py \
  medley_pd/tests/test_multimodal_evaluation_reports.py

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  /tmp/medley-task1-tests/bin/python -m pytest -q tests \
  medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_*.py
```

Analytic tests check unequal run sizes, pooled versus mean-run denominators,
signed deltas, exact RNG-reproduced draws/percentiles, full repeated-subject
clusters, duplicate-draw multiplicity, swap antisymmetry, zero-error differences,
missing/inconsistent inputs, too few clusters, all six synthetic model pairs,
CSV/JSON hashes, source preservation, overwrite/output protection and argument-order
reproduction. A fresh-process CLI test asserts torch, transformers and
safetensors.torch are absent from imported modules. The regression suite includes
existing tiny synthetic training tests as prerequisite checks; C3 analysis itself
never trains. Pandas emits a deprecation warning for the retained round-trip CSV
float parser; this is currently supported and preserves exact target values.

The full regression command above passed **903 tests, 1 skipped** (57.81s).
The skip is the opt-in actual-GPU NF4 test. The independent final C3 rerun
passed **41 tests** (16.85s). Existing PEFT/Transformers compatibility warnings
and pandas CSV-parser deprecation warnings remain; there were no failures.

An independent read-only reviewer inspected joins, provenance, delta/aggregation,
cluster sampling, multiplicity, report protection and CLI imports, and separately
reran core tests. It confirmed a defect where equal architecture names could
receive `fully_controlled_configuration=true` despite changed intrinsic projector
settings. The flag now checks fusion/fusion_a/fusion_b_training equality and
exports that condition; a regression test reproduces the defect. Review also
confirmed C2's finite-tensor default is unchanged and exact subset exclusions are
retained privately. Initial report tests contained a metadata-key typo and an
invalid config snapshot; both test fixtures were corrected before the final gate.

## Engineering principles and readiness

Modularity is provided by separate ingestion, matching, metrics, bootstrap and
report modules. Configurability follows existing snapshots/overrides. Backward
compatibility preserves the eight-column schema, splits, artifacts and C2 default
validation. Reproducibility uses hashes, isolated RNG and explicit estimands.
Incremental gates verify each layer. Ablation support uses arbitrary labels,
variant config identities and declared differences. No redundant experiment is
run: C3 consumes completed predictions only.

C3 software correctness is verified synthetically. Real patient prediction
analysis and evidence for a performance difference remain unverified. Actual
experiments require audited clinical source/mappings and leakage exclusions,
complete compatible C1/C2 runs, immutable accessible LLM/tokenizer identity,
appropriate hardware and dependencies, baseline training provenance or explicitly
descriptive subset interpretation, and a prespecified aggregation/uncertainty
protocol with adequate subject counts. Phase D requires separately authorized,
verified ablation inputs. C3 does not execute those experiments.
