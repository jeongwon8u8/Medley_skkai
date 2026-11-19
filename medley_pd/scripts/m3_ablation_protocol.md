# M3 diagnostic ablations

Seed 0, Fold 1–5, 113 patients. Each condition runs 30 epochs with validation
MAE checkpoint selection and one held-out test evaluation. C1 clinical text,
CBraMod cache, AdamW, Huber, and configured QLoRA budgets are fixed. Patient IDs
and scores are never printed by the launcher. Reports stay under the ignored
outputs directory.

| Arm | Validation | Pooling | Fusion | Purpose |
|---|---|---|---|---|
| A | SHA-256 rank | Attention K=4 | M3 soft tokens | Reuse verified 30-epoch anchor |
| B | MoCA band quotas, SHA-256 within bands | Attention K=4 | M3 soft tokens | B − A measures the validation policy change |
| C | SHA-256 rank | Attention K=1, fixed input projection | M3 soft tokens | Attention in pooling contrast |
| D | SHA-256 rank | Mean K=1, fixed input projection | M3 soft token | D − C measures pooling; D is the fusion reference |
| E | SHA-256 rank | Mean K=1, same fixed projection | M2 late fusion | E − D measures the fusion architecture change |

Stratification uses only outer-train labels and proportional largest-remainder
quotas for MoCA ≤24, 25–27 and ≥28. Train/validation counts remain 72 and
18–19. The test membership and ordering are unchanged. Completed stratified
manifests are independently checked against current source labels before
reuse and reporting. They do not store those labels.

C–E use a deterministic frozen linear input projection with a shared seed.
D and E therefore have identical EEG vectors for identical cached segments.
The downstream projection, text/EEG interaction, regression head, trainable
capacity and compute remain architecture-specific. Attention retains
learned query/attention layers. C is a separate K=1 control; A versus D is
not the pooling-only contrast.

Every arm reports MAE, RMSE, Spearman and signed prediction-minus-target
bias overall and for MoCA ≤24 and ≥28. Reports require all five outer Folds.
Test data are never used for fitting, calibration or checkpoint selection.

Use one worker by default; at most two with verified free VRAM. The launcher
budgets 7 GiB per worker and 2 GiB reserve. This is an admission heuristic,
not a measured peak. Reuse state from the same scientific configuration
and compatible code/runtime identity. No training occurs in the plan action.

## Two-hour D/E profile

`--profile de45` runs D and E on Fold 4 and Fold 5 only, in one GPU worker.
It requires the existing A results for those Folds and uses them as a
descriptive reference. Only E versus D is an isolated matched contrast.
B and C remain available in the full profile. The launcher stops admitting
work at a 120-minute deadline and terminates an in-flight worker at that
deadline. The last completed epoch and existing completed results are retained.

The report covers the common 44 patients. It labels the combined results
`matched_4_5` and exports a private paired prediction table with D, E and A
predictions, E−D prediction changes and E−D absolute error changes. The
Fold-wise report includes MAE, RMSE, Spearman and MoCA tail metrics.
