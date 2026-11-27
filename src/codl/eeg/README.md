# 작업 범위

CBraMod, S-CEReBrO, CAPE 등 검토 중이며 실제 구현/사전학습 가중치는 미포함입니다.


## Completed: external three-cohort PD/HC pretraining → frozen MoCA/fusion (2026-11-27)

This is a separate experiment from the earlier directly MoCA-supervised pilot.
The user deferred BrainLat and explicitly selected the base Turku/UNM/UCSD pool:
121 people (PD60/HC61), 200 recordings, 65,771 valid cached windows. Seed0
cohort×label stratification gives train96/validation25; all recordings of each
person remain together. BrainLat, PD-EEG and Iowa are excluded from pretraining
and checkpoint selection. The paper-based modified architecture and common
19-channel, 200 Hz, 0.3–45 Hz, 1-second, amplitude flag-only preprocessing remain fixed.

Random-initialized PD/HC training stopped after23 epochs; external validation
BCE selected epoch3. Validation AUROC=0.711538, balanced accuracy=0.500000 at
threshold0.5 (all25 classified HC). Encoder/classifier checkpoints, split,
logs and numerical audits are saved; reloading exactly reproduces predictions
and features. This is selection-set performance, not an independent external test.

The frozen encoder's eval-mode patient EEG256 vectors were extracted once for
PD-EEG113; parameters and BatchNorm buffers were verified unchanged. All eight
new downstream settings completed the existing seed0 outer5-fold split, with
exactly one OOF prediction per person/model. MAE: EEG Ridge2.564717,
EEG MLP2.538488; FT Concat C0/C1=2.453615/2.376630; Clinical Ridge Stack
C0/C1=2.219973/2.220110; FT Stack C0/C1=2.240521/2.211330. Nested head/clinical
OOF excludes held-out patients from preprocessing/training/early stopping;
existing clinical inner checkpoints were partition-checked and replayed.
FT Concat averages five independently trained member predictions. Clinical
baselines were reused. No clear incremental MAE advantage over clinical-only
was found; the best overall MAE remains Clinical Ridge C1=2.194190.

Full Korean Notion-ready report, complete metrics, ΔMAE, limitations, and
commands: [RESULTS_EXTERNAL_3COHORT_KO.md](RESULTS_EXTERNAL_3COHORT_KO.md).
Private artifacts: `outputs/external_pd_hc/external_3cohort121_pd_hc_directed_cheb_seed0/`
and `outputs/eeg_gnn/external_3cohort121_frozen_pd_eeg_moca_seed0/`.
Entry points: `train_external_pd_hc.py --three-cohort` with
`external_3cohort121_seed0.json`, `run_external_frozen_moca.py`,
`audit_external_frozen_moca.py`, `report_external_3cohort.py`.
Full suite27 passed; all downstream checkpoint replays and OOF audits passed.
No shared preprocessing edits or new dependencies in this experiment.


Architecture A의 새 학습 결과: [README_ARCHITECTURE_A](../fusion/README_ARCHITECTURE_A.md).
