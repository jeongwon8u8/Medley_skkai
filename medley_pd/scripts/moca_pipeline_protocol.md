# PD-EEG MoCA comparison protocol

All models evaluate the same 113 PD patients on the existing patient-level
Seed 0 × Fold 1–5 assignment by default (Seed 0–9 is optional). Each patient contributes one OOF prediction per
seed. We report MAE, RMSE and Spearman per seed, plus the same metrics after
averaging OOF predictions per patient when multiple seeds are requested. Models are paired by patient,
seed, fold and original-unit MoCA target.

| Model | Existing recipe reused | Original training and selection |
|---|---|---|
| EEG-only | Frozen CBraMod, mean of cached segments, RidgeCV | 90–91 outer-training patients. Train-only median/scaling, RidgeCV selects alpha. Reuse validated Fold predictions. |
| Clinical-only C0/C1 | RidgeCV on age/sex (C0), plus UPDRS-III, duration and LEDD (C1) | 90–91 outer-training patients. Median/scaling fitted on the training partition. Reproduce using the existing legacy predictor. |
| Stack C0/C1 | EEG scalar prediction + C0/C1, RidgeCV head | 90–91 outer-training patients. Inner 5-fold OOF EEG inputs train the head. A refit EEG model generates held-out test inputs. |
| M1 | Qwen3-4B clinical-text regression | 72 training patients and 18–19 inner-validation patients. NF4 QLoRA, AdamW, Huber loss. Run 30 epochs and select lowest validation MAE. |
| M2 | CBraMod attention pooling + clinical text, late fusion | Same multimodal budget and selection as M1. |
| M3 | CBraMod attention pooling as soft tokens in the clinical LLM | Same multimodal budget and selection as M1. |

The historical multimodal run allowed a patience of five epochs. A Fold that
reached epoch 30 can be reused if its scientific configuration and saved
predictions match. A shorter Fold can supply an epoch-state snapshot for
continuation. The original outputs remain intact.

The CBraMod cache is frozen and its specification and input fingerprints are
checked. The legacy models average segments and impute clinical features
inside training partitions. The multimodal models retain their configured
attention pooling and serializer missing-value handling. The held-out test
set is used only for final evaluation. Sample counts and training recipes
differ, so this comparison measures paired test performance across methods.
The controlled default is documented in the appended GPU comparison
section. The historical path remains available through `--legacy-cpu`.
