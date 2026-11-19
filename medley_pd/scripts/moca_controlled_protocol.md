
## Controlled GPU comparison

This result uses the controlled baseline protocol below in place of the
historical baseline training conditions above.

| Model | Training | Model selection |
|---|---|---|
| EEG-only | Fixed CBraMod segment means, GPU Ridge | Existing EEG alpha grid scored by MAE on the shared inner validation patients |
| Clinical-only C0/C1 | GPU Ridge with common C0/C1 variables | Existing clinical alpha grid scored on the same validation patients |
| Stack C0/C1 | GPU Ridge head on 5-fold EEG OOF scores generated exclusively within shared inner training patients plus C0/C1 | Choose EEG alpha first, then Stack alpha, by the shared validation MAE |
| M1–M3 | Existing NF4 QLoRA and 30-epoch Huber training | Lowest MAE on the shared inner validation patients |

All models use the same 72 training patients, 18–19 validation patients and
22–23 outer test patients per Fold. Fits and preprocessing are restricted
to training patients. The GPU baselines use train-only median imputation
and standardization, analytical float64 Ridge with an unpenalized intercept,
and no validation refit. Frozen feature aggregation and the neural
serializer remain architecture-specific. C1 is the matched clinical input
comparison. C0 additionally tests the effect of using fewer clinical variables.
Ridge has no epochs. EEG-only history and selection metadata are stored
separately from the historical RidgeCV artifacts.
