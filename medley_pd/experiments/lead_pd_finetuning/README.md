# LEAD P-Base PD/HC fine-tuning

This experiment adapts LEAD to PD/HC EEG for later frozen feature extraction in PD-EEG MoCA regression and clinical fusion. The classification experiment uses EEG alone. Downstream evaluation is a separate experiment.

## Data and preprocessing

| Cohort | PD | HC | Original rate | Selected resting recordings |
| --- | ---: | ---: | ---: | --- |
| [Turku](https://osf.io/pehj9/) | 20 | 20 | 500 Hz | Raw eyes open and eyes closed where present |
| [New Mexico ds003490 1.1.0](https://openneuro.org/datasets/ds003490/versions/1.1.0) | 25 | 25 | 500 Hz | PD OFF sessions, unique HC sessions, marker-bounded EO and EC |
| [UCSD ds002778 1.0.5](https://openneuro.org/datasets/ds002778/versions/1.0.5) | 15 | 16 | 512 Hz | PD OFF and HC eyes-open sessions |
| Total before signal QC | 60 | 61 | | |

The preprocessing notebook uses the Medley alias mapping and interpolates missing or flat channels on the standard_1005 montage where supported. The order is Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2. Apply average reference, a 0.5–45 Hz bandpass, the 50 Hz Turku or 60 Hz New Mexico/UCSD notch, and resample to 200 Hz. Cache non-overlapping 400-sample, two-second windows in microvolts. A peak magnitude above 100 µV flags a window for audit and keeps it eligible. Colab applies the single per-window/channel Z-score. Splits are grouped by person and stratified by cohort and diagnosis where feasible. The seed-0 target split is 70/15/15.

Get raw recordings from the official cohort sources and the private pool TSV from the team through approved channels. Review dataset terms and institutional authorization before transferring processed EEG to Google Drive or Colab. The two OpenNeuro releases declare CC0. Verify cloud transfer rights for Turku separately.

## Reproduce

The two notebook copies incorporate the documented runtime corrections. They were statically validated during integration. Consult the archived run for the historical execution environment and artifacts.

1. On Windows, create a dedicated Python 3.12 kernel. In PowerShell, from this experiment directory, set your own paths and install the local requirements.

   ```powershell
   $env:LEAD_DATA_ROOT = "<directory containing cohort folders and the private pool TSV>"
   $env:LEAD_PROJECT_ROOT = "<local scratch and output directory>"
   New-Item -ItemType Directory -Force "$env:LEAD_PROJECT_ROOT/configs","$env:LEAD_PROJECT_ROOT/repos"
   Copy-Item configs/dataset_configs/*.yaml "$env:LEAD_PROJECT_ROOT/configs/"
   py -3.12 -m venv .venv-preprocess
   .\.venv-preprocess\Scripts\python.exe -m pip install -r requirements/local.txt
   git clone https://github.com/jeongwon8u8/Medley_skkai "$env:LEAD_PROJECT_ROOT/repos/Medley_skkai"
   git -C "$env:LEAD_PROJECT_ROOT/repos/Medley_skkai" fetch origin refs/pull/3/head
   git -C "$env:LEAD_PROJECT_ROOT/repos/Medley_skkai" checkout --detach d5bf678556ab9aeb0acf44ba2e7f0c5ad1c3b6bb
   code .
   ```

   Select `.venv-preprocess\Scripts\python.exe` and run [local preprocessing](notebooks/LEAD_PD_Preprocessing_Local.ipynb). Its ZIP and private audit are written under `LEAD_PROJECT_ROOT`. Place the private pool at `LEAD_DATA_ROOT/pool_base_3cohorts_private.tsv` or set the notebook's input override. Keep the original raw trees in place.
2. After authorization, upload the ZIP, select a Colab GPU, and open the [Colab notebook](notebooks/LEAD_PD_Finetuning_Colab.ipynb). Set `PACKAGE`, `CLOUD_TRANSFER_AUTHORIZED`, and `OVERLAP_ACKNOWLEDGMENT` after reviewing cohort identity risks. Use `MOUNT_DRIVE` and a private `OUTPUT_ROOT` for persistence. Run setup through the smoke check, then enable the training call.
3. To resume, select the same ZIP, run directory, environment and settings, and set `RESUME=True`. The training checkpoint uses the `history` field and a zero-based `epoch`. Resume rebuilds the CSV, and the evaluation helper restores a missing CSV from the same fields. For Drive, writes stage on the runtime's local disk and then copy to Drive. The copy can be interrupted and provides no atomicity guarantee. Keep a separate complete backup of the last checkpoint.
4. Run the held-out evaluation and encoder export only after training has completed. Keep private run outputs in approved storage. Use the notebook download cell if needed.

## Model and configuration

[Archived training configuration](configs/experiment_config.json) records seed 0, full FP32 fine-tuning, batch 32, accumulation 1, encoder/head learning rates 1e-5/1e-3, AdamW weight decay 0.01, cross-entropy, clip norm 1.0, maximum 30 epochs, and patience 5. Selection uses validation subject-level AUROC. Evaluation averages PD probabilities per subject and applies a 0.5 threshold.

The Medley base is `d5bf678556ab9aeb0acf44ba2e7f0c5ad1c3b6bb`. Official LEAD is pinned to `dddee9c94f13b83d76f41ac9d452a74f84a76c5b`. The completed run recorded Colab torch 2.14.1+cu130. The Colab notebook retains the accelerator-enabled torch installed by the runtime. The notebook retrieves the seed-41 P-Base checkpoint from [YiheWang/LEAD](https://huggingface.co/YiheWang/LEAD). The encoder has 12 layers, 8 heads, width 128, feed-forward width 256, and patch length and stride 50. It applies GELU to the last token per channel and flattens before the temporary classification head.

## Results and limitations

The completed seed-0 run showed a widening train/validation loss gap and substantially lower held-out than selected validation AUROC. This pattern is consistent with overfitting and provides limited evidence of generalization. Cohort sizes and acquisition and medication differences limit interpretation. The archive contains aggregate metrics and figures. Public sharing permission for those artifacts and exact run results has not been established, so their distribution is deferred.

Person-level overlap across fine-tuning, LEAD pretraining, and downstream PD-EEG remains unresolved. Head-only and partial baselines remain unevaluated. PD/HC classification does not establish a downstream MoCA benefit. Catastrophic forgetting is an untested hypothesis. Using this observed test set to repeatedly select hyperparameters would invalidate a final held-out comparison.

## Reuse with Medley

Use the exported `lead_pd_finetuned.yaml` as an encoder configuration. Set `repo` to the local official LEAD checkout and `weights` to the authorized local `lead_pd_finetuned.pth`. Keep `name: lead_pd_finetuned` and `type: lead` so feature caches use a distinct namespace and the existing frozen adapter. The expected output is 19 × 128, or 2432 features per window and per subject after averaging. The original frozen adapter and the shared MoCA/fusion code are unchanged.
