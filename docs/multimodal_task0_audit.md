# Task 0 — RunPod environment and Medley repository audit

Audit date: 2026-11-15 (Asia/Seoul). Scope: inspection, repository setup, and future integration proposal only. Architecture A/B and Task 1 are **not implemented or started**.

## 1. Environment and repository setup

| Check | Observed result |
|---|---|
| User / host / cwd | `root` / `59e256625f8b` / initial `/workspace` |
| OS | Ubuntu 24.04.5 LTS |
| Python / pip / Git | 3.12.3 / 26.2.1 / 2.43.0 |
| Python executable / environment | `/usr/local/bin/python`; prefix and base prefix `/usr`; no active venv/Conda variables |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB; initial utilization 0%, no running GPU processes |
| Driver | 595.91.07; `nvidia-smi` advertises CUDA 13.2 |
| Actual PyTorch | `torch.__version__ == 2.8.0+cu128`; `torch.version.cuda == 12.8` |
| CUDA execution | `torch.cuda.is_available() == True`; GPU identity confirmed; small GPU matrix multiplication and CPU equality assertion passed; BF16 support reports True |
| Toolkit | `nvcc` release 12.8, V12.8.93 |
| Disk | `/workspace` resolves to root `overlay`, approximately 50 GiB total, 48 GiB available at inspection |
| Persistent storage | **Not established**: no separate persistent `/workspace` mount appears in `findmnt -T /workspace`. Do not assume this checkout survives Pod/container deletion. RunPod console volume configuration needs owner confirmation. |
| Remote VS Code | SSH connection and VS Code IPC environment present; running `.vscode-server` processes and remote CLI found. Strong evidence of Remote SSH execution on the GPU container; editor-selected folder and RunPod account/Pod identity cannot be independently established from these checks. `VSCODE_CWD` was `/root`, while command cwd was `/workspace`. |
| GitHub | HTTPS clone and subsequent `git fetch origin` succeeded; read access confirmed, write access neither needed nor tested |

The GPU runtime works. **No PyTorch, CUDA, driver, or GPU package was reinstalled.** The driver capability display is not the PyTorch runtime version. Shell sandbox launch failed with a user-namespace/bwrap error; successful checks ran through the approved escalated command mechanism. No host security settings were changed.

`/workspace` initially contained only `.cache`, with no existing checkout or uncommitted repository work. Cloned `https://github.com/jeongwon8u8/Medley_skkai` at `medley-pd-code`, fetched origin, inspected clean status and branches, then created `feat/multimodal-framework`. Audit base commit: `967ad1d83fb5c55277143bb48edbd9b309644543` (`Add files via upload`). Remote branches include main, medley-pd-code, and feat/evaluation-swaptest. No commit or push performed.

Read `AGENTS.md`, `docs/ownership.md`, `docs/contracts.md`, `docs/prompts/lead.md`, and `docs/research_plan.md`. This user-authorized new audit document is the sole intended file change. Existing source, shared contracts, private config, splits, and baselines remain untouched.

## 2. Actual structure and dependencies

Paths below are repository-relative. Working repository: `/workspace/Medley_skkai`; original experiment application: `medley_pd/`.

- `medley_pd/configs/config.py`, `config_loader.py`: common Python defaults and JSON experiment handling.
- `medley_pd/configs/{datasets,encoders,tasks}/*.yaml`: dataset mappings, encoder inputs, task definitions. `local.example.yaml` documents private paths; actual `local.yaml` is absent.
- `medley_pd/medley/`: EEG preprocessing, caches, participants, splits, sklearn fusion, tasks, evaluation, metrics, reporting, permutation tests and tools.
- `medley_pd/{train,test,extract,run}.py`, `medley_pd/scripts/*.sh`: entry points. Shell wrappers change cwd to `medley_pd`.
- `medley_pd/splits/`: four **tracked fixed CSVs**, plus README. Plain `rg --files` initially hid CSVs because ignore rules match them; `git ls-files` and direct filesystem reads confirmed them. They are present, not missing.
- `medley_pd/tests/test_configs.py`: four pytest cases after parametrization. `tests/smoke_test.py` under this directory is an executable synthetic workflow, not a small unit test.
- Root `src/codl/`, `configs/`, `experiments/`, `tests/`, `docs/` form a separate collaboration scaffold. Root `pyproject.toml` packages `src/codl`, not the `medley_pd/medley` application. Many domain directories are interface scaffolds; do not assume they implement the proposed multimodal architecture.

Dependency files are root `pyproject.toml` and `medley_pd/requirements.txt`; no application lockfile was found. Root requires Python >=3.10, numpy >=1.24,<3, sklearn >=1.3,<2, with pytest dev extra. Legacy requirements include einops, huggingface_hub, mne, numpy, openneuro-py, pandas, pyyaml, safetensors, sklearn >=1.2, scipy, timm, torch; mostly unpinned. `matplotlib` is imported for plots but is not explicitly listed. CBraMod fallback imports `braindecode`, also not listed.

Installed: torch 2.8.0+cu128, numpy 2.5.2, PyYAML 6.0.3. Missing distributions: pandas, scipy, scikit-learn, mne, einops, huggingface-hub, safetensors, timm, openneuro-py, braindecode, pytest, transformers, peft, accelerate, bitsandbytes. `pip check` reports no broken installed requirements; this **does not** establish application completeness. Environment search found no pyvenv.cfg under `/workspace` or shallow `/opt` and common root venv locations; `/opt/conda`, `/opt/venv`, `/venv` absent. This was a bounded search, not an inventory of every filesystem location.

Recommended setup, not installed here: keep Python 3.12.3 and working torch/cu128. Use a separate project venv with access to the template packages if desired, constrain torch to the observed build, and inspect resolver changes before installation. First add only pandas/scipy/sklearn/pytest for cache adapters and tests; add EEG extraction dependencies only when extraction is authorized. Record resolved versions and `pip check`, then test imports and GPU operations. Do not run an unrestricted upgrade of requirements or the GPU stack. Python 3.11 is used in root CI, but that alone is not a reason to replace this working interpreter. Existing experiment versions are unknown, so exact historical numerical reproduction is not promised.

## 3. Existing data flow

1. `participants.read_raw_table(cfg)` reads participants TSV (CSV supported) under `data_root/dataset.dir`; configured source columns are required. `load_participants` maps to standard columns, trims subject/group strings, marks patients using PD/PARK regex and derives subgroup from the configured source column.
2. `task_cohort` selects patients by default and removes missing target rows. `evaluate.prepare` adds original `target` and task label `y`, loads features, and aligns participant rows to successful feature subjects. Missing caches cause subject exclusion, not synthetic imputation of EEG.
3. `eeg.FileIndex` searches configured file extensions/patterns, picks the first sorted match, preferring non-derivative files. It is not a multi-visit aggregation API.
4. `eeg.preprocess` keeps the recording reference; normalizes channel names (T3/T4/T5/T6 aliases); selects/interpolates missing or flat channels; bandpasses 0.3–75 Hz (upper limit capped by original Nyquist), applies dataset notch (PD 50 Hz, Iowa 60 Hz), resamples to 200 Hz, forms non-overlapping 5-second windows, discards incomplete trailing samples, converts volts to microvolts and divides by encoder amp_div. Interpolation needs at least 20 usable source channels if interpolation occurs; this check precedes final selection of 19 target channels.
5. Frozen encoder `.embed` produces one vector per 5-second window. `features.run_extract` writes compressed NPZ (`feat`, `flag`), `spec.json`, and `extract_log.csv`. Encoders load lazily only if extraction is needed. Existing caches are skipped; optional legacy cache copies are possible.
6. Current baselines use `load_features` to average windows into one patient vector; clinical matrices contain NaNs until train-fold preprocessing. CV uses fixed patient assignments. Stack trains its EEG head on inner out-of-fold scores; Concat fits PCA and clinical transforms on the outer training fold.

No raw EEG, real feature loading, encoder initialization, model weight download, or full training was executed in this audit.

## 4. Reusable APIs and exact return contracts

All locations in this table are under `medley_pd/`.

| Source and signature | Return / implications |
|---|---|
| `medley/config.py: load_config(dataset=None, encoder=None, task=None, overrides=())` | Merged cfg dictionary; reads configs without datasets or weights |
| `medley/participants.py: load_participants(cfg)` | DataFrame with mapped clinical/label columns, subject, group, is_patient, subgroup |
| `task_cohort(cfg, d=None)` | Patient/target-filtered DataFrame |
| `clinical_sets(cfg, cross_site=False)` | Name -> list of permitted standard variables; raises for target/excluded variables |
| `medley/eeg.py: preprocess(path, cfg)` | `(segs, flag, info)`; float32 `[T,19,1000]`, bool `[T]`, metadata dict for default encoders |
| `medley/encoders/__init__.py: build_encoder(cfg)` | Frozen encoder object; `.embed(segs)` -> float32 `[T,D_e]` |
| `medley/features.py: feature_dir(cfg)` | `feature_root/dataset.name/encoder.name` |
| `load_segments(cfg, sid)` (line 132) | `(z["feat"], z["flag"].astype(bool))`, **no temporal averaging** |
| `load_features(cfg, subjects, drop_flagged=False)` (line 102) | `(F, noise, ok, miss)`; float64 `[N,D_e]`, noise `[N]`, successful IDs in row order, missing IDs |
| `medley/splits.py: get_splits(cfg, d, regenerate=False)` | DataFrame subject/seed/fold; **can write if missing** |
| `make_splits(subjects, y, subgroup, ev, binary=False)` | New assignment DataFrame; do not invoke for missing fixed splits |
| `medley/evaluate.py: prepare(cfg, cross_site=False)` | `(d,F,C,sets)`; C maps set names to `[N,p]` arrays |
| `cv_predict(cfg, d, F, C, y_fit, y_true, splits, inv=lambda v: v, noise=None, verbose=True)` | Long prediction DataFrame; runs existing baseline/fusion fits |
| `summarize(pred, cfg, out_dir, per_seed=True)` | `(summary,paired,sub)` DataFrames and output CSVs; writes to supplied directory |
| `compare_predictions(cfg, P, encoders, path)` | Paired comparison DataFrame and CSV; P maps encoder labels to predictions |
| `medley/metrics.py: regression_metrics(y, p)` | n, MAE, RMSE, R2, Spearman, Pearson |
| `bootstrap_ci(y, p, n=2000, seed=42)` / `bootstrap_diff(y, pa, pb, n=2000, seed=42)` | Patient bootstrap intervals; differences A minus B |
| `medley/report.py: per_case(pred, is_clf, extra=None)` | Patient-level wide table, seed means/SD and errors or probabilities |

### Window cache and tensor shapes

Default cache path from cwd `medley_pd`: `./features/<dataset>/<encoder>/<subject>.npz`. Private `local.yaml` may override feature_root; none exists here. Legacy convention: `<sid>_<condition>.npz`, commonly `_EO.npz`, only copied during extraction. No `.npz`, `.pth`, or `.safetensors` files were found under `/workspace`; default data/features/results directories are absent. No actual patient tensor dimension was measured.

Expected from code: CBraMod receives `[T,19,5,200]`, averages the five within-window patches and flattens channels -> float32 `[T,3800]`. S-CEReBrO channel pooling -> `[T,3420]` (19 x 180), mean pooling -> `[T,180]`. These cached vectors already pool within a 5-second window; using load_segments preserves the between-window sequence, not raw encoder patch tokens. Custom torch_module dimensions depend on output layout and pooling.

Artifact flags are Boolean `[T]`: any absolute sample exceeds 100 microvolts **before** encoder amplitude normalization. Flagging does not remove windows by default. `load_segments` returns all windows/flags, with no max_segments restriction, missing-file handling, spec validation, finite-value checks, or shape checks. A future adapter must validate these and apply synchronized truncation/filtering explicitly. `load_features` applies max_segments first, optionally removes flagged windows, returns their mean feature, and reports flag fraction over the truncated original windows. All-flagged subjects become missing only when drop_flagged=True. Future flags and padding masks must remain distinct.

`spec.json` is extraction provenance, but readers do not enforce it. Existing spec hashing omits flag_uv, min_good_channels, source data hashes, and checkpoint contents (only file basenames retained for selected paths); legacy copies are not individually verified. Matching directory names do not prove cache equivalence. Audit provenance before reuse without overwriting existing caches.

## 5. Clinical variables and leakage controls

PD mapping: subject <- participant_id; group <- group; standard names age, sex, moca, ctt, updrs_part_iii, updrs_total, disease_duration, ledd, pigd_score, td_score map to same-named raw columns. Subgroup additionally reads raw subject_id, extracting leading letters. Education is only a commented placeholder, not a configured input.

PD MoCA sets: C0=[age,sex]; C1=C0+[updrs_part_iii,disease_duration,ledd]; C2=C1+[pigd_score,td_score]. Iowa maps subject <- participant_id, group <- GROUP, age <- AGE, sex <- GENDER, moca <- MOCA, updrs_iowa <- UPDRS; C0=[age,sex], C1_iowa=C0+[updrs_iowa]. Cross-site MoCA uses C0 only. Iowa TYPE is deliberately unmapped; UPDRS version/part and clinical units remain unresolved by source alone.

NA strings are normalized; sex accepts configured mappings, M/F aliases, or numeric codes with a warning. Numeric conversion is heuristic (at least half the nonmissing entries parse); actual coding must be verified privately. Clinical imputation uses training-fold medians with keep_empty_features=True, then standardization. Future text serialization should use explicit missing markers or training-only statistics and a stable unit-aware template; do not serialize whole participant rows, IDs, group/subgroup labels, targets, or arbitrary free text.

MoCA task target is current moca, no log transform; classification uses moca<26. Forbidden MoCA inputs: **moca, ctt, updrs_total**, plus any label-derived, future, or otherwise leaking variables discovered in the private data dictionary. `clinical_sets` enforces task.exclude/target; JSON `resolve_sets` rejects unknown variables and target, but allow_excluded=True deliberately removes exclusion restrictions. A future independent loader must reject that bypass for formal experiments and use an explicit allowlist. Existing CTT comparison permits MoCA in C1_moca for its separate CTT task; that permission must never carry over to MoCA prediction.

Dataset comments and historical README counts/results are prior reports, not freshly reproduced clinical findings. Without participants tables, units, observation timing, label definition, duplicate IDs and visit linkage cannot be validated.

## 6. Fixed subject splits

Files are `medley_pd/splits/<dataset>__<task>.csv`; schema `subject,seed,fold`. Existing checks (stdlib CSV, no patient IDs printed):

| File | Rows | Unique subjects | Seeds | Folds / test fold sizes |
|---|---:|---:|---|---|
| pd_eeg__moca.csv | 1130 | 113 | 0–9 | 1–5 / 22–23 |
| pd_eeg__moca_impaired.csv | 1130 | 113 | 0–9 | 1–5 / 22–23 |
| iowa_rest__moca.csv | 1000 | 100 | 0–9 | 1–5 / 20 |
| iowa_rest__moca_impaired.csv | 1000 | 100 | 0–9 | 1–5 / 20 |

All passed: unique (subject,seed), identical cohort across seeds, all ten seeds and five folds each seed. CSV content unchanged. CTT split absent. These checks establish assignment consistency, not clinical identity equivalence across visits or label stratification against absent labels.

Read-only SHA256 provenance:

- iowa_rest__moca.csv: `7c406cec43032efef64153fd6f4fe05642d57dd5f8d6cd913dbb79641c53d8ba`
- iowa_rest__moca_impaired.csv: `d85872f4faaec26c9427559726f622f31c30fe1370d9a338ed1d85dd00617513`
- pd_eeg__moca.csv: `5ef85c2056831d71cc04b603b06001739e5d69671a3a57a646597aebb2af6602`
- pd_eeg__moca_impaired.csv: `588dd93de15cc008ab9e334a5508bd2239b2a67cd137ef59a37d3dad352bb36c`

`get_splits` loads existing files, rejects new subjects/missing seeds, and filters out unavailable subjects. It does not fully validate duplicates, fold range or complete per-seed coverage. Crucially, missing file OR regenerate=True calls make_splits and writes a CSV, even with the default regenerate=False. Reachable via train.py, run.py cv -> run_cv, and permtest -> run_permtest; --new-splits explicitly regenerates. Do not call these on missing splits. Future loader must fail before this API when the fixed file is absent and validate the complete manifest without writing.

Current outer CV tests one fold and trains on all others; no separate persistent validation assignment exists. Future early stopping/hyperparameter selection needs a reproducible inner patient-level validation split from the outer training patients, held separately from the immutable official files. All windows/visits of a patient must follow that patient. Fix a common available cohort before comparing model outputs.

## 7. Existing fusion, prediction and evaluation bridge

`medley/fusion.py` defines `Fusion(cfg, ctx=None)`, `fit(self,F,C,y,cache)`, `predict(self,F,C)` -> `[N]`. F is a NumPy patient mean embedding; C is numeric clinical input with NaNs. cfg selects regression/classification; shared cache holds EEG model and inner OOF scores. Context from cv_predict contains subjects_train, subjects_test and seed; fold index is not passed explicitly. `predict` receives no subject argument, so a segment-aware wrapper must deliberately bind test IDs through context. Cross-site context currently lacks seed.

Ridge EEG/clinical heads live in heads.py/tasks.py; Stack uses inner OOF EEG scores plus clinical Ridge/logistic regression. Concat uses training-fit EEG PCA and clinical impute/scale followed by Ridge/logistic regression. Do not change these implementations or force neural data loaders through their existing NumPy contract. Prefer a separate PyTorch runner and adapter, reusing participant/cache APIs and evaluation functions. An optional future explicitly imported registration wrapper can be considered later; it must preserve baseline behavior and distinguish clinical set context, model lifecycle and patient ordering.

Compatible CV export is one row per subject/seed/model with columns:

`seed, fold, subject, subgroup, model, y, yhat, target`

MoCA regression: y and target are observed original-unit scores; yhat is one original-unit score per patient, detached CPU scalar, not logits or window scores. Unique (subject,seed,model); same subject/fold/cohort across models; finite values; verify labels and order before evaluation. Existing model names include mean, noise, eeg, clinical:C0, fusion_stack:C0, fusion_concat:C0. Future proposal: fusion_softtoken:C0 / fusion_repr:C0, plus an EEG-only path as required by draft contracts. Names beginning fusion_ participate in automatic fusion comparisons. Baseline rows must be present for paired deltas. Config, code revision, split/cache hashes and seed must accompany outputs; patient predictions remain private.

`summarize` computes per-seed metrics, seed-averaged patient metrics, patient bootstrap CIs, paired deltas and subgroup metrics. MAE is primary, with RMSE/R2/Spearman/Pearson; MAE deltas A-B, lower better. `test.py` additionally needs config_used.yaml and optionally subjects.csv with noise_ratio for per-case output. `compare_predictions` requires matching cohorts, folds and model keys, and checks equality of non-EEG baseline predictions across encoders. Do not use it unchanged to compare runs with incompatible model sets. Root codl metrics have a separate classification contract; legacy Medley returns NaN for single-class AUROC while the root contract expects an error. This semantic difference is documented, not changed.

## 8. Configuration architecture and future component map

Current order: COMMON Python dictionary -> dataset/encoder/task YAML -> private local paths and encoder overrides -> experiment JSON -> CLI overrides. `medley.config.load_config` legacy --set parses YAML values and dotted paths, anchors default ./split_root to the application directory, and injects dataset reference if encoder reference is empty. Other relative roots remain cwd-dependent. JSON `load_json(path,kind="train",overrides=())` ignores underscore comment keys, warns about unknown file keys, rejects unknown CLI top-level keys, parses JSON values with string fallback. `build_cfg(exp)` applies JSON models/clinical sets/evaluation/fusion_params, validates registry names and records experiment metadata. It imports fusion and therefore currently needs sklearn. Config mapping does not constitute a strict typed schema.

Propose a new independent multimodal YAML/JSON schema and CLI, with namespaced model/trainer/serializer/adapter settings; call existing load_config on a copy for legacy dataset/task/encoder specs. Validate unknown keys, allowlisted inputs, shapes, fixed split existence, revision/dtype/quantization, IDs and paths before starting a run. Do not add Architecture A/B keys to current TRAIN_KEYS or change old override behavior. The exact future path ownership requires Lead review because src/codl is a separate scaffold.

All rows below are proposals only, no files implemented:

| Future component | Suggested integration and contract |
|---|---|
| Multimodal configuration loader | Independent namespace/entry point; read existing config, preserve old CLI |
| Medley data adapter | Reuse task_cohort, clinical_sets, feature_dir, load_segments; strict read-only split/provenance validation |
| Patient-level PyTorch Dataset | One patient item: EEG `[T,D_e]`, flags `[T]`, permitted clinical fields, label scalar, internal ID |
| Variable-length EEG collator | Padded `[B,Tmax,D_e]`, Boolean valid `[B,Tmax]`, separate artifact flags, labels `[B]`; explicit empty/all-flagged policy |
| Clinical text serializer | Deterministic allowlisted fields, units, missing markers, versioned template; no labels/identifiers |
| LLM backbone factory | Explicit model/tokenizer revision, local cache policy, hidden size, padding, dtype, LoRA/quantization config |
| EEG attention pooler | Mask-aware `[B,T,D_e]` -> `[B,K,D_p]` configurable learned EEG summary tokens |
| EEG-to-LLM projector | `[B,K,D_p]` -> `[B,K,H_llm]`; validate encoder dimension at load |
| Soft-token fusion (B, primary) | Combine EEG tokens with clinical token embeddings through inputs_embeds; consistent masks/positions and regression pooling |
| Representation-level fusion (A) | Separate EEG and clinical representation pooling, configurable fusion module |
| MoCA regression head | Fused representation -> `[B]`; target original units and any train-only normalization recorded |
| Common PyTorch trainer | Training-only preprocessing, masked batches, inner validation, checkpoints, deterministic seeds and GPU memory logs |
| CV runner | Read official 10-seed/5-fold manifest; never regenerate; derive inner validation only within training subjects |
| Prediction/evaluation bridge | Export the existing long schema, call summarize/report without modifying evaluation; matched baseline cohort |

For B, a frozen LLM still needs gradient flow through its operations to train EEG/projector inputs; do not wrap its forward pass in no_grad when optimizing those inputs. Verify inputs_embeds support, tokenizer padding, context budget K+clinical tokens and pooling location on the chosen model. A causal decoder must pool a position that can attend to the intended inputs. Preserve a separately evaluable EEG-only prediction path per draft research contracts; final architectural details require later implementation review.

## 9. Risks, compatibility and decisions

**Confirmed blockers/issues:** absent Medley analysis dependencies and pytest; no patient table/cache/encoder weights in workspace; persistence not established; default encoder paths unconfigured; broad dependency ranges/no application lock; matplotlib and fallback braindecode not explicitly declared; root pytest configuration only discovers tests/ and root packaging does not install medley. Fixed split generation has side effects; cache readers lack strict validation. Existing source is writable as root, so future runners must explicitly enforce read-only split use.

**Unverified risks:** actual clinical coding/units/timing and visit identity; historical dependency versions and numerical reproduction; bit-level cache provenance; pretrained model licenses/access; Transformers/PEFT/bitsandbytes compatibility; exact GPU peak memory and model-specific inputs_embeds/gradient support. No incompatibility in torch/cu128 GPU execution was found. No LLM compatibility claim can be tested with absent libraries/weights.

24 GiB VRAM is a budget, not a guarantee. Approximate raw 7B BF16 weights alone take 14 GB decimal before activations/gradients/optimizer state; full fine-tuning is unsuitable as a default. Later start with a small backbone or frozen/LoRA configuration, bounded token length/K, small batch, gradient accumulation/checkpointing and measured memory. QLoRA is a later option, not a package installation performed here. HF documentation describes bitsandbytes quantization and adapter-based training; model/hardware/version support must be validated in the project environment before choosing a locked stack. See [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes) and [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization). The [official PyTorch previous-version matrix](https://pytorch.org/get-started/previous-versions/) documents release-specific builds; keep the already functioning observed cu128 installation.

Next steps requiring owner/Lead decisions: confirm persistent volume and private data/cache locations; provide authorized access without copying data into Git; validate the clinical allowlist/units and subject identity manifest; select model/revision/license and tuning/memory budget; approve a reproducible dependency lock and independent module ownership. None requires changing existing data during Task 0.

Suggested future sequence, **not executed**: establish persistence/private paths -> isolate and lock minimal dependencies -> run existing configuration tests and small adapter/cache contract checks -> verify immutable manifest/cohort alignment -> implement independent config/data/serializer layer -> implement modular B and A -> trainer/inner validation/CV/export bridge -> compare against untouched baselines. Stop at this audit until the next task is explicitly requested.

## 10. Executed validation and limits

| Check | Actual outcome |
|---|---|
| whoami/hostname/pwd, nvidia-smi, Python/pip/Git | Passed via approved commands after sandbox-launch failure |
| PyTorch import/version/device + CUDA matrix multiplication | Passed; BF16 support True; no training |
| nvcc --version | Passed, CUDA toolkit 12.8.93 |
| HTTPS clone / git fetch | Passed; clean base, feature branch created |
| df -hT / findmnt | Passed as inspection; persistent mount **not confirmed** |
| python -m pip check | Passed: no broken installed requirements; application dependencies still absent |
| python scripts/check_repository.py | Passed: 7 required scaffold files |
| AST parsing of medley_pd/src/root tests Python files | Passed: 39 files; syntax validation only, no module imports |
| JSON/YAML config parsing | Passed for medley_pd config files |
| load_config for pd_eeg/iowa_rest x cbramod/scerebro | Passed: five folds, seeds 0–9, absolute split root, CZ/PZ reference propagation and existing MoCA split paths |
| Read-only validation of four split CSVs | Passed as detailed above; no IDs printed or copied to report |
| python -m pytest -q | **Failed to start**: /usr/local/bin/python: No module named pytest; no tests collected or run |
| medley_pd/tests/test_configs.py | Inspected, not executed: pytest/sklearn absent |
| medley_pd/tests/smoke_test.py | Inspected, not executed: dependencies absent; also performs many synthetic baseline fits/permutation workflows and temporarily writes/deletes local.yaml, outside this audit-only validation scope |
| Existing end-to-end training/extraction and numerical results | Not executed; missing data/cache/dependencies and prohibited full training/downloads |

No dependency installation, model/dataset download, existing implementation edit, split regeneration, commit, or push. New deliverable: docs/multimodal_task0_audit.md only. Final Git and split integrity checks are recorded after report creation in the completion message.
