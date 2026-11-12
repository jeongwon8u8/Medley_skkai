# Medley PD — EEG + 임상정보 통합 실험 코드

PD 환자 안에서 인지기능(MoCA, CTT)을 추정할 때 **pretrained EEG encoder**와 **비인지 임상정보**의 상호보완 가치를 평가하는 공통 코드입니다.
**데이터셋·인코더·과제·fusion 모델만 바꿔 끼우고, 전처리·split·평가 설정은 모두 공통**으로 유지합니다.

```
configs/
  config.py                 ★ 공통 설정 (예전 base.yaml): 전처리·평가·비교 모델 기본값. 여기만 고치면 전체에 적용
  config_loader.py          실험 JSON → 설정 (변수별 켜기/끄기, 결합 방식 켜기/끄기, key=value 덮어쓰기)
  train_config_pd_eeg.json  ★ 실험 조건: 임상변수 조합(변수마다 true/false), 결합 방식, 구간 수, 실험 이름
  train_config_iowa_rest.json
  test_config_pd_eeg.json   ★ 평가 조건: 평가할 실험·인코더, 환자별 CSV·그림, 순열 검정
  test_config_iowa_rest.json
  datasets/pd_eeg.yaml      PD-EEG (ds007526 v1.0.2)
  datasets/iowa_rest.yaml   Iowa Rest (ds004584 v1.0.0) — 기준전극 Pz 등 확인 완료
  local.example.yaml        개인 경로 예시 (configs/local.yaml 은 각자 만들고 GitHub 에 올리지 않음)
  encoders/cbramod.yaml     frozen CBraMod (기존 프로토콜)
  encoders/scerebro.yaml    frozen S-CEReBrO (BioFoundation PR #18, commit 361c017)
  tasks/moca.yaml           실험 ①: MoCA, clinical set C0/C1/C2
  tasks/ctt.yaml            실험 ②: CTT, C1(⑤-A) / C1_moca(⑤-B)
  tasks/moca_impaired.yaml  인지 저하 분류: MoCA < 26 (type: classification)
medley/
  participants.py   참가자 표 → 표준 열 이름, 환자·하위 집단, 입력 금지 변수 검사
  eeg.py            파일 찾기·채널 선택·보간·필터·리샘플링·구간 나누기
  encoders/         인코더 레지스트리 (cbramod, scerebro, torch_module)
  features.py       피처 추출·캐시 (spec.json 으로 사양 고정)
  splits.py         고정 split (데이터셋 × 과제마다 하나, 인코더와 무관)
  fusion.py         fusion 레지스트리 (stack, concat + 팀 모델 추가 자리)
  tasks.py          과제 종류(회귀/분류)별 모델·출력
  evaluate.py       cv / cross / compare 평가, 요약·paired 비교·하위 집단
  permtest.py       순열 검정 (정답 섞기 / EEG 섞기, 기관 내·기관 간)
  tools.py          setup (인코더 코드·가중치 받기, local.yaml 생성), collect (요약 결과 모으기)
train.py / test.py / extract.py   ★ JSON 으로 움직이는 실행 진입점 (scripts/*.sh 가 부름)
scripts/            ★ extract.sh, train.sh, test.sh, run_all.sh
medley/report.py    ★ 환자별 per_case.csv, 그림
run.py              기존 실행 진입점 (download, inspect, setup, collect, cross 등은 여기)
splits/             고정 split (팀 공유 — 누가 돌려도 같은 환자 분할)
results_summary/    요약 결과 (collect 로 생성, 팀 공유)
notebooks/team_quickstart.ipynb   ★ 팀원 시작용: WORK 만 정하면 처음부터 끝까지 재현
notebooks/run_colab.ipynb   Colab 실행 순서 (초기 버전)
notebooks/verify_scerebro_pd_eeg.ipynb   S-CEReBrO 검증 기록 (원본 대조 + 전체 실행 + 인코더 비교)
notebooks/iowa_2x2.ipynb   Iowa 구조 확인 → 두 인코더 → 누출 점검 → 기관 간 검증 → 2×2 표
tests/smoke_test.py         가짜 데이터로 평가 파이프라인 점검 (GPU·EEG 파일 불필요)
```

## medley_pd 사용 시 반드시 지켜야 할 규칙

1. 분할 파일(splits/)은 새로 만들지 않습니다.
--new-splits 옵션을 사용하지 않으며, splits/ 폴더의 CSV 파일을 수정하거나 삭제하지 않습니다. 모든 실험이 같은 환자 분할을 사용해야 모델 간 성능을 짝지어 비교할 수 있습니다. 분할이 하나라도 다르면 그 결과는 다른 결과와 비교할 수 없습니다.

2. 환자 관련 파일은 GitHub에 올리지 않습니다.
per_case.csv, predictions.csv.gz, subjects.csv, participants.tsv, EEG 원본, 특징 파일(.npz), 개인 설정 파일(configs/local.yaml)은 올리지 않습니다. 이 파일들에는 환자 ID, 임상정보, 개인 경로가 포함되어 있습니다. 결과를 공유할 때는 python run.py collect로 모은 요약 파일만 올립니다.

3. 공통 설정(configs/config.py)은 팀 합의 없이 변경하지 않습니다.
필터 대역, fold 수, seed, 규제 범위 등 공통값은 팀이 합의한 경우에만 변경합니다. 실험마다 달라지는 조건은 train_config_<데이터셋>.json에서만 조정합니다. 공통값을 개인적으로 바꾸면 해당 결과는 팀의 다른 결과와 비교할 수 없게 됩니다.

4. 실험마다 고유한 experiment_name을 지정합니다.
experiment_name=no_ledd처럼 실험 조건을 알 수 있는 이름을 정합니다. 같은 이름을 사용하면 기존 결과를 덮어쓰게 되므로, overwrite=true는 본인의 실험을 다시 실행할 때만 사용합니다.

5. 코드는 main 브랜치에 직접 올리지 않고, 브랜치를 만들어 Pull request로 올립니다.
새 브랜치에서 작업한 뒤 Pull request를 생성하고, 한 명 이상의 확인을 받은 후 병합합니다. 이를 통해 한 사람의 실수가 전체 코드에 바로 영향을 주는 것을 방지합니다.

추가 권장 사항

정답 변수(moca)나 입력 금지 변수(ctt, updrs_total)를 포함한 결과는 본 결과로 사용하지 않습니다. 코드는 기본적으로 이를 차단하며, allow_excluded=true는 누출 여부를 확인하는 용도로만 사용합니다.
새 결합 모델은 medley/fusion.py에 등록하고, 같은 실행 안에서 stack, concat, 임상정보만 사용한 모델과 함께 비교합니다.


## 팀원 시작 가이드

**Colab**: `notebooks/team_quickstart.ipynb` 를 열고 맨 위 `WORK`(내 작업 폴더)만 정한 뒤 위에서부터 실행하세요. 누구의 Drive 에도 의존하지 않습니다.

**VS Code·서버**
```bash
git clone https://github.com/jeongwon8u8/Medley_skkai && cd Medley_skkai/medley_pd
pip install -r requirements.txt
python tests/smoke_test.py                                   # 설치 확인 (가짜 데이터)
python run.py setup --work ~/medley_work --data ~/medley_work/data   # 인코더 코드·가중치 받기 + configs/local.yaml 생성
python run.py download --dataset pd_eeg                      # 데이터 받기 (OpenNeuro)
python run.py extract --dataset pd_eeg --encoder cbramod     # GPU
python run.py cv --dataset pd_eeg --encoder cbramod --task moca
```
- 경로는 `configs/local.yaml` 에 한 번만 적습니다 (`setup` 이 만들어 줌). 이 파일은 GitHub 에 올라가지 않습니다.
- **`splits/` 는 지우거나 `--new-splits` 로 다시 만들지 마세요.** 팀 결과 비교의 기준입니다.
- 피처는 같은 코드·가중치면 각자 만들어도 같습니다. 용량이 커서 GitHub 에 올리지 않습니다.
- 새 결과는 `python run.py collect` 로 `results_summary/` 에 모아 올립니다 (환자별 예측·피처 제외, 개인 경로 제거).

## 실험 실행 (JSON + bash, VS Code)

**무엇을 어디서 바꾸나**

| 바꾸는 것 | 위치 |
|---|---|
| 모든 실험에서 통일하는 값 (필터, 5-fold, seed 0~9, Ridge·로지스틱 범위, bootstrap) | `configs/config.py` |
| 데이터셋·인코더·과제 사양 (열 이름, 기준전극, 입력 사양, MoCA < 26, 입력 금지 변수) | `configs/datasets`, `encoders`, `tasks` 의 yaml |
| 개인 경로 | `configs/local.yaml` (`python run.py setup` 이 만듦, GitHub 제외) |
| 실험마다 바꾸는 조건 | `configs/train_config_<데이터셋>.json` |
| 평가 조건 | `configs/test_config_<데이터셋>.json` |
| 한 번만 바꿔 보기 | 실행할 때 `key=value` |

**처음 한 번 (서버 터미널)**
```bash
cd medley_pd
pip install -r requirements.txt
python run.py setup --work /data2/<내 폴더>/medley_work --data /data1/<데이터 폴더>   # 인코더 받기 + local.yaml
python run.py download --dataset pd_eeg
python run.py download --dataset iowa_rest
```

**실행**
```bash
bash scripts/extract.sh pd_eeg cbramod          # 특징 추출 (GPU). limit=3 으로 시험 실행
bash scripts/extract.sh pd_eeg scerebro
bash scripts/train.sh   pd_eeg cbramod          # 학습: 고정 split 의 모든 seed·fold
bash scripts/train.sh   pd_eeg scerebro
bash scripts/test.sh    pd_eeg                  # 평가 + 환자별 CSV + 그림 + 인코더 비교
bash scripts/run_all.sh iowa_rest               # 위 다섯 줄을 한 번에
```
오래 걸리면 `nohup bash scripts/run_all.sh pd_eeg > pd_eeg.log 2>&1 &`.

**임상변수 켜기/끄기** — `clinical_sets` 안의 조합마다 변수별로 `true`/`false`. 조합을 여러 개 적으면 한 번에 학습하고 같은 split 에서 짝지어 비교합니다 (clinical-only, 각 결합 모델, EEG-only 와).
```json
"clinical_sets": {
  "C0":  {"age": true, "sex": true, "updrs_part_iii": false, "disease_duration": false, "ledd": false, ...},
  "C1":  {"age": true, "sex": true, "updrs_part_iii": true,  "disease_duration": true,  "ledd": true,  ...}
}
```
```bash
bash scripts/train.sh pd_eeg cbramod experiment_name=no_ledd clinical_sets.C1.ledd=false
bash scripts/train.sh pd_eeg cbramod experiment_name=updrs_only 'clinical_sets.U={"updrs_part_iii":true}'
bash scripts/test.sh  pd_eeg experiment_name=no_ledd
```
- 데이터셋에 없는 변수, 정답(`moca`)을 켜면 멈춥니다.
- 과제가 입력 금지로 정한 변수(`ctt`, `updrs_total`)는 `allow_excluded=true` 일 때만 켤 수 있습니다 (누출 확인용, 본 결과로 쓰지 않음).
- 같은 `experiment_name` 이 이미 있으면 멈춥니다 (`overwrite=true` 로 덮어쓰기).

**결합 방식 켜기/끄기** — `"models": {"fusion": {"stack": true, "concat": true}}`. 새 결합 모델은 `medley/fusion.py` 에 등록한 뒤 여기에 `"이름": true`. `concat` 의 PCA 차원은 `fusion_params.concat_pca`.

**결과** — `results/<dataset>/moca_impaired/<encoder>/<experiment_name>/`
- `predictions.csv.gz`, `subjects.csv`, `config_used.yaml` (train)
- `test/summary.csv`, `paired.csv`, `metrics_per_seed.csv`, `subgroup.csv`, `per_case.csv`, `figures/*.png` (test)
- 인코더 비교: `results/<dataset>/moca_impaired/compare_<실험>_cbramod_vs_scerebro.csv / .png`
- 순열 검정: test JSON 의 `permtest.label` / `permtest.eeg` 를 `true` (오래 걸림)
- `per_case.csv` 에는 환자 ID 가 있어 GitHub 에 올리지 않습니다. `python run.py collect` 도 가져가지 않습니다.

**결과 읽을 때 주의** — `mean` 기준선의 AUROC 가 0.5 보다 낮게(약 0.4) 나오는 것은 버그가 아닙니다. `mean` 은 fold 마다 "학습 fold 의 인지저하 비율"을 모든 환자에게 똑같이 예측하는데, 평가 fold 에 양성이 많으면 학습 fold 에는 적어서 예측이 낮아지므로 fold 를 모아 AUROC 를 계산하면 0.5 아래로 치우칩니다. 찍는 수준은 0.5 로 보고, `eeg vs mean` 비교는 이 점을 감안해 읽습니다.

**그림의 한글** — 한글 글꼴이 없는 환경(Colab 기본)에서는 그림 글자가 자동으로 영어로 나옵니다. 한글로 보려면 Colab 에서 `!apt-get -qq install -y fonts-nanum` 후 런타임을 다시 시작하세요.

**원본 `run.py cv` 와 같은 결과인지** — 가짜 데이터(PD-EEG·Iowa 형식, 두 인코더, seed 3개)로 `run.py cv/compare/permtest` 와 `train.sh`·`test.sh` 를 돌려 비교: 예측값 차이 0, summary·paired·compare·permtest 모두 같음. `tests/smoke_test.py` 도 통과.

## 실행 (기존 run.py)

```bash
python tests/smoke_test.py                                    # 설치 확인 (가짜 데이터, 2~4분)
python run.py download --dataset pd_eeg
python run.py inspect  --dataset pd_eeg                       # 열·결측·채널 매핑 확인
python run.py extract  --dataset pd_eeg --encoder cbramod     # --limit 3 으로 시험 실행
python run.py cv       --dataset pd_eeg --encoder cbramod --task moca
python run.py cross    --train pd_eeg --test iowa_rest --encoder cbramod --task moca
python run.py compare  --dataset pd_eeg --task moca --encoders cbramod scerebro   # 인코더끼리 짝지은 비교
python run.py permtest --dataset pd_eeg --encoder cbramod --task moca --mode label  # 순열 검정: 정답 섞기
python run.py permtest --dataset pd_eeg --encoder cbramod --task moca --mode eeg    # 순열 검정: EEG 만 섞기
python run.py permtest --dataset pd_eeg --test iowa_rest --encoder cbramod --task moca   # 기관 간 순열 검정
python run.py collect                                                               # 요약 결과 → results_summary/
```
경로 등은 `--set key=value` 로 덮어씁니다 (예: `--set paths.data_root=/content/data`).
기존 노트북의 CBraMod 캐시(`<sid>_EO.npz`)는 `--set paths.legacy_feature_dir=<폴더>` 로 재사용됩니다.

## 공통 프로토콜

| 항목 | 설정 | 비고 |
|---|---|---|
| 전처리 | reference 유지 → 인코더 채널을 이름으로 선택(없거나 평평하면 보간) → 0.3–75 Hz → **노치 = 데이터셋 line_freq** → 인코더 fs 로 리샘플링 → 비중첩 구간 → μV / amp_div | 기존과 동일. 노치만 데이터셋별 (PD-EEG 50, Iowa 60 Hz) |
| 잡음 | 100 μV 초과 구간은 표시만 (제외 안 함) | 피험자별 비율을 `noise` baseline 으로 사용 |
| 피험자 표현 | 구간 임베딩 평균 | |
| 대상 | 환자만, target 있음, 피처 있음 | |
| split | 피험자 단위 5-fold × seed 10개, target 분위수(+하위 집단) 층화, **파일로 고정** | 모든 인코더·모델이 같은 split |
| EEG-only | StandardScaler + RidgeCV (α = 10^0…10^5) | 기존 relate 실험과 동일 |
| clinical-only | 중앙값 대체 + StandardScaler + RidgeCV (α = 10^-2…10^3) | |
| fusion `stack` | EEG 예측 점수 1개 + clinical → Ridge. 학습 fold EEG 점수는 내부 5-fold OOF | 기존 실험 3 스태킹과 동일 구조 |
| fusion `concat` | EEG PCA 32차원(학습 fold 적합) ‖ clinical → Ridge | "단순 concatenation + head" 비교군 |
| baseline | `mean`(학습 평균), `noise`(잡음 비율만) | |
| 지표 (회귀) | MAE, RMSE, R², Spearman, Pearson (원래 단위) | log 변환 과제도 역변환 후 계산 |
| 분류 과제 | `type: classification`, `positive: {below: 26}` → 0/1. 모든 모델이 로지스틱(C 는 학습 fold 안 5-fold AUROC 로 선택), stack 의 EEG 점수는 logit. split 은 정답(0/1)으로 층화 | 지표: AUROC(주), 균형정확도·민감도·특이도(0.5 기준), Brier. paired·compare 는 ΔAUROC(양수 = A가 나음) |
| 불확실성 | seed 간 평균 ± SD, seed 평균 예측의 피험자 bootstrap 95% CI | |
| paired 비교 | fusion vs clinical, fusion vs EEG, EEG vs mean, clinical vs mean | ΔMAE·ΔSpearman CI, seed 중 A가 나은 횟수 |

## 순열 검정 (`permtest`)

부트스트랩 신뢰구간은 "이 환자들 안에서의 흔들림"만 반영합니다. Iowa 누출 점검에서 정답을 섞어도 EEG Spearman 0.30(CI 0.10~0.48)이 나온 사례처럼, CI 만으로는 우연과 신호를 가를 수 없어 순열 검정을 표준으로 둡니다.

| 모드 | 섞는 것 | 묻는 것 |
|---|---|---|
| `label` | 정답(target)을 환자끼리 | 각 모델 성능이 우연 분포 밖인가? |
| `eeg` | EEG 피처만 환자끼리 (정답·임상정보 유지) | EEG 의 **추가 가치**(fusion − clinical)가 우연 밖인가? |
| 기관 간 (`--test`) | 학습 기관의 정답(또는 EEG) | 처음 보는 기관에서의 성능이 우연 밖인가? |

- p = (1 + 우연 결과 중 관측 이상으로 좋은 횟수) / (1 + 순열 수). 기관 내는 고정 split 의 첫 seed 로 계산 (`--seeds` 로 변경).
- **보정 확인**: 신호 없는 가짜 EEG 10세트에서 p 가 0.08~1.0 으로 고르게 퍼짐 (거짓 양성이 5% 수준).
- **비교 항목이 많으면 5% 는 우연히 걸립니다. 결론에는 미리 정한 주 검정만 씁니다**: ① EEG-only Spearman (label) ② stack:C1 − clinical:C1 MAE (eeg) ③ 기관 간 EEG-only Spearman (label).

## 결과 파일 — `results/<dataset>/<task>/<encoder>/`

- `summary.csv`: 모델별 성능 (`avg_*` = seed 평균 예측, `*_mean/_sd` = seed 간)
- `paired.csv`: 모델 쌍 비교. **dMAE < 0 이고 CI 가 0을 넘지 않으면 A가 확실히 나음**
- `subgroup.csv`: 하위 집단(PD/PDM)별 MAE
- `permtest_label.csv`, `permtest_eeg.csv`: 순열 검정 (관측값, 우연 분포 평균·95% 경계, p)
- `compare_<A>_vs_<B>.csv` (`compare` 실행 시, 과제 폴더): 같은 환자·같은 seed 로 짝지은 인코더 간 ΔMAE·ΔSpearman CI. EEG 를 쓰지 않는 모델의 예측이 인코더 간 다르면 실행을 멈춤
- `metrics_per_seed.csv`, `predictions.csv.gz`, `config_used.yaml`
- 기관 간: `results/cross/<A>__to__<B>/<task>/<encoder>/`

## 확장 방법

**데이터셋 추가** — `configs/datasets/<이름>.yaml` 하나. `columns` 에 표준 이름 ← 원본 열 이름, `line_freq`, `file_pattern`, `patient_regex`. 그 데이터셋에만 있는 변수 조합은 `clinical_sets.<task>` 로 지정.

**인코더 추가** — `medley/encoders/<이름>.py` 에 `embed(segs[N,C,T]) -> [N,D]` 를 구현하고 `@register("<type>")`, `configs/encoders/<이름>.yaml` 에 입력 사양(fs, seg_sec, amp_div, channels). 코드·가중치만 있으면 `type: torch_module` 로 코드 없이 연결 가능.

**fusion 모델 추가** — `medley/fusion.py` 에 `Fusion` 을 상속해 `fit(F, C, y, cache)` / `predict(F, C)` 를 구현하고 `@register_fusion("이름")`, `train_config_*.json` 의 `models.fusion` 에 `"이름": true` 추가 (기존 `run.py` 는 `configs/config.py` 의 `models.fusion`). 모든 clinical set 에 자동 적용됩니다.

**과제 추가** — `configs/tasks/<이름>.yaml` 에 target, transform, exclude(입력 금지), clinical_sets. 분류는 `type: classification` 과 `positive: {below: x}` (또는 `above`) 만 더하면 같은 split·비교 모델·compare·cross 가 그대로 작동.

## S-CEReBrO 연결 방식

| 항목 | 설정 | 근거 (BioFoundation) |
|---|---|---|
| 샘플링·패치 | 200 Hz, 1초 패치, 5초 구간 (토큰 19채널 × 5패치) | `SCEReBrO_tiny.yaml`, `make_mumtaz.py` |
| 필터 | 0.3–75 Hz + 노치 (공통 전처리와 동일) | `make_mumtaz.py` |
| 정규화 | 구간마다 채널별 min-max → [-1, 1], eps 1e-6 (μV) | `LMDBDataset._min_max_normalize` |
| 전극 좌표 | (활성 전극, 기준전극) BESA 3D 좌표 [19, 2, 3] | `referential_coordinates` |
| 구조 | tiny: embed 180, depth 6, heads 5, max_channels 64, num_channels 19 | `SCEReBrO_tiny.yaml` |
| 가중치 | `model.` 접두어 제거, 모양 맞는 텐서만 로드, 90% 미만이면 중단 | `checkpoints.py`, `_partial_load` |
| 임베딩 | 토큰 [B, 95, 180] → 채널별 패치 평균 → 3,420차원 (`pool: mean` 이면 180) | CBraMod 와 같은 풀링 |

## 기존 노트북 대비 바뀐 점

- 과제를 분류(PD vs HC, MoCA<26) → **연속값 회귀**로 바꾸고 지표를 MAE·Spearman 중심으로
- 노치 주파수를 데이터셋별로 (Iowa 는 60 Hz 전원 → 기존 50 Hz 고정이면 잡음이 남음)
- split 을 파일로 고정해 인코더·모델 간 paired 비교가 가능하도록
- `mean`·`noise` baseline 을 모든 실험에 포함
- 입력 금지 변수(target, `updrs_total` 등)가 clinical set 에 들어가면 실행이 멈춤
- 피처 캐시에 전처리 사양(spec.json)을 기록해 다른 설정이 섞이지 않도록
- 하위 집단(PD/PDM) 층화와 하위 집단별 성능 보고

## 검증 기록 (2026-11-12, Colab T4)

| 검증 | 방법 | 결과 |
|---|---|---|
| 평가 코드 | `tests/smoke_test.py` (가짜 데이터) | 통과 |
| 참가자 표 | `inspect --dataset pd_eeg` | PD 116명 (PD 31 / PDM 85), MoCA 113명 — 기존과 동일 |
| EEG 전처리·CBraMod 추출 | 새 코드로 2명 추출 → 기존 노트북 캐시와 비교 | 구간 수·피처·잡음 표시 모두 동일 (최대 차이 0.0) |
| 기존 결과 재현 | `cv --task moca` EEG-only | Spearman 0.41, R² 0.15 (기존 추가 실험 B: r 0.446, R² 0.16) |
| 재실행 재현성 | 같은 고정 split 으로 다시 실행 | 소수점까지 동일 |
| 누출 점검 | MoCA 를 환자끼리 섞고 실행 (2회) | 모든 모델 MAE ≈ mean 기준선, R² ≤ 0.04 |
| S-CEReBrO 가중치 | `extract --encoder scerebro` | 97/97 텐서 로드 (shape mismatch 0) |
| S-CEReBrO 입력·출력 | BioFoundation 원본 `referential_coordinates`, `LMDBDataset._min_max_normalize`, 모델 직접 호출과 비교 (sub-030) | 좌표·정규화·임베딩 차이 모두 0.0 |
| 전처리 순서 차이 | 원본(리샘플링→필터) vs 공통(필터→리샘플링) | 피험자 평균 임베딩 상관 1.00000, 상대 차이 0.05% |
| S-CEReBrO 전체 추출 | PD 116명 | 전원 성공, 구간 48개(1명 49), 보간 Cz 114·F7+Cz 1·Cz+O2 1 (기존 보고서와 일치) |
| 인코더 간 일관성 | `cv` 2회 (cbramod, scerebro), `compare` | 같은 113명·같은 split, EEG 미사용 모델 예측이 소수점까지 동일 |
| Iowa 구조 | eeg.json, 헤더 100명 전수 확인 | 기준전극 Pz·접지 AFz 전원, 19채널 중 Pz 만 없음(보간), 500 Hz 전원, 추가 채널(I1·I2·Iz·Resp·XYZ)은 미사용 |
| Iowa 진폭 | 필터 후 채널 SD 를 PD-EEG 와 비교 | 중앙값 15~18 vs 12 μV (단위 일치). 잡음 구간 56% 는 이마 채널 눈 깜빡임 |
| Iowa 추출 | 두 인코더 100명 | 전원 성공, 구간 수 분포·잡음 비율이 두 인코더에서 동일 |
| 분류 과제 추가 후 회귀 불변 | 수정 전·후 코드로 같은 가짜 데이터 회귀 실행 | MoCA·CTT 예측 최대 차이 0.0 (기존 결과 그대로 유효) |
| Iowa 연결 | `inspect --dataset iowa_rest` | 열 매핑 일치, PD 100명 전원 MoCA |
| Iowa 누출 점검 | MoCA 섞기 3회 | 2회는 기준선, 1회는 EEG Spearman 0.30 → 누출은 아니지만 Iowa 기관 내 EEG 결과(0.23)는 우연 범위 → 순열 검정 도입 |
| 순열 검정 보정 | 신호 없는 가짜 EEG 10세트 | p 0.08~1.0 고르게 분포 (거짓 양성 약 5%) |

**첫 결과 (PD-EEG, CBraMod, MoCA, seed 10개)**: clinical-only 대비 EEG 를 더한 stack fusion 의 ΔMAE — C0 −0.119 (95% CI −0.204, −0.035), C1 −0.070 (−0.137, −0.002), C2 −0.053 (−0.112, +0.006). 임상변수가 늘수록 EEG 의 추가 이득이 줄어듦.

**인코더 비교 (PD-EEG, MoCA, 같은 split)**: S-CEReBrO − CBraMod 의 ΔMAE — EEG-only +0.148 (95% CI +0.006, +0.298, S-CEReBrO 가 나은 seed 0/10), stack C0 +0.081 (+0.013, +0.148), stack C1 +0.067 (+0.016, +0.120). S-CEReBrO 는 clinical-only 대비 추가 이득이 확인되지 않음 (stack C1 ΔMAE −0.002, CI −0.057, +0.052). frozen + 선형 head, 채널별 패치 평균 풀링 기준이며, 구간·채널별 min-max 정규화로 절대 진폭 정보가 사라지는 점이 원인 후보 (미검증 가설).

## 결과 기록 — MoCA 예측, frozen 인코더 + 선형 head (2026-11-12, 순열 검정 전 예비 결과)

**기관 내** (seed 10개 평균 예측. 괄호는 Spearman)

| 모델 | PD-EEG CBraMod | PD-EEG S-CEReBrO | Iowa CBraMod | Iowa S-CEReBrO |
|---|---|---|---|---|
| mean | 2.513 | 2.513 | 3.042 | 3.042 |
| EEG-only | **2.239 (0.41)** | 2.387 (0.32) | 2.974 (0.23) | 3.105 (0.13) |
| clinical:C0 (나이·성별) | 2.241 (0.47) | 2.241 (0.47) | 2.915 (0.34) | 2.915 (0.34) |
| stack:C0 | **2.123 (0.52)** | 2.204 (0.48) | 2.908 (0.36) | 2.940 (0.32) |
| clinical:C1 / C1_iowa | 2.198 (0.50) | 2.198 (0.50) | 2.865 (0.33) | 2.865 (0.33) |
| stack:C1 / C1_iowa | **2.129 (0.52)** | 2.196 (0.49) | 2.850 (0.34) | 2.878 (0.31) |

- EEG 추가 이득(stack − clinical, 95% CI 가 0 을 넘지 않음)은 PD-EEG × CBraMod 의 C0(−0.119)·C1(−0.070)에서만 확인.
- Iowa 기관 내 EEG 는 MoCA 섞기 결과와 구별되지 않음.

**기관 간** (공통 변수 C0, 전체 구간. MAE (Spearman))

| 방향 | 모델 | CBraMod | S-CEReBrO |
|---|---|---|---|
| PD-EEG → Iowa | mean | 2.940 | 2.940 |
| | clinical:C0 | 2.809 (0.36) | 2.809 (0.36) |
| | EEG-only | 3.408 (**0.27**) | 4.077 (0.05) |
| | stack:C0 | 3.179 (0.40) | 2.897 (0.34) |
| Iowa → PD-EEG | mean | 2.690 | 2.690 |
| | clinical:C0 | 2.418 (0.42) | 2.418 (0.42) |
| | EEG-only | 2.622 (**0.41**) | 2.714 (0.04) |
| | stack:C0 | **2.130 (0.52)** | 2.417 (0.43) |

- CBraMod 는 기관을 넘어 **순위** 정보가 옮겨 감 (0.27, 0.41). 예측값의 크기(보정)는 기관마다 크게 어긋남 (EEG-only MAE 가 mean 보다 나쁨).
- S-CEReBrO 는 기관을 넘어가는 신호가 거의 없음. 앞 24구간(2분)만 쓰면 대체로 약간 나빠짐 (`*_seg24` 결과).
- 다음: 순열 검정으로 주 검정 확정, 기관별 피처 표준화·평균 기준전극 조건으로 보정 실패 원인 확인.

## 실행 전 확인할 것

- **Iowa**: 참가자 표는 확인 완료. `TYPE` 은 GROUP 과 같은 값이라 입력 금지, UPDRS 는 버전·Part 미확인이라 `updrs_iowa` 로 분리하고 기관 간 검증은 C0 만. 기록 길이는 사람마다 5초 구간 24~68개(대부분 24~37개, 2~3분)이고 안정 구간 표시가 없어 전체 사용. PD-EEG(48구간)와 길이를 맞춘 기관 간 비교는 `max_segments=24`. 기준전극 Pz(eeg.json), 접지 AFz, 63채널 500 Hz, 60 Hz 노치. 19채널 중 Pz 만 보간.
- **S-CEReBrO**: `encoder.repo`(BioFoundation, commit 361c017 체크아웃)와 `encoder.weights`(S_CEReBrO.safetensors)를 지정합니다. 전극 좌표에 기록 기준전극이 들어가므로 데이터셋 yaml 의 `reference` 가 필요합니다 (PD-EEG = CZ, Iowa = PZ). `run.py setup` 이 코드·가중치를 받아 줍니다. 시험 실행 때 `loaded=… shape_mismatch=0` 을 확인하세요.
- **교육 수준**: 참가자 표에 있으면 `columns` 와 C0 에 추가하세요 (MoCA 는 교육 영향이 큼).
- **CTT**: 공개 `ctt` 열의 정의(CTT1/CTT2, 단위)를 확인 후 `ctt.yaml` 확정.
- `mean` baseline 의 Spearman 이 음수로 나오는 것은 교차검증에서 평가 fold 를 뺀 평균을 쓰기 때문에 생기는 정상적인 착시입니다.
