"""공통 설정 (자주 안 바꾸는 값) — 예전 configs/base.yaml 을 옮긴 것

모든 데이터셋·인코더·실험이 같은 값을 쓴다. 여기를 바꾸면 모든 실험이 바뀌므로, 바꿨다면 팀에 알린다.
실험마다 바꾸는 조건(임상변수 켜기/끄기, 결합 방식, 구간 수 등)은 configs/train_config_<데이터셋>.json 에 둔다.

설정이 합쳐지는 순서 (뒤가 앞을 덮어씀)
  1) 이 파일 COMMON
  2) configs/datasets/<데이터셋>.yaml   데이터셋 사양 (열 이름, 기준전극, 전원 주파수 ...)
     configs/encoders/<인코더>.yaml     인코더 입력 사양 (200 Hz, 5초, 채널 ...)
     configs/tasks/<과제>.yaml          과제 정의 (moca_impaired: MoCA < 26 → 1, 입력 금지 변수)
  3) configs/local.yaml                 개인 경로 (GitHub 에 올리지 않음, `python run.py setup` 이 만듦)
  4) configs/*.json                     실험 조건
  5) 실행할 때 key=value                 한 번만 바꿔 보기
"""

COMMON = {
    # 경로 기본값. 실제 경로는 configs/local.yaml 에 적는다
    "paths": {
        "data_root": "./data",            # 데이터셋 폴더들의 상위 폴더
        "feature_root": "./features",     # 피처 캐시: features/<dataset>/<encoder>/<subject>.npz
        "result_root": "./results",       # 결과: results/<dataset>/<task>/<encoder>/<실험 이름>/
        "split_root": "./splits",         # 고정 split: splits/<dataset>__<task>.csv (저장소 안, 팀 공유)
        "legacy_feature_dir": None,       # 기존 노트북 캐시(<sid>_EO.npz) 폴더. 같은 전처리·CBraMod일 때만 지정
    },

    # 전처리 (기존 TDBRAIN·Mumtaz·PD-EEG 프로토콜과 동일. 인코더 입력 사양은 encoders/*.yaml)
    "preprocess": {
        "l_freq": 0.3,
        "h_freq": 75.0,
        "notch": None,                    # None 이면 데이터셋의 line_freq (PD-EEG 50 Hz, Iowa 60 Hz)
        "flag_uv": 100.0,                 # 품질 표시 기준 (구간 제외에는 쓰지 않음)
        "min_good_channels": 20,          # 보간에 쓸 최소 채널 수
    },

    # 평가 프로토콜
    "evaluation": {
        "n_folds": 5,
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # 분할 seed 반복. 모든 모델이 같은 split 사용
        "stratify_bins": 4,               # target 분위수 구간으로 층화 (분류는 정답 0/1 로 층화)
        "stratify_subgroup": True,        # 하위 집단(PD/PDM 등)도 함께 층화
        "eeg_alphas": [0, 5, 11],         # 회귀: RidgeCV alpha = logspace(0, 5, 11)
        "clinical_alphas": [-2, 3, 11],   # 회귀: logspace(-2, 3, 11)
        "eeg_Cs": [-5, 1, 13],            # 분류: EEG 로지스틱 C = logspace(-5, 1, 13), 학습 fold 안 5-fold AUROC 로 선택
        "clinical_Cs": [-3, 3, 13],       # 분류: clinical 로지스틱 C
        "stack_inner_folds": 5,           # 스태킹: 학습 fold 안 EEG 점수는 내부 OOF (누수 방지)
        "concat_pca": 32,                 # concat: EEG 피처 PCA 차원 (JSON fusion_params 로 실험별 변경 가능)
        "bootstrap": 2000,                # seed 평균 예측에 대한 피험자 bootstrap
        "bootstrap_seed": 42,
        "max_segments": None,             # 피험자당 앞에서 N개 구간만 (JSON 에서 지정)
    },

    # 비교할 모델 기본값 (JSON models 로 켜고 끔)
    "models": {
        "baselines": ["mean", "noise"],   # 학습 비율 예측, 잡음 비율만으로 예측 (sanity check)
        "clinical": True,
        "eeg": True,
        "fusion": ["stack", "concat"],    # medley/fusion.py 의 FUSION 레지스트리 이름
    },
}


# ---------------------------------------------------------------------------
# 실험 JSON 에서 쓸 수 있는 키와 기본값 (오타 검사용). 새 조건은 여기에 추가한다.
# ---------------------------------------------------------------------------
TRAIN_KEYS = {
    "dataset": None,              # pd_eeg | iowa_rest
    "encoder": None,              # cbramod | scerebro (bash 에서 두 번째 인자로 바꿀 수 있음)
    "task": "moca",               # moca (MoCA 점수 회귀, 주 과제) | moca_impaired (MoCA < 26 분류) | ctt (회귀)
    "experiment_name": "main",    # 결과 폴더 이름. 같은 이름을 인코더마다 두면 test 가 인코더끼리 비교
    "overwrite": False,           # 같은 이름의 결과가 있으면 기본은 중단
    "seeds": None,                # None 이면 COMMON 의 seed 0~9
    "max_segments": None,         # 피험자당 앞 N개 구간만 (예: 24 = 2분)
    "models": {},                 # {"baselines": {"mean": true, "noise": true}, "eeg": true, "clinical": true,
                                  #  "fusion": {"stack": true, "concat": true}}
    "fusion_params": {},          # {"concat_pca": 32}
    "clinical_sets": {},          # {"C0": {"age": true, "sex": true, "ledd": false, ...}, "C1": {...}}
    "evaluation": {},             # 공통 평가값을 이 실험에서만 덮어쓰기 (예: {"bootstrap": 500}). 권장하지 않음
    "allow_excluded": False,      # true 면 과제의 입력 금지 변수(ctt, updrs_total)도 켤 수 있음 (누출 확인 전용)
    "limit": None,                # extract 시험 실행: 하위 집단마다 앞 N명
    "include_controls": False,    # extract: HC 도 추출
}

TEST_KEYS = {
    "dataset": None,
    "task": "moca",               # 학습 때와 같은 과제
    "experiment_name": "main",    # 평가할 실험 이름
    "max_segments": None,         # 학습 때 max_segments 를 썼다면 같은 값
    "encoders": ["cbramod", "scerebro"],   # 평가할 인코더. 2개 이상이면 첫 인코더 기준으로 짝지은 비교
    "save_per_case": True,        # 환자별 예측 per_case.csv
    "save_png": True,             # 그림
    "permtest": {"label": False, "eeg": False, "n": 100, "seeds": None},   # 순열 검정 (시간이 오래 걸림)
}
