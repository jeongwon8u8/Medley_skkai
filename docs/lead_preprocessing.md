# LEAD 인코더 전처리 (PD MoCA 회귀)

- 담당: 은빈 · 작성: 2026-11-24 · 브랜치: `feat/lead-encoder` (기준 `feat/multimodal-framework`)
- 목적: frozen LEAD(P-Base)를 **사전학습 때와 같은 입력 조건**으로 쓰기 위한 전처리. 결합 모델은 기존 `medley/fusion.py`(stack, concat) 그대로 사용

## 1. 변경 요약

| 항목 | 기존 (CBraMod) | LEAD | 이유 | 적용 범위 |
|---|---|---|---|---|
| 저역 차단 (h_freq) | 75 Hz | **45 Hz** | 팀 통일 결정. 인지 관련 리듬은 1–30 Hz, 45 Hz 이상은 근육 잡음 위주. LEAD 사전학습도 45 Hz | **공통** (`configs/config.py`) → 모든 인코더 |
| 고역 차단 (l_freq) | 0.3 Hz | **0.5 Hz** | LEAD 사전학습 조건 | LEAD만 (`lead.yaml`) |
| 기준(reference) | 기록 기준 유지 (PD-EEG Cz, Iowa Pz) | **평균 기준** (기준전극 자리 보간 후) | LEAD 사전학습 조건. 기관마다 다른 기준을 통일. Iowa 는 Pz 기준이라 두정부 신호가 약하게 기록됨 | LEAD만 |
| 노치 | 데이터셋 전원 주파수 | 같음 (50/60 Hz) | — | 변경 없음 |
| 채널 | 10-20 표준 19채널 | 같음 | LEAD 사전학습도 19채널 | 변경 없음 |
| 샘플링 | 200 Hz | 같음 | LEAD 사전학습 속도 중 하나 | 변경 없음 |
| 구간 길이 | 5초 (1,000샘플) | **2초 (400샘플)** | LEAD 기본 입력 (패치 50 × 8) | LEAD만 (`seg_sec`) |
| 진폭 처리 | μV ÷ 100 | **구간·채널별 z-score** | LEAD 원본 `normalize_batch_ts` 와 같음. 사전학습 입력 크기와 맞춤 | LEAD만 (인코더 안) |
| 잡음 구간 | 100 μV 초과 표시만 | 같음 | — | 변경 없음 |
| 인코더 | CBraMod (고정) | **LEAD P-Base (고정)**, 라벨 없는 대조학습 가중치 | — | — |
| 출력 | 19 × 200 = 3,800차원 | **19 × 128 = 2,432차원** (채널별 마지막 패치 토큰). 채널 평균 = `encode()` 128차원 | 채널 선택·Fusion 토큰에 그대로 사용 | LEAD만 |

## 2. 바뀐 파일

| 파일 | 내용 |
|---|---|
| `medley_pd/configs/config.py` | `h_freq` 75 → 45, `rereference: None` 추가 |
| `medley_pd/configs/encoders/lead.yaml` | 새 파일. LEAD 입력 사양 + `preprocess` 덮어쓰기 + 모델 구조 |
| `medley_pd/medley/eeg.py` | 인코더별 `preprocess` 덮어쓰기, `rereference: average` 처리 |
| `medley_pd/medley/config.py` | 캐시 사양(`spec.json`)에 인코더별 전처리·기준 반영 |
| `medley_pd/medley/encoders/lead.py` | 새 파일. LEAD 원본 모델 + P-Base 로드, z-score, 채널 토큰 출력 |
| `medley_pd/medley/encoders/__init__.py` | 레지스트리에 `lead` 추가 |
| `medley_pd/configs/local.example.yaml` | LEAD 경로 예시 |
| `medley_pd/scripts/check_lead.py` | 새 파일. 동작 확인 7항목 |
| `medley_pd/medley/tools.py` | `run.py setup` 이 LEAD 코드(검증 커밋)·P-Base 가중치를 받고 `local.yaml` 에 경로 기록 |

## 3. 팀원이 같은 전처리를 쓰는 방법

**원리**: 전처리는 노트북 코드가 아니라 설정 파일(`configs/encoders/lead.yaml`)에 있다. 같은 브랜치에서 `encoder=lead` 로 실행하면 누구나 같은 전처리가 적용된다. 추출할 때 특징 폴더의 `spec.json` 에 전처리 사양이 기록되고, 사양이 다른 설정으로 같은 폴더에 추출하면 자동으로 멈춘다.

1. 코드 받기: `git fetch` → `git switch feat/lead-encoder` (병합 후에는 기준 브랜치 `git pull`)
2. 환경: `pip install -r medley_pd/requirements.txt`
3. LEAD 코드·가중치·개인 경로를 한 번에: `python run.py setup --work <작업 폴더> --data <데이터 폴더>`
   - LEAD 를 검증 커밋(`dddee9c`)으로 받고, P-Base 가중치를 받고, `configs/local.yaml` 에 경로를 적는다
4. 확인: `python scripts/check_lead.py --n 8` → 모두 통과
5. 특징
   - 권장: 공유 Drive 의 `features/pd_eeg/lead/` 를 `configs/local.yaml` 의 `feature_root` 로 지정 → **추출 없이 같은 특징** 사용
   - 직접 추출: `python extract.py --config configs/train_config_pd_eeg.json encoder=lead`
6. 같은 사양인지 확인: 각자의 `features/pd_eeg/lead/spec.json` 을 공유본과 비교 (같으면 같은 전처리)

## 4. 주의

- `h_freq` 45 Hz 는 **공통 설정 변경**이라 CBraMod·S-CEReBrO 의 기존 캐시(75 Hz)와 사양이 달라짐
  - extract 는 사양 불일치를 감지해 멈추지만, **train 은 캐시 사양을 확인하지 않으므로** 예전 75 Hz 캐시를 그대로 읽음 → 비교 실험 전에 다시 추출하거나 결과에 75 Hz 캐시임을 명시
  - 팀 규칙상 공통 설정 변경이므로 PR 설명과 `docs/requests/lead.md` 에 기록하고 Lead 확인 후 병합
- Iowa(ds004584)와 TDBRAIN 은 LEAD 사전학습 데이터에 포함 → 주 평가는 PD-EEG, Iowa 결과는 "사전학습 포함" 표시
- LEAD 코드·가중치 라이선스 CC BY-NC-SA 4.0 (비상업 연구용, 출처 표시)
- 이 환경에서는 실행 검증 전. `check_lead.py` 결과를 PR 에 첨부
