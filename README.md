# Co-Deep Learning — Evidence-Preserving EEG–Clinical Multimodal AI

> 연구 프로젝트 | 뇌파 신호의 객관적 근거를 보존하는 임상 맥락 인지형 EEG–임상정보 멀티모달 모델 개발

**Status:** Research in progress (2026-11). **Target:** 2027-01-26.  
**Aim:** Preserve an independent EEG prediction pathway while using trustworthy clinical context selectively; evaluate predictive utility and reliance under controlled perturbations.

## 시작하기

**처음 참여하는 팀원은 반드시 [START_HERE.md](START_HERE.md)를 먼저 읽어주세요.**  
Codex 공통 규칙은 [AGENTS.md](AGENTS.md), 역할·소유권은 [docs/ownership.md](docs/ownership.md), 실험 규약은 [docs/contracts.md](docs/contracts.md)입니다.

## 연구 단계

| Stage | 범위 | 현재 상태 |
|---|---|---|
| 1 | EEG foundation model evaluation | 기존 결과 정리됨; 재현 검증 별도 |
| 2 | EEG-only / clinical-only / simple fusion | 기존 결과 정리됨; 일부 실험은 추가 개선 필요 |
| 3 | Clinical reliance & robustness | 정보 교체 실험 수행; 프로토콜 표준화 필요 |
| 4 | Evidence-preserving fusion | 예비 실험 수행; 정식 아키텍처 개발·검증 예정 |

**중요:** `docs/results/interim_results.md`에 포함된 수치는 중간 보고서에서 전달된 값이며, 이 저장소의 스크립트를 실행해서 재생성한 결과가 아닙니다. 데이터·설정·분할·코드가 확인되기 전까지 재현 완료로 표기하지 마세요.

## 제안 구조 (설계 중)

```text
EEG ------------------> EEG Encoder ----------> EEG-only Head ------> p_eeg
                           |                         |
                           v                         v
                      EEG embedding --------> Selective Fusion -----> p_fusion
                           ^                         ^
Clinical variables -> Textification -> Clinical/Text Encoder -> constrained clinical contribution
```

위 다이어그램은 구현 확정 사항이 아닌 **설계 방향**입니다. LLM, CAPE, contrastive alignment, reliability gating은 비교·검증 후 채택합니다.

## 디렉터리

```text
AGENTS.md                   Codex 공통 규칙
START_HERE.md               신규 팀원 안내
.github/                   PR 템플릿 / CI
configs/                   실험 설정 예시 (실제 실험 값은 추후 기록)
docs/                      연구 규약, 역할, 계획, 실험 결과, 프롬프트
src/codl/                  실험 코드 (기본 골격)
  common/                  공통 로깅·평가·분할
  data/                    로더/전처리 계약
  eeg/                     EEG encoder
  clinical/                임상정보 표현
  fusion/                  멀티모달 모델
  robustness/              교란·의존도 평가
experiments/               Stage별 실험 엔트리포인트 자리
scripts/                   개발 검증용 유틸리티
tests/                     스모크 테스트
```

## 로컬 환경 (Python)

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q src
```

현재 포함된 코드는 **협업용 기본 골격 및 평가 유틸리티**입니다. 완성된 EEG 훈련 파이프라인이나 모델 가중치는 제공하지 않습니다. 프로젝트 소스가 기존 팀원 PC에 있다면 소유권 및 데이터 보안 검토 후 단계적으로 옮겨야 합니다.

## 문서

- [연구계획 및 일정](docs/research_plan.md)
- [중간 결과 (제공된 수치)](docs/results/interim_results.md)
- [실험 규약](docs/contracts.md)
- [역할 및 책임](docs/ownership.md)
- [실험 기록 템플릿](docs/experiment_template.md)
- [GitHub 협업](docs/git_workflow.md)
- [데이터 보안 및 누출 방지](docs/data_policy.md)

## 연구자료

원본 연구계획서는 팀이 별도로 관리합니다. 저작권·접근권한·연구윤리 승인 여부를 확인한 자료만 저장소에 추가하세요. 임상 EEG 원시 신호, 개인식별정보, 다운로드한 외부 데이터셋, API 키, 대형 체크포인트는 Git에 올리지 않습니다.

## 공개 범위

처음에는 **Private repository**로 만드는 것을 권장합니다. 데이터 접근권한, 협업자, 선행 코드 라이선스, 공개 가능한 결과물을 확인한 후 공개 여부를 결정하세요.
