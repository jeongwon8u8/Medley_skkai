# Co-Deep Learning Codex 공통 작업 규칙

## 목표 및 우선순위
- 연구: EEG의 독립적 예측 근거를 유지하면서 구조화 임상정보를 선택적으로 결합하여 예측 성능·강건성·신뢰도를 검증한다.
- 2026-11 현재 Stage 1~3와 Stage 4 예비 실험 결과를 보유하고, EEG 인코더/LLM 임상 표현/뇌파 중심 정렬 구조를 검토한다.
- 이 문서, `docs/ownership.md`, `docs/contracts.md`, 현재 담당 역할 프롬프트를 우선 확인한다. 실제 코드의 확인된 동작을 문서로 임의 덮어쓰지 않는다.
- 구현되지 않은 아키텍처, 보고서 수치, 재현되지 않은 결과는 '예정', '예비', '보고 수치'로 명시한다.

## 모든 Codex 세션의 시작
1. `AGENTS.md`, `docs/ownership.md`, `docs/contracts.md`, 내 `docs/prompts/*.md`를 읽는다.
2. `git status --short`, `git branch --show-current`로 미커밋 변경과 브랜치를 확인한다.
3. 수정할 파일, 실험 질문, 데이터·계약 영향, 검증 방법을 간단히 제시한다.
4. 본인 담당 경로에서 작업한다. 미커밋 변경과 기존 파일은 지우지 않는다.
5. 공유 파일 수정은 `docs/requests/<role>.md`에 먼저 요청한다. 임의 복제·우회 구현 금지.

## 역할 및 영역
- Lead/Integration: 공통 설정·스키마·split·평가 도구·CI·통합 (`src/codl/common/`, `src/codl/data/`, `configs/`, `.github/`, `pyproject.toml`, 공통 문서).
- EEG: EEG encoder, EEG-only 연구 (`src/codl/eeg/`, `experiments/stage1/`).
- Clinical: 임상 변수 선별, 텍스트 변환, 임상 encoder (`src/codl/clinical/`).
- Fusion: 단순 결합, evidence-preserving 아키텍처 (`src/codl/fusion/`, `experiments/stage2/`, `experiments/stage4/`).
- Evaluation: 교란, 정보 의존도, 강건성 분석 (`src/codl/robustness/`, `experiments/stage3/`).
- 상세 소유권은 `docs/ownership.md`를 따른다. 팀원의 실제 이름/GitHub ID는 확인 전까지 작성하지 않는다.

## 연구·데이터 규칙
- 대상자의 식별정보/EEG 원본/민감 임상 데이터/보안 설정/토큰/접근 불가 외부 모델 가중치를 Git에 올리지 않는다.
- 환자 단위 분할, 같은 대상자의 여러 EEG 세그먼트·방문 간 교차 분할 방지. 환자 ID는 안전한 비식별 내부 키를 사용하고 Git에는 올리지 않는다.
- 전처리는 train에서 fit한 통계만 val/test에 적용한다. 전처리·sampling/channel/missingness/label 프로토콜과 version을 명시한다.
- 진단 자체나 미래 결과를 반영한 변수는 leakage audit 후 제외하거나 의도된 '누출 진단 전용' 실험으로 격리한다.
- 근거 보존: `p_eeg`와 `p_fusion`를 별도로 출력/평가하도록 설계한다. 모달리티 변경 실험에서는 EEG와 정답을 고정한다.
- 같은 판정군/반대 판정군 정보 교체는 통제된 스트레스 테스트이지 실제 임상적으로 가능한 환자 시나리오라는 주장은 하지 않는다.
- 단계별 AUROC는 다른 과제·분할·전처리·표본의 수치일 수 있다. 동일 조건 재현 없이 직접 우열 비교 금지.

## 코드 및 실험 기록
- Python 패키지 및 공통 인터페이스는 `docs/contracts.md`를 참고한다. 인터페이스 변경은 Lead와 합의한다.
- 각 실험은 가설, 데이터/코호트, split, seed, 학습/평가 파라미터, 코드 커밋, 교란 조건, 결과 위치를 기록한다.
- 논문 관련 사실, 레퍼런스, 모델 라이선스는 원문 확인 없이 단정하지 않는다.
- 새 패키지 설치, 공통 유틸 변경, 포맷/리네임/대규모 리팩터링은 Lead가 관리한다.
- 버그 수정은 최소 수정 및 작은 테스트로 검증한다. 다수 파일 무관 변경을 피한다.

## Git 및 리뷰
- 개인 브랜치 `feat/<area>-<topic>` 또는 `fix/<area>-<topic>`, PR로 `main`에 병합한다.
- 신규 문서 최초 등록 외 `main` 직접 push 금지. force push, `git reset --hard`, `git add .` 사용 금지.
- 커밋·push·PR·병합은 요청·승인된 경우에만 한다. Codex가 자동 수행했다고 주장하지 않는다.
- `git diff --name-only`, `python -m pytest -q`, 가능한 lint/type 검증 후 PR에 결과를 적는다.
- 기존 변경 충돌은 소유자와 조율하고, 필요한 경우 리뷰어와 해결한다.

## 완료 보고
- 변경 파일, 핵심 결과, 수행한 테스트, 실행하지 못한 테스트의 이유, 남은 일, 타 역할 요청사항을 요약한다.
- 새로운 점수는 재현 조건/불확실성을 같이 적는다. 검증되지 않은 결과를 논문 결과로 즉시 승격하지 않는다.
