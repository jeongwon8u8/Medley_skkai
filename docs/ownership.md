# 최신 역할 및 파일 소유권 (임시 배정안)

> **담당자 이름/깃허브 ID 미확정.** 연구팀 5명에 맞춘 역할 제안이며, 실제 배정 후 수정 필요. 코딩 에이전트는 확정되지 않은 개인을 추측하지 않는다.

| 역할 | 담당자 | 주요 업무 | 소유 경로 | 요청 문서 |
|---|---|---|---|---|
| Lead / Integration | TBD | 공통 규약, 데이터/평가 인터페이스, CI, 통합 PR | `src/codl/common/**`, `src/codl/data/**`, `configs/**`, `.github/**`, `pyproject.toml`, `docs/contracts.md` | `docs/requests/lead.md` |
| EEG Encoder | TBD | CBraMod/S-CEReBrO baseline 및 CAPE 검토 | `src/codl/eeg/**`, `experiments/stage1/**` | `docs/requests/eeg.md` |
| Clinical / Language | TBD | 변수 선택·누출 확인·textification·LLM embedding 검토 | `src/codl/clinical/**` | `docs/requests/clinical.md` |
| Fusion / Model | TBD | EEG-only 보존, 단순 결합, 정렬/선택적 반영 | `src/codl/fusion/**`, `experiments/stage2/**`, `experiments/stage4/**` | `docs/requests/fusion.md` |
| Robustness / Evaluation | TBD | 환자정보 교체, flip 등 스트레스 테스트 | `src/codl/robustness/**`, `experiments/stage3/**` | `docs/requests/evaluation.md` |

## 공통 경계

- `src/codl/common/`, `src/codl/data/`는 Lead 소유이며 각 팀원은 재사용만 한다.
- EEG, clinical, fusion, robustness 모듈은 명시된 public interface(`docs/contracts.md`)로만 연결한다.
- `tests/`에는 담당 모듈에 대한 테스트를 추가할 수 있지만, 공통 fixture·공통 테스트 변경은 Lead와 협의한다.
- 문서에 없는 파일 이동/삭제나 다른 사람 실험의 재실행·덮어쓰기를 금지한다.
- 상호 의존적 실험 엔트리포인트는 PR에서 소유자와 Lead의 리뷰를 요청한다.
- Lead가 다른 역할의 구현을 직접 덮어쓰지 않는다. 통합 변경은 PR로 조율한다.
- 정기적으로 역할 소유권·데이터 접근권한을 재확인한다.

## 변경 요청 절차

1. 요청자는 `docs/requests/<role>.md`에 원하는 공통 변경과 이유, 필요한 타입/함수를 적는다.
2. Lead가 승인·계약 문서 업데이트 후 공통 수정을 수행한다.
3. 요청자는 `main` 최신 내용을 자신의 브랜치에 반영하고 작업을 이어간다.
