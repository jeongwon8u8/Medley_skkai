# GitHub + Codex 협업 절차

## 개념

- 같은 저장소를 팀원들이 각각 `git clone`합니다. 각자 VS Code 및 Codex 인스턴스를 실행합니다.
- Codex의 대화 내용/실행 상태가 GitHub로 자동 공유되는 것은 아닙니다. 공유할 결정은 PR 설명·docs·issues에 남깁니다.
- **하나의 GitHub 계정이나 Codex 로그인/인증 토큰을 여러 명이 공유하지 않습니다.** 개인 계정과 팀원별 권한을 씁니다.

## Branch protection 권장

- `main`: PR 기반 병합, 승인 리뷰 1명 이상, CI(`tests`) 통과, direct push 제한, force push 차단.
- 예외/긴급 수정은 관리자가 합의한 절차를 사용. repo 생성 직후 최초 커밋은 예외.

## 한 작업 = 한 브랜치 = 한 PR

1. 시작 전 `git status`와 branch 확인.
2. `git switch main` + `git pull --ff-only` 후 새 `feat/<area>-<topic>` 브랜치.
3. 작은 단위로 변경, 직접 테스트 및 `git diff --name-only`로 범위 확인.
4. 담당 경로만 `git add <path>`하여 커밋.
5. push 후 PR 생성. 데이터/실험 프로토콜 변경 여부, 테스트, 재현 명령, 결과를 첨부.
6. 적절한 담당자와 Lead가 리뷰·병합. 병합 후 각자 `main` 갱신.

## 충돌 방지

- 공통 코드·config·환경 종속성은 Lead에게 요청.
- 다른 역할 폴더를 직접 고치거나 동일 파일을 대규모 리팩터링하지 않는다.
- 큰 파일·실험 산출물은 Git 밖 승인된 저장소 사용.
- 실험 설정과 결과 경로에 고유 `experiment_id`를 부여하고 overwrite하지 않는다.

## 역할과 작업 이슈 예시

- `[EEG] CBraMod channel masking baseline`
- `[Clinical] numeric-to-text format ablation`
- `[Fusion] compare simple fusion to constrained fusion`
- `[Evaluation] opposite-class swap protocol & negative flip rate`
- `[Lead] establish grouped folds and unified report schema`
