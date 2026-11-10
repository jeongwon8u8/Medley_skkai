# START HERE — 팀원 공통 시작 안내

이 저장소는 **동일한 GitHub 저장소를 각자 로컬 PC/서버에 clone하고, 각자 VS Code Codex를 실행**하는 협업 구조입니다. 하나의 Codex 세션/로그인을 공유하지 않습니다. Codex가 팀원 간 실시간 동기화되는 것은 아니며, 소스와 결정 사항은 **Git 커밋·PR·문서**를 통해 공유합니다.

## 1. 저장소 최초 설정 (관리자 1명)

1. GitHub에서 비공개 저장소를 생성합니다. 추천 이름: `co-deep-learning-eeg-clinical`.
2. 이 ZIP의 **내부 폴더 파일 전체**를 새 저장소 루트에 업로드하거나 Git으로 push합니다. ZIP 파일 자체만 올리지 마세요.
3. GitHub Settings → Collaborators and teams에서 팀원 4명을 초대합니다.
4. Settings → Branches / Rulesets에서 `main` PR 필수, 리뷰 1명 이상, 가능하면 CI 통과를 요구하도록 설정합니다.
5. `docs/ownership.md`의 임시 역할 담당자를 실제 GitHub ID로 업데이트하고 PR로 합칩니다.
6. 데이터 접근권한 및 저장 장소는 팀 내부에서 별도로 정합니다.

**로컬 Git으로 업로드 (PowerShell, 저장소 루트에서):**

```powershell
git init
git branch -M main
git add README.md START_HERE.md AGENTS.md .gitignore .gitattributes .editorconfig pyproject.toml configs docs src scripts tests experiments .github
git commit -m "chore: initialize EEG-clinical research collaboration repository"
git remote add origin https://github.com/<OWNER>/co-deep-learning-eeg-clinical.git
git push -u origin main
```

`<OWNER>`는 실제 GitHub 조직명/계정명으로 바꾸세요. GitHub에서 README를 미리 생성했다면 remote와 로컬의 초기 이력이 다를 수 있으므로 빈 저장소를 만드는 편이 간단합니다. 보호 규칙은 최초 push 뒤 설정해도 됩니다.

## 2. 각 팀원 초기 설정

```powershell
git clone https://github.com/<OWNER>/co-deep-learning-eeg-clinical.git
cd co-deep-learning-eeg-clinical
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest -q
git status
git branch --show-current
```

VS Code에서 **클론한 폴더를 열고 Codex를 사용**하세요. 원격 서버의 별도 폴더에서 작업하는 팀원도 같은 GitHub remote를 사용합니다.

## 3. Codex 첫 지시

Codex 대화에 다음 문장을 붙여넣으세요.

> 이 저장소의 `AGENTS.md`, `docs/ownership.md`, `docs/contracts.md`, `docs/git_workflow.md`를 읽고, 내 역할에 해당하는 `docs/prompts/*.md`를 읽어주세요. 먼저 `git status`와 현재 브랜치를 확인하고 이번 작업에 필요한 파일 및 계획을 알려주세요. 내 담당 영역 밖 수정·새 패키지 설치·커밋·push·PR 생성은 승인 없이 수행하지 마세요. 기존 변경을 덮어쓰지 마세요.

## 4. 역할 선택 (담당자 확정 전에는 임시)

| 역할 | 프롬프트 | 기본 브랜치 예시 |
|---|---|---|
| 연구 통합·실험 프로토콜 | `docs/prompts/lead.md` | `feat/integration-*` |
| EEG 인코더 | `docs/prompts/eeg.md` | `feat/eeg-*` |
| 임상정보·LLM | `docs/prompts/clinical.md` | `feat/clinical-*` |
| Fusion 아키텍처 | `docs/prompts/fusion.md` | `feat/fusion-*` |
| 강건성·평가 | `docs/prompts/evaluation.md` | `feat/evaluation-*` |

**역할은 제안안**입니다. 실제 팀원/실험 우선순위가 확정되면 수정하세요. 한 사람이 여러 역할을 맡아도 소유 영역 원칙은 유지합니다.

## 5. 일상 작업 흐름

```powershell
git switch main
git pull --ff-only origin main
git switch -c feat/eeg-cape-baseline  # 자신의 작업명으로 변경
# 담당 폴더에서 작업
python -m pytest -q
git diff --name-only
git add src/codl/eeg docs/requests/eeg.md   # 자신의 파일만 지정
git commit -m "feat(eeg): add encoder experiment"
git push -u origin feat/eeg-cape-baseline
```

GitHub에서 PR을 `main` 대상으로 열어 리뷰를 받고 병합합니다. 진행 중인 Codex가 이전 문서를 읽었을 수 있으므로 `main` 갱신 후 반드시 규칙 파일을 다시 읽게 하세요.

## 6. 금지사항 / 확인사항

- `.env`, 환자 정보, 원시 EEG, 임상 메타데이터, 다운받은 데이터, 인증 토큰, 체크포인트를 push하지 않습니다.
- **같은 환자가 train/test에 중복되지 않도록** split과 스키마를 먼저 확인합니다.
- 정답과 직접 연결된 MMSE, MoCA, UPDRS, BDI 기록 유무 등의 변수는 과제별 누출 검사 없이 사용하지 않습니다.
- 성능 숫자는 데이터셋 버전·환자 수·split·seed·코드 커밋·평가 단위와 같이 기록합니다.
- 연구 결과를 검증 없이 과장하거나 재현 완료로 표시하지 않습니다.
- `git add .`, force push, `git reset --hard`, main 직접 push는 하지 않습니다.
- **남의 수정/실험 파일을 삭제하거나 전체 포맷팅하지 않습니다.**

## 7. 문제가 생기면

공통 코드 수정 요청은 `docs/requests/<역할>.md`에 작성하고 연구 통합 담당자에게 전달합니다. 대형 모델 가중치·데이터를 공유하려면 Git LFS를 곧바로 켜기보다 연구기관 정책에 맞는 저장소와 접근권한을 먼저 정하세요.
