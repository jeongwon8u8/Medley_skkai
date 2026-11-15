# FUSION — 공통 수정 요청

요청을 작성할 때 날짜, 목적, 변경이 필요한 공통 파일/API, 입력·출력 타입, 호환성 영향, 담당자 확인 여부를 기록하세요.

## Open

- (없음)

## Resolved

- (없음)

## Phase A3 — 2026-11-15

User explicitly authorizes A3 integration, config, tests, commit and push.
Requested shared changes: optional positive `fusion_b_training` group LR overrides
(null inherits training.lr_new_modules), and ClinicalTrainer hooks for optimizer,
subject/target validation and model identifier. M1 defaults and behavior remain
unchanged; M1 never reads EEG. No split/data contract changes. Implemented within
this authorized integration scope; no external owner identity is assumed.

Review follow-up: normalize M3-only optional settings out of M1 checkpoint config
digests so existing Task 6 inference checkpoints remain compatible. No M1 weight
scope or export format change.

Final artifact-path review adds medley_pd/.gitignore entries for outputs/ and
checkpoints/: Task 6 private path policy names these application-relative
directories, while prior root-only ignore entries did not cover their JSON
metadata. No generated artifacts are added.

## Phase B — 2026-11-15

User explicitly authorizes sequential B1–B4 implementation and shared integration.
Requested: optional fusion_a configuration fields; factor M1 suffix hidden-state
selection into a reusable helper; trainer device/input and optimizer hooks; reuse
A3 safe checkpoint machinery with role-specific scopes. Preserve old M1/M3
checkpoint identity by excluding A-only settings for those architectures. No
preprocessing, cache, cohort or fixed-split changes. Owner identities remain TBD.
