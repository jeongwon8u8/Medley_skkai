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

## Phase C1 — 2026-11-15

User authorizes shared experiment protocol/config/provenance integration and
commit/push. Add optional protocol policies, reuse existing split_root, seed/fold
selection and training.validation_fraction. Exclude protocol-only settings from
inference checkpoint identity to preserve previous M1/M2/M3 files. No outer CSV,
preprocessing, features, model behavior, baseline output or earlier test changes.

## Phase C2 — 2026-11-15

User authorizes runner/registry/early-stopping/run-state/CLI integration and
commit/push. Add only missing runner selection/dataloader/M0-reference settings;
reuse existing training defaults and C1 manifests. Exclude orchestration-only
runner settings from inference checkpoint digests to preserve old checkpoints.
No changes to model definitions, optimizer groups, outer splits or baseline files.

## Phase C3 — 2026-11-15

User authorizes read-only matched evaluation/statistics/reporting and commit/push.
Add minimal Task 1 analysis options; extract inference config digest into the
torch-free config module preserving the clinical checkpoint helper name; C2
completion validation keeps its full default while exposing explicit byte-hash
verification for prediction-only C3 (no tensor/model loading). No source artifacts,
metrics, prior tests, model/trainer behavior or fixed assignments are changed.

## Phase D — 2026-11-15

User authorizes sequential D1–D4 implementation, synthetic bounded training,
independent review, one commit and development-branch push. Shared changes add
explicit Phase D opt-in, controlled variants, modular pooling/adapter/numeric
branch support, architecture-aware checkpoint identity, and C2/C3 integration.
Preserve old defaults/tests, fixed splits, baseline predictions and caches.
Numeric skip initially supports only M2 with predictor-only encoding and no
fitted scaling; M1/M3 requests are rejected. No real experiment is authorized.


## Architecture A FT 실행 (2026-11-27, 사용자 승인)

외부121명 frozen GNN 벡터와 FT 두 projection 및 팀 RepresentationFusionHead를 재사용한다. 원격 클래스 본문을 출처 commit과 함께 fusion 소유 경로로 가져오고 LLM 모듈은 포함하지 않는다. 기존 raw Concat 기본 동작과 공유 split/평가/전처리는 유지한다. 테스트와 집계 문서만 함께 업로드하며 환자 데이터는 제외한다.
