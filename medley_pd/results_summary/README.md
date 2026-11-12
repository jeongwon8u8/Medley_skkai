# 요약 결과 (팀 공유)

`python run.py collect` 가 각자의 결과 폴더에서 요약 파일만 골라 이 폴더로 모읍니다.

- 포함: `summary.csv`, `paired.csv`, `subgroup.csv`, `metrics_per_seed.csv`, `compare_*.csv`, `permtest_*.csv`, `table_*.csv`, `config_used.yaml`(개인 경로 제거)
- 제외: 환자별 예측(`predictions.csv.gz`), 피처, EEG 원본
- 구조: `<dataset>/<task>/<encoder>/…`, 기관 간은 `cross/<학습>__to__<평가>/<task>/<encoder>/…`

결과를 해석할 때는 루트 README 의 **결과 기록**(조건·날짜·주 검정)을 함께 보세요.
