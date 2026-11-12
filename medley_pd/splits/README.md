# 고정 split

`<dataset>__<task>.csv` (subject, seed, fold). 처음 `cv` 를 실행할 때 만들어지고, 이후 모든 인코더·모델이 같은 파일을 씁니다.
팀원 모두 같은 분할로 비교하려면 이 폴더의 csv 를 저장소에 함께 올리세요.

- `pd_eeg__moca.csv`: PD-EEG PD 113명 × seed 0–9, 5-fold (2026-11-12 생성, 검증 기록의 결과에 사용)
