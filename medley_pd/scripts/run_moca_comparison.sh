#!/usr/bin/env bash
# Eight-model PD-EEG OOF comparison. Run inside tmux for interactive monitoring.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == --m3-ablation ]]; then
    shift
    exec bash scripts/run_m3_ablation.sh "$@"
fi
seeds=0
parallel=3
baseline_workers=5
mode=(--controlled-gpu)
while (($#)); do
    case "$1" in
        --seeds) seeds="${2:?expected 0 or 0-9}"; shift 2 ;;
        --legacy-cpu) mode=(); shift ;;
        --baseline-workers) baseline_workers="${2:?expected 1 to 5}"; shift 2 ;;
        --max-parallel) parallel="${2:?expected 1, 2 or 3}"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/run_moca_comparison.sh [--seeds 0|0-9] [--max-parallel 1|2|3] [--baseline-workers 1–5] [--legacy-cpu]"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$seeds" == 0 || "$seeds" == 0-9 ]] || exit 2
[[ "$parallel" =~ ^[1-3]$ && "$baseline_workers" =~ ^[1-5]$ ]] || exit 2
umask 077
mkdir -p outputs/multimodal_logs
exec 9>outputs/multimodal_logs/moca_comparison.lock
flock -n 9 || { echo "Another MoCA comparison is already active." >&2; exit 2; }
logs=$(mktemp -d outputs/multimodal_logs/moca-comparison.XXXXXX)
python_bin="${PYTHON:-python}"
seed_list=(0)
[[ "$seeds" != 0-9 ]] || seed_list=(0 1 2 3 4 5 6 7 8 9)
folds_per_model=$(("${#seed_list[@]}" * 5))
total=$((folds_per_model * 3))
cat >"$logs/settings.txt" <<EOF
seeds=$seeds
folds=1 2 3 4 5
max_epochs=30
early_stopping_patience=30
max_parallel=$parallel
baseline_workers=$baseline_workers
baseline_mode=${mode[*]:-legacy-cpu}
llm_path=${MEDLEY_LLM_PATH:-/workspace/models/Qwen3-4B}
llm_revision=${MEDLEY_LLM_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}
llm_device=${MEDLEY_LLM_DEVICE:-cuda:0}
EOF
echo "Progress and model logs: $logs"
if ((parallel > 1)); then
    echo 'Concurrent GPU peak memory is unverified. Set --max-parallel 1 to reduce concurrent usage.' >&2
fi
# Fit the controlled baselines before starting the neural workers.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$python_bin" \
    scripts/compare_multimodal_oof.py baseline --seeds "$seeds" --all-baselines --full-epochs "${mode[@]}" --baseline-device "${MEDLEY_LLM_DEVICE:-cuda:0}" \
    --baseline-workers "$baseline_workers" \
    2>&1 | tee "$logs/baselines.log"

: >"$logs/folds.tsv"
export MOCA_LOG_DIR="$logs" MOCA_FOLDS_PER_MODEL="$folds_per_model" MOCA_TOTAL="$total"
export MOCA_PYTHON="$python_bin"
export MEDLEY_FIGURES_DIR=outputs/multimodal_figures_30epochs
# FD 8 remains attached to the supervisor while tqdm streams stay in per-model logs.
exec 8>&1
worker() {
    local model="$1" command
    shift
    local -a cmd=("$BASH" -c '
        set -uo pipefail
        [[ ! -t 1 ]] || stty cols 120 rows 30
        model="$1"; shift
        failures=0
        for seed in "$@"; do
            for fold in 1 2 3 4 5; do
                printf "Processing %s seed %s fold %s\n" "$model" "$seed" "$fold"
                while true; do
                    CUDA_VISIBLE_DEVICES= "$MOCA_PYTHON" scripts/moca_fold_status.py "$model" "$seed" "$fold"
                    status=$?
                    ((status == 4)) || break
                    sleep 30
                done
                outcome=reused
                if ((status == 3)); then
                    bash scripts/continue_multimodal_30.sh --allow-fresh "$model" "$seed" "$fold"
                    status=$?
                    outcome=completed
                fi
                if ((status != 0)); then outcome="failed:$status"; failures=1; fi
                (
                    flock 7
                    printf "%s\t%s\t%s\t%s\n" "$model" "$seed" "$fold" "$outcome" >> "$MOCA_LOG_DIR/folds.tsv"
                    global=$(awk "\$4 !~ /^failed/ {n++} END{print n+0}" "$MOCA_LOG_DIR/folds.tsv")
                    local_done=$(awk -v model="$model" "\$1==model && \$4 !~ /^failed/ {n++} END{print n+0}" "$MOCA_LOG_DIR/folds.tsv")
                    printf "[%s %s/%s | total %s/%s] seed %s fold %s %s\n" \
                        "$model" "$local_done" "$MOCA_FOLDS_PER_MODEL" "$global" "$MOCA_TOTAL" "$seed" "$fold" "$outcome" |
                        tee -a "$MOCA_LOG_DIR/progress.log" >&8
                ) 7>"$MOCA_LOG_DIR/progress.lock"
            done
        done
        exit "$failures"
    ' worker "$model" "$@")
    if command -v script >/dev/null; then
        printf -v command '%q ' "${cmd[@]}"
        TERM=xterm-256color SHELL="$BASH" exec script -qefc "$command" /dev/null </dev/null
    fi
    exec "${cmd[@]}"
}
pids=()
terminate() {
    local pid
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait || true
}
trap 'terminate; exit 130' INT
trap 'terminate; exit 143' TERM
models=(M1 M2 M3)
failed=0
for ((start=0; start<${#models[@]}; start+=parallel)); do
    pids=()
    for model in "${models[@]:start:parallel}"; do
        worker "$model" "${seed_list[@]}" >"$logs/$model.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
done
pids=()
if ((failed)); then
    printf 'At least one Fold failed. See %s/folds.tsv and per-model logs. Final OOF evaluation was not published.\n' "$logs" >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$python_bin" \
    scripts/compare_multimodal_oof.py evaluate --seeds "$seeds" --all-baselines --full-epochs "${mode[@]}" --baseline-device "${MEDLEY_LLM_DEVICE:-cuda:0}" \
    --baseline-workers "$baseline_workers" \
    2>&1 | tee "$logs/evaluation.log"
