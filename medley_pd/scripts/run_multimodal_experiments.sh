#!/usr/bin/env bash
# One worker per model, serial seed/fold within each worker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
seeds=0
parallel=3
eeg=()
while (($#)); do
    case "$1" in
        --seeds) seeds="${2:?expected 0 or 0-9}"; shift 2 ;;
        --max-parallel) parallel="${2:?expected 1, 2 or 3}"; shift 2 ;;
        --eeg-predictions) eeg=(--eeg-predictions "${2:?expected legacy predictions CSV}"); shift 2 ;;
        -h|--help)
            printf '%s\n' 'Usage: bash scripts/run_multimodal_experiments.sh [--seeds 0|0-9] [--max-parallel 1|2|3] [--eeg-predictions FILE]'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ "$seeds" == 0 || "$seeds" == 0-9 ]] || { echo 'Expected --seeds 0 or 0-9' >&2; exit 2; }
[[ "$parallel" =~ ^[1-3]$ ]] || { echo 'Expected --max-parallel 1, 2 or 3' >&2; exit 2; }
if ((parallel > 1)); then
    printf '%s\n' 'RTX 4090 24 GB: concurrent peak VRAM has not been verified. OOM is possible; use --max-parallel 1 to limit this batch to one GPU worker.' >&2
fi
umask 077
mkdir -p outputs/multimodal_logs
logs=$(mktemp -d outputs/multimodal_logs/oof.XXXXXX)
printf 'Worker logs: %s/{M1,M2,M3}.log (follow with tail -f in tmux)\n' "$logs"

python_bin="${PYTHON:-python}"
if ! "$python_bin" scripts/compare_multimodal_oof.py baseline --seeds "$seeds" "${eeg[@]}" 2>&1 | tee "$logs/EEG-only.log"; then
    echo 'EEG-only validation or reproduction failed; GPU workers were not started.' >&2
    exit 1
fi
seed_list=(0)
[[ "$seeds" != 0-9 ]] || seed_list=(0 1 2 3 4 5 6 7 8 9)

# Use a separate pseudo-terminal for each worker so tqdm does not mix streams.
worker() {
    local model="$1" command terminal
    local -a arguments=("$BASH" -c '
        set -e
        if [[ -t 1 ]]; then stty cols 120 rows 30; fi
        model="$1"; shift
        for seed in "$@"; do
            printf "Running %s seed %s\n" "$model" "$seed"
            bash scripts/train_multimodal_folds.sh "$model" "$seed"
        done
    ' worker "$model" "${seed_list[@]}")
    if command -v script >/dev/null; then
        printf -v command '%q ' "${arguments[@]}"
        terminal="${TERM:-xterm-256color}"
        [[ "$terminal" != dumb ]] || terminal=xterm-256color
        TERM="$terminal" SHELL="$BASH" exec script -qefc "$command" /dev/null </dev/null
    else
        exec "${arguments[@]}"
    fi
}

models=(M1 M2 M3)
failed=()
pids=()
stop_workers() {
    local pid
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait || true
}
trap 'stop_workers; exit 130' INT
trap 'stop_workers; exit 143' TERM
for ((start=0; start<${#models[@]}; start+=parallel)); do
    group=("${models[@]:start:parallel}")
    pids=()
    for model in "${group[@]}"; do
        worker "$model" >"$logs/$model.log" 2>&1 &
        pids+=("$!")
        printf 'Started %s · %s/%s.log\n' "$model" "$logs" "$model"
    done
    for index in "${!group[@]}"; do
        model="${group[index]}"
        if wait "${pids[index]}"; then
            printf 'Completed %s\n' "$model"
        else
            status=$?
            failed+=("$model (exit $status)")
            printf 'Failed %s (exit %s) · %s/%s.log\n' "$model" "$status" "$logs" "$model" >&2
        fi
    done
    pids=()
done
if (("${#failed[@]}")); then
    printf 'Model failures: %s. OOF aggregation was skipped.\n' "${failed[*]}" >&2
    exit 1
fi
"$python_bin" scripts/compare_multimodal_oof.py evaluate --seeds "$seeds" "${eeg[@]}" 2>&1 | tee "$logs/evaluation.log"
