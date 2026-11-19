#!/usr/bin/env bash
# Usage: bash scripts/continue_multimodal_30.sh [--allow-fresh] [M1|M2|M3 [SEED [FOLD ...]]]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
fresh=()
if [[ "${1:-}" == --allow-fresh ]]; then
    fresh=(--allow-fresh)
    shift
fi
models=(M1 M2 M3)
if (($#)); then
    case "$1" in M1|M2|M3) models=("$1"); shift ;; *) exit 2 ;; esac
fi
seed="${1:-0}"
if (($#)); then shift; fi
if (($# == 0)); then set -- 1 2 3 4 5; fi
[[ "$seed" =~ ^[0-9]$ ]] || exit 2
for fold in "$@"; do [[ "$fold" =~ ^[1-5]$ ]] || exit 2; done
export MEDLEY_FIGURES_DIR=outputs/multimodal_figures_30epochs
for model in "${models[@]}"; do
    for fold in "$@"; do
        bash scripts/train_multimodal.sh "$model" "$seed" "$fold" --resume-or-start \
            --continue-from outputs/multimodal_runs --output-dir outputs/multimodal_runs_30epochs \
            --set training.max_epochs=30 --set training.early_stopping_patience=30 "${fresh[@]}"
    done
done
