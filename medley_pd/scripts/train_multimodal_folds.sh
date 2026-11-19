#!/usr/bin/env bash
# Usage: bash scripts/train_multimodal_folds.sh M1|M2|M3 SEED [FOLD ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model="${1:?model required (M1, M2 or M3)}"
seed="${2:?seed required}"
shift 2
case "$model" in M1|M2|M3) ;; *) exit 2 ;; esac
[[ "$seed" =~ ^[0-9]$ ]] || exit 2
if (($# == 0)); then
    set -- 1 2 3 4 5
fi
for fold in "$@"; do
    [[ "$fold" =~ ^[1-5]$ ]] || exit 2
done
for fold in "$@"; do
    if bash scripts/test_multimodal.sh "$model" "$seed" "$fold" --missing-ok; then
        printf 'Skipping verified completion for %s seed %s fold %s\n' "$model" "$seed" "$fold"
    else
        status=$?
        if ((status != 3)); then
            exit "$status"
        fi
        bash scripts/train_multimodal.sh "$model" "$seed" "$fold" --resume-or-start
    fi
done
