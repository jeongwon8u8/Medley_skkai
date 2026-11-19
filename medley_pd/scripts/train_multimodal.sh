#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model="${1:?usage: train_multimodal.sh M1|M2|M3 SEED FOLD [--resume] [--set KEY=VALUE]}"
seed="${2:?seed required}"
fold="${3:?fold required}"
shift 3
case "$model" in
    M1) config=clinical_only ;;
    M2) config=fusion_a ;;
    M3) config=fusion_b ;;
    *) echo "Expected M1, M2 or M3" >&2; exit 2 ;;
esac
[[ "$seed" =~ ^[0-9]+$ && "$fold" =~ ^[0-9]+$ ]] || exit 2
options=()
while (($#)); do
    case "$1" in
        --resume|--restart-incomplete|--resume-completed|--resume-or-start|--allow-fresh) options+=("$1"); shift ;;
        --set|--output-dir|--continue-from)
            options+=("$1" "${2:?option requires a value}"); shift 2 ;;
        *) echo "Unsupported training option: $1" >&2; exit 2 ;;
    esac
done
umask 077
handoff=$(mktemp -d)
trap 'rm -f "$handoff/result.json"; rmdir "$handoff"' EXIT
cmd=("${PYTHON:-python}" -u -m multimodal.experiment_runner
     --config "configs/multimodal/$config.json" --model "$model" --seed "$seed" --fold "$fold"
     --set "llm.model_id=${MEDLEY_LLM_PATH:-/workspace/models/Qwen3-4B}"
     --set "llm.revision=${MEDLEY_LLM_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"
     --set "llm.device=${MEDLEY_LLM_DEVICE:-cuda:0}"
     --set 'clinical_text.sex_code_mapping={"0":"female","1":"male"}'
     --execute --max-runs 1 --result-file "$handoff/result.json" "${options[@]}")
mkdir -p outputs/multimodal_logs
log=$(mktemp "outputs/multimodal_logs/${model}_${seed}_${fold}.XXXXXX.log")
printf 'Log %s\n' "$log" >&2
if [[ -t 1 && -t 2 ]] && command -v script >/dev/null; then
    printf -v command '%q ' "${cmd[@]}"
    SHELL="$BASH" script -qefc "$command" "$log"
else
    "${cmd[@]}" 2>&1 | tee "$log"
fi
if ! "${PYTHON:-python}" scripts/export_multimodal.py figures "$handoff/result.json" \
        --output "${MEDLEY_FIGURES_DIR:-outputs/multimodal_figures}"; then
    printf 'Training completed, but automatic figure generation failed. See %s for the execution directory.\n' "$log" >&2
    exit 1
fi
