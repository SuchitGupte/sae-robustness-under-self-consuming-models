#!/usr/bin/env bash
# Train SAEs on all self-consuming model checkpoints (gen_000..gen_010) for one layer.
#
# Usage:
#   bash run.sh --layer 0                          # layer 0, all gens
#   bash run.sh --layer 5 --device 1               # specific GPU
#   bash run.sh --layer 0 --no_wandb               # disable W&B
#   bash run.sh --layer 0 --training_tokens 5000000  # smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pull --layer and --device out; remaining args pass through to train.py
LAYER=""
DEVICE="0"
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layer)  LAYER="$2";  shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *)        PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "${LAYER}" ]]; then
    echo "Error: --layer is required" >&2
    echo "Usage: bash run.sh --layer <N> [--device <ID>] [extra args...]" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${DEVICE}"

GENERATIONS=(gen_000 gen_001 gen_002 gen_003 gen_004 gen_005 gen_006 gen_007 gen_008 gen_009 gen_010)

for GEN in "${GENERATIONS[@]}"; do
    echo "======================================================"
    echo "  Generation : ${GEN}  |  Layer : ${LAYER}"
    echo "======================================================"

    python "${SCRIPT_DIR}/train.py" \
        --gen_label "${GEN}" \
        --layer     "${LAYER}" \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

    echo "  Done with ${GEN} layer ${LAYER}."
    echo
done

echo "All generations complete for layer ${LAYER}."
