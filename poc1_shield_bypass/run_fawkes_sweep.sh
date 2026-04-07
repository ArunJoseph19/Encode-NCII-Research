#!/usr/bin/env bash
# run_fawkes_sweep.sh — Fawkes epsilon sweep on george_w_bush.
#
# Runs the full pipeline sequentially for three epsilon conditions:
#   ε=8  (--mode low)  → george_w_bush_fawkes_eps8
#   ε=16 (--mode mid)  → george_w_bush_fawkes_eps16
#   ε=32 (--mode high) → george_w_bush_fawkes_eps32
#
# Each condition:
#   Step 1 — Cloak 100 images  (--method fawkes --max_images 100)
#   Step 2 — Train SDXL LoRA   (--steps 500 --rank 4)
#   Step 3 — Generate 30 images (10 per prompt × 3 default prompts)
#   Step 4 — ArcFace evaluation vs. original images
#
# Usage:
#   bash poc1_shield_bypass/run_fawkes_sweep.sh

set -euo pipefail

SUBJECT="george_w_bush"
INPUT="data/consenting_subjects/${SUBJECT}"

# ---------------------------------------------------------------------------
# run_condition <epsilon_int> <fawkes_mode>
# ---------------------------------------------------------------------------
run_condition() {
    local EPS="$1"
    local MODE="$2"

    local TAG="${SUBJECT}_fawkes_eps${EPS}"
    local CLOAKED="poc1_shield_bypass/cloaked_images/${TAG}"
    local LORA="poc1_shield_bypass/loras/${TAG}_sdxl_500_r4"
    local RESULTS="poc1_shield_bypass/results/${TAG}"

    echo ""
    echo "============================================================"
    echo "POC 1 — Fawkes ${MODE} (ε=${EPS}/255)"
    echo "Subject : ${SUBJECT}"
    echo "Input   : ${INPUT}"
    echo "Output  : ${RESULTS}"
    echo "Config  : SDXL, steps=500, rank=4"
    echo "============================================================"

    echo ""
    echo ">>> Step 1: Cloak 100 images (--mode ${MODE})"
    python poc1_shield_bypass/01_cloak_images.py \
        --input      "${INPUT}" \
        --output     "${CLOAKED}" \
        --method     fawkes \
        --mode       "${MODE}" \
        --max_images 100

    echo ""
    echo ">>> Step 2: Train LoRA (steps=500, rank=4, SDXL)"
    python poc1_shield_bypass/02_train_lora.py \
        --images "${CLOAKED}" \
        --output "${LORA}" \
        --steps  500 \
        --rank   4

    echo ""
    echo ">>> Step 3: Generate images (30 total)"
    python poc1_shield_bypass/03_generate_eval.py \
        --lora       "${LORA}" \
        --output     "${RESULTS}" \
        --num_images 10

    echo ""
    echo "Condition complete. Generated images saved to: ${RESULTS}"
    echo "Run 04_arcface_similarity.py on Mac to evaluate."
}

# ---------------------------------------------------------------------------
# Run three conditions sequentially
# ---------------------------------------------------------------------------

echo "============================================================"
echo "POC 1 — Fawkes Epsilon Sweep"
echo "Subject    : ${SUBJECT}"
echo "Conditions : ε=8 (low) → ε=16 (mid) → ε=32 (high)"
echo "Images     : 100 per condition"
echo "============================================================"

run_condition 8  low
run_condition 16 mid
run_condition 32 high

echo ""
echo "============================================================"
echo "Sweep complete. Generated images saved to:"
echo "  poc1_shield_bypass/results/${SUBJECT}_fawkes_eps8/"
echo "  poc1_shield_bypass/results/${SUBJECT}_fawkes_eps16/"
echo "  poc1_shield_bypass/results/${SUBJECT}_fawkes_eps32/"
echo ""
echo "Run 04_arcface_similarity.py on Mac to evaluate."
echo "============================================================"
