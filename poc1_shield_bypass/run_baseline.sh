#!/usr/bin/env bash
# run_baseline.sh — Full pipeline for george_w_bush with no cloaking (control condition).
#
# Mirrors the Fawkes High run (steps=500, rank=4, SDXL) exactly, except
# Step 1 copies images unchanged instead of applying adversarial perturbation.
#
# Output: poc1_shield_bypass/results/george_w_bush_baseline/
#
# Usage:
#   bash poc1_shield_bypass/run_baseline.sh

set -euo pipefail

SUBJECT="george_w_bush"
INPUT="data/consenting_subjects/${SUBJECT}"
CLOAKED="poc1_shield_bypass/cloaked_images/${SUBJECT}_baseline"
LORA="poc1_shield_bypass/loras/${SUBJECT}_baseline_sdxl_500_r4"
RESULTS="poc1_shield_bypass/results/${SUBJECT}_baseline"

echo "============================================================"
echo "POC 1 — Baseline pipeline (no cloaking)"
echo "Subject : ${SUBJECT}"
echo "Input   : ${INPUT}"
echo "Output  : ${RESULTS}"
echo "Config  : SDXL, steps=500, rank=4"
echo "============================================================"

# Step 1: Copy images unchanged (baseline — no perturbation)
echo ""
echo ">>> Step 1: Baseline copy (no cloaking)"
python poc1_shield_bypass/01_cloak_images.py \
    --input  "${INPUT}" \
    --output "${CLOAKED}" \
    --method baseline

# Step 2: Train LoRA on the unmodified images
echo ""
echo ">>> Step 2: Train LoRA (steps=500, rank=4, SDXL)"
python poc1_shield_bypass/02_train_lora.py \
    --images  "${CLOAKED}" \
    --output  "${LORA}" \
    --steps   500 \
    --rank    4

# Step 3: Generate evaluation images
echo ""
echo ">>> Step 3: Generate images"
python poc1_shield_bypass/03_generate_eval.py \
    --lora       "${LORA}" \
    --output     "${RESULTS}" \
    --num_images 10

# Step 4: Evaluate ArcFace identity similarity
echo ""
echo ">>> Step 4: ArcFace similarity vs. original images"
python poc1_shield_bypass/04_arcface_similarity.py \
    --generated "${RESULTS}" \
    --reference "${INPUT}" \
    --output    "${RESULTS}/similarity_scores.json" \
    --skip_fid

echo ""
echo "============================================================"
echo "Baseline pipeline complete."
echo "Results: ${RESULTS}/similarity_scores.json"
echo "============================================================"
