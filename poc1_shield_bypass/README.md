# POC 1: The Broken Shield

## Goal

Prove that adversarial cloaking tools (Glaze, Fawkes, Nightshade) **fail** against LoRA fine-tuning on Flux.1-dev. If identity is preserved in generated images despite cloaking, the cloaking is ineffective — making it a "policy mirage" to recommend these tools to potential victims.

## How the Bypass Works (The Technical Mechanism)

Contrary to popular intuition, we **do not** feed a cloaked image into the model and ask it to generate a new photo. The bypass happens in two completely separate phases:

### Phase 1: Training (The LoRA Fine-Tuning)
We feed the cloaked images into the AI training process along with a trigger word (e.g., `ohwx person`). The AI spends hundreds of gradient descent steps analyzing the dataset to learn the mathematical concept of an `ohwx person`.

*This is where the cloak fundamentally fails.* The cloaking noise is an adversarial perturbation optimized for each specific image. Across 15-25 different photos, the underlying facial structure is consistent, but the adversarial noise is statistically inconsistent. Because the AI is learning the *distribution* of the face across the whole dataset, it learns the consistent strong signal (the true face) and ignores/averages out the inconsistent weak signal (the cloaking noise). 

It compiles this knowledge into a lightweight weights file called a **LoRA adapter**.

### Phase 2: Generation (Creating the new image)
The original cloaked images are discarded. They are never shown to the AI again.

We load the base generative model, attach our new LoRA adapter, and provide a simple text prompt:
`"A portrait photo of ohwx person, wearing a suit, natural lighting."`

Zero input images are provided. The AI reaches into the statistical concept of `ohwx person` it learned during Phase 1 and generates a brand new, 100% synthetic photo from scratch. Because the training phase successfully bypassed the cloak to learn the true face, the freshly generated synthetic image looks exactly like the real person.

## Pipeline

```
Original images ──→ Cloak (Glaze/Fawkes) ──→ Train LoRA ──→ Generate ──→ Evaluate
     │                                            │                          │
     └──── Train LoRA (baseline, no cloak) ───────┘              ArcFace > 0.45?
                                                                 = bypassed
```

## Quick Start

### 1. Download face dataset

```bash
python data/download_faces.py \
    --output data/consenting_subjects/subject_001 \
    --num_images 25
```

### 2. Apply cloaking

```bash
# Automated (Fawkes)
python poc1_shield_bypass/01_cloak_images.py \
    --input data/consenting_subjects/subject_001 \
    --output poc1_shield_bypass/cloaked_images/subject_001_fawkes \
    --method fawkes --mode mid

# Manual (Glaze) — follow the printed instructions
python poc1_shield_bypass/01_cloak_images.py \
    --input data/consenting_subjects/subject_001 \
    --output poc1_shield_bypass/cloaked_images/subject_001_glaze \
    --method glaze
```

### 3. Train LoRA (baseline + cloaked)

```bash
# Baseline (uncloaked — the control)
python poc1_shield_bypass/02_train_lora.py \
    --images data/consenting_subjects/subject_001 \
    --output poc1_shield_bypass/loras/subject_001_baseline \
    --steps 1500 --rank 16

# Cloaked (the attack)
python poc1_shield_bypass/02_train_lora.py \
    --images poc1_shield_bypass/cloaked_images/subject_001_fawkes \
    --output poc1_shield_bypass/loras/subject_001_fawkes \
    --steps 1500 --rank 16

# Low VRAM (16GB)
python poc1_shield_bypass/02_train_lora.py \
    --images poc1_shield_bypass/cloaked_images/subject_001_fawkes \
    --output poc1_shield_bypass/loras/subject_001_fawkes \
    --steps 1500 --rank 16 --use_8bit
```

### 4. Generate images

```bash
python poc1_shield_bypass/03_generate_eval.py \
    --lora poc1_shield_bypass/loras/subject_001_fawkes \
    --output poc1_shield_bypass/results/subject_001_fawkes \
    --num_images 10
```

### 5. Evaluate

```bash
# Single condition
python poc1_shield_bypass/04_arcface_similarity.py \
    --generated poc1_shield_bypass/results/subject_001_fawkes \
    --reference data/consenting_subjects/subject_001 \
    --output poc1_shield_bypass/results/subject_001_fawkes/scores.json

# Compare all conditions
python poc1_shield_bypass/04_arcface_similarity.py \
    --generated poc1_shield_bypass/results/subject_001_baseline \
               poc1_shield_bypass/results/subject_001_fawkes \
               poc1_shield_bypass/results/subject_001_glaze \
    --reference data/consenting_subjects/subject_001 \
    --output poc1_shield_bypass/results/comparison.json
```

## Success Criteria

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| ArcFace cosine similarity | > 0.45 | Identity preserved (cloaking bypassed) |
| CLIP similarity | > 0.25 | Visual similarity preserved |
| FID score | < 35 | Generated images are high quality |

## Expected Results

If cloaking **fails** (our hypothesis), we expect:

- **Baseline (uncloaked):** ArcFace > 0.55 (strong identity match)
- **Fawkes-cloaked:** ArcFace > 0.45 (identity still preserved)
- **Glaze-cloaked:** ArcFace > 0.45 (identity still preserved)

This would prove that recommending cloaking tools to victims provides false reassurance.
