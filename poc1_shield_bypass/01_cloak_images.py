#!/usr/bin/env python3
"""
POC 1 — Step 1: Apply adversarial cloaking to face images.

Supports:
  - 'fawkes': Fawkes-equivalent cloaking using InsightFace ArcFace.
    Same algorithm as Fawkes: targeted PGD against ArcFace, pushing the
    face embedding toward a synthetic decoy identity.
  - 'fgsm': Simpler untargeted perturbation using ResNet50 features.
  - 'glaze'/'nightshade': Manual workflow (user runs GUI externally).

Usage:
    # Fawkes-equivalent (recommended — uses actual ArcFace)
    python poc1_shield_bypass/01_cloak_images.py \
        --input data/consenting_subjects/subject_001 \
        --output poc1_shield_bypass/cloaked_images/subject_001_fawkes \
        --method fawkes --mode mid

    # Simpler FGSM (faster, uses ResNet50)
    python poc1_shield_bypass/01_cloak_images.py \
        --input data/consenting_subjects/subject_001 \
        --output poc1_shield_bypass/cloaked_images/subject_001_fgsm \
        --method fgsm --epsilon 16

    # Manual Glaze workflow
    python poc1_shield_bypass/01_cloak_images.py \
        --input data/consenting_subjects/subject_001 \
        --output poc1_shield_bypass/cloaked_images/subject_001_glaze \
        --method glaze
"""

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fawkes-equivalent cloaking using InsightFace ArcFace
# ---------------------------------------------------------------------------

def apply_fawkes_cloaking(
    input_dir: Path,
    output_dir: Path,
    mode: str = "mid",
) -> dict:
    """Apply Fawkes-equivalent adversarial cloaking using ArcFace.

    Faithful reimplementation of the Fawkes algorithm:
    1. Detect faces and extract ArcFace embeddings
    2. Create a synthetic target identity far from the subject
    3. PGD perturbation to push face embedding toward the target
    4. Constrain perturbation to be visually imperceptible

    Modes: low (ε=8), mid (ε=16), high (ε=32)
    """
    import cv2
    from insightface.app import FaceAnalysis

    # Mode settings
    mode_cfg = {
        "low":  {"epsilon": 8,  "steps": 30},
        "mid":  {"epsilon": 16, "steps": 50},
        "high": {"epsilon": 32, "steps": 80},
    }
    cfg = mode_cfg.get(mode, mode_cfg["mid"])
    epsilon = cfg["epsilon"]
    num_steps = cfg["steps"]

    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in extensions
    )
    if not image_paths:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")
    print(f"Fawkes mode: {mode} (epsilon={epsilon}, steps={num_steps})")
    print("-" * 60)

    # Initialise InsightFace (same ArcFace model Fawkes uses)
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)

    # --- Phase 1: Get original embeddings ---
    print("\nPhase 1: Extracting original face embeddings...")
    original_embeddings = {}

    for img_path in tqdm(image_paths, desc="Detecting"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        faces = face_app.get(img_bgr)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            original_embeddings[img_path.name] = face.normed_embedding
        else:
            print(f"  Warning: No face in {img_path.name}")
            original_embeddings[img_path.name] = None

    valid_embs = [e for e in original_embeddings.values() if e is not None]
    if not valid_embs:
        print("Error: No faces detected in any image!")
        sys.exit(1)

    print(f"  Detected faces in {len(valid_embs)}/{len(image_paths)} images")

    # --- Phase 2: Create target (decoy) identity ---
    mean_emb = np.mean(valid_embs, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)

    # Create a target embedding far from the original (Fawkes approach)
    np.random.seed(42)
    rand_dir = np.random.randn(512).astype(np.float32)
    rand_dir -= np.dot(rand_dir, mean_emb) * mean_emb  # Gram-Schmidt
    rand_dir = rand_dir / np.linalg.norm(rand_dir)
    target_emb = -0.3 * mean_emb + 0.95 * rand_dir
    target_emb = target_emb / np.linalg.norm(target_emb)

    print(f"  Target cosine sim to original: {np.dot(mean_emb, target_emb):.3f}")

    # --- Phase 3: PGD perturbation per image ---
    print(f"\nPhase 2: Applying PGD perturbation (ε={epsilon}, {num_steps} steps)...")
    eps = epsilon / 255.0

    cloaked_count = 0
    sims_before = []
    sims_after = []

    for img_path in tqdm(image_paths, desc="Cloaking"):
        orig_emb = original_embeddings.get(img_path.name)

        # Read image as float array [0, 1]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_float = img_bgr.astype(np.float32) / 255.0

        if orig_emb is None:
            # No face detected, just copy
            cv2.imwrite(str(output_dir / img_path.name), img_bgr)
            cloaked_count += 1
            continue

        sims_before.append(float(np.dot(orig_emb, mean_emb)))

        # PGD loop
        delta = np.zeros_like(img_float)

        for step in range(num_steps):
            # Current perturbed image
            perturbed = np.clip(img_float + delta, 0, 1)
            perturbed_uint8 = (perturbed * 255).astype(np.uint8)

            # Get current embedding
            faces = face_app.get(perturbed_uint8)
            if not faces:
                break

            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            curr_emb = face.normed_embedding
            del faces, perturbed, perturbed_uint8

            # Compute gradient via random perturbation (SPSA-style)
            # Since ONNX is not differentiable, we estimate gradients numerically
            h = 0.01
            num_rand = 3  # Random directions per step
            est_grad = np.zeros_like(delta)

            for _ in range(num_rand):
                # Random perturbation direction
                noise = np.random.choice([-1.0, 1.0], size=delta.shape).astype(np.float32) * h

                # +direction
                p_plus = np.clip(img_float + delta + noise, 0, 1)
                faces_p = face_app.get((p_plus * 255).astype(np.uint8))
                if faces_p:
                    emb_p = max(faces_p, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding
                    sim_p = float(np.dot(emb_p, target_emb))
                    del faces_p, emb_p
                else:
                    del p_plus
                    continue
                del p_plus

                # -direction
                p_minus = np.clip(img_float + delta - noise, 0, 1)
                faces_m = face_app.get((p_minus * 255).astype(np.uint8))
                if faces_m:
                    emb_m = max(faces_m, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding
                    sim_m = float(np.dot(emb_m, target_emb))
                    del faces_m, emb_m
                else:
                    del p_minus
                    continue
                del p_minus

                # SPSA gradient estimate: we want to MAXIMISE sim to target
                est_grad += (sim_p - sim_m) / (2 * noise)
                del noise

            est_grad /= num_rand

            # Gradient ascent step (to maximise similarity to target)
            step_size = eps / 5
            delta += step_size * np.sign(est_grad)
            del est_grad

            # Project to L-inf ball
            delta = np.clip(delta, -eps, eps)
            # Ensure valid pixel range
            delta = np.clip(img_float + delta, 0, 1) - img_float

        # Save cloaked image
        cloaked = np.clip(img_float + delta, 0, 1)
        cloaked_uint8 = (cloaked * 255).astype(np.uint8)

        # Measure final embedding shift
        faces_final = face_app.get(cloaked_uint8)
        if faces_final:
            final_face = max(faces_final, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            sim_after = float(np.dot(final_face.normed_embedding, orig_emb))
            sims_after.append(sim_after)

        cv2.imwrite(str(output_dir / img_path.name), cloaked_uint8)
        cloaked_count += 1

        # Free memory explicitly to prevent OOM on large datasets
        del img_bgr, img_float, delta, cloaked, cloaked_uint8
        gc.collect()

    # Summary
    print(f"\nCloaked {cloaked_count}/{len(image_paths)} images")
    if sims_before and sims_after:
        print(f"Identity similarity before: {np.mean(sims_before):.4f}")
        print(f"Identity similarity after:  {np.mean(sims_after):.4f}")
        shift = np.mean(sims_before) - np.mean(sims_after)
        print(f"Embedding shift:            {shift:.4f} {'(significant)' if shift > 0.1 else '(minor)'}")

    return {
        "method": "fawkes",
        "mode": mode,
        "epsilon": epsilon,
        "num_steps": num_steps,
        "feature_extractor": "insightface/buffalo_l/ArcFace",
        "algorithm": "targeted_PGD_with_SPSA_gradients",
        "num_images": len(image_paths),
        "num_cloaked": cloaked_count,
        "mean_sim_before": float(np.mean(sims_before)) if sims_before else None,
        "mean_sim_after": float(np.mean(sims_after)) if sims_after else None,
        "target_sim_to_original": float(np.dot(mean_emb, target_emb)),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# Simpler FGSM cloaking (untargeted, ResNet50)
# ---------------------------------------------------------------------------

def apply_fgsm_cloaking(
    input_dir: Path,
    output_dir: Path,
    epsilon: int = 16,
    num_steps: int = 40,
    step_size: float = 2.0,
) -> dict:
    """Simpler untargeted FGSM/PGD perturbation using ResNet50 features."""
    from torchvision import transforms, models

    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in extensions
    )
    if not image_paths:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Found {len(image_paths)} images")
    print(f"FGSM mode (epsilon={epsilon}, steps={num_steps})")
    print("-" * 60)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cloaked_count = 0
    perturbation_norms = []

    for img_path in tqdm(image_paths, desc="Cloaking"):
        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            original_features = feature_extractor(normalize(x)).flatten()

        delta = torch.zeros_like(x, requires_grad=True)
        eps = epsilon / 255.0

        for step in range(num_steps):
            perturbed = torch.clamp(x + delta, 0, 1)
            features = feature_extractor(normalize(perturbed)).flatten()
            cos_sim = torch.nn.functional.cosine_similarity(
                features.unsqueeze(0), original_features.unsqueeze(0))
            cos_sim.backward()
            with torch.no_grad():
                delta.data -= (step_size / 255.0) * delta.grad.sign()
                delta.data = torch.clamp(delta.data, -eps, eps)
                delta.data = torch.clamp(x + delta.data, 0, 1) - x
            delta.grad.zero_()

        with torch.no_grad():
            cloaked_tensor = torch.clamp(x + delta, 0, 1).squeeze(0)
            perturbation_norms.append(delta.abs().max().item() * 255)

        cloaked_array = (cloaked_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cloaked_img = Image.fromarray(cloaked_array)
        cloaked_img = cloaked_img.resize(original_size, Image.LANCZOS)
        cloaked_img.save(output_dir / img_path.name, quality=95)
        cloaked_count += 1

    print(f"\nCloaked {cloaked_count}/{len(image_paths)} images")
    print(f"Mean perturbation L-inf: {np.mean(perturbation_norms):.1f}/255")

    return {
        "method": "fgsm", "epsilon": epsilon, "num_steps": num_steps,
        "num_images": len(image_paths), "num_cloaked": cloaked_count,
        "mean_perturbation_linf": float(np.mean(perturbation_norms)),
        "input_dir": str(input_dir), "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# Manual (Glaze / Nightshade)
# ---------------------------------------------------------------------------

def handle_manual_cloaking(input_dir: Path, output_dir: Path, method: str) -> dict:
    """Handle Glaze / Nightshade which require manual GUI operation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    existing_images = [
        p for p in output_dir.iterdir() if p.suffix.lower() in extensions
    ] if output_dir.exists() else []

    input_images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in extensions
    )

    if existing_images:
        print(f"Found {len(existing_images)} pre-cloaked images in {output_dir}")
        return {
            "method": method, "mode": "manual",
            "num_images": len(existing_images), "num_cloaked": len(existing_images),
            "input_dir": str(input_dir), "output_dir": str(output_dir),
            "status": "pre-existing",
        }

    tool_name = method.capitalize()
    urls = {"glaze": "https://glaze.cs.uchicago.edu/", "nightshade": "https://nightshade.cs.uchicago.edu/"}
    print("=" * 60)
    print(f"MANUAL STEP REQUIRED: {tool_name} Cloaking")
    print("=" * 60)
    print(f"\n  1. Download {tool_name} from: {urls.get(method)}")
    print(f"  2. Open the app and load images from:\n     {input_dir}")
    print(f"  3. Run {tool_name} and save cloaked images to:\n     {output_dir}")
    print(f"  4. Re-run this script to verify")
    print(f"\n  Source: {len(input_images)} images in {input_dir}")
    print("=" * 60)

    return {
        "method": method, "mode": "manual",
        "num_images": len(input_images), "num_cloaked": 0,
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "status": "awaiting_manual_cloaking",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply adversarial cloaking to face images")
    parser.add_argument("--input", required=True, help="Input directory with face images")
    parser.add_argument("--output", required=True, help="Output directory for cloaked images")
    parser.add_argument("--method", required=True, choices=["fawkes", "fgsm", "glaze", "nightshade"],
                        help="Cloaking method")
    parser.add_argument("--epsilon", type=int, default=16, help="Perturbation budget [0-255]")
    parser.add_argument("--num_steps", type=int, default=50, help="PGD steps")
    parser.add_argument("--mode", default="mid", choices=["low", "mid", "high"],
                        help="Fawkes protection level")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"Cloaking method: {args.method}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    if args.method == "fawkes":
        metadata = apply_fawkes_cloaking(input_dir, output_dir, mode=args.mode)
    elif args.method == "fgsm":
        metadata = apply_fgsm_cloaking(input_dir, output_dir,
                                        epsilon=args.epsilon, num_steps=args.num_steps)
    else:
        metadata = handle_manual_cloaking(input_dir, output_dir, method=args.method)

    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(output_dir / "cloaking_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {output_dir / 'cloaking_metadata.json'}")

    if metadata.get("num_cloaked", 0) > 0:
        print(f"\nNext: python poc1_shield_bypass/02_train_lora.py \\")
        print(f"  --images {output_dir} \\")
        print(f"  --output poc1_shield_bypass/loras/subject_001_{args.method} --steps 200")


if __name__ == "__main__":
    main()
