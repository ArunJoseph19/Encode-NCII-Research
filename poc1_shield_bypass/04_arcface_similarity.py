#!/usr/bin/env python3
"""
POC 1 — Step 4: Evaluate identity similarity and image quality.

Computes ArcFace identity similarity, CLIP similarity, and FID between
generated images and the original reference images. This is the key
measurement: if ArcFace similarity > 0.45, the cloaking has been bypassed.

Usage:
    python poc1_shield_bypass/04_arcface_similarity.py \
        --generated poc1_shield_bypass/results/subject_001_glaze \
        --reference data/consenting_subjects/subject_001 \
        --output poc1_shield_bypass/results/subject_001_glaze/similarity_scores.json

    # Compare all conditions
    python poc1_shield_bypass/04_arcface_similarity.py \
        --generated poc1_shield_bypass/results/subject_001_baseline \
                    poc1_shield_bypass/results/subject_001_glaze \
                    poc1_shield_bypass/results/subject_001_fawkes \
        --reference data/consenting_subjects/subject_001 \
        --output poc1_shield_bypass/results/comparison.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def evaluate_single(
    generated_dir: str,
    reference_dir: str,
    output_path: str | None = None,
    skip_fid: bool = False,
) -> dict:
    """Evaluate a single generated-vs-reference comparison.

    Args:
        generated_dir: Directory of generated images.
        reference_dir: Directory of reference (original) images.
        output_path: Optional path to save JSON results.
        skip_fid: Skip FID computation (faster).

    Returns:
        Dict with all evaluation metrics.
    """
    # Import here to allow the script to show --help without GPU
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.arcface_wrapper import ArcFaceWrapper
    from evaluation.metrics import compute_clip_similarity
    from evaluation.fid_score import compute_fid

    results = {
        "generated_dir": generated_dir,
        "reference_dir": reference_dir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 1. ArcFace identity similarity (primary metric)
    print("=" * 60)
    print("ArcFace Identity Similarity")
    print("=" * 60)
    try:
        arcface = ArcFaceWrapper()
        arcface_results = arcface.batch_similarity(generated_dir, reference_dir)
        results["arcface"] = arcface_results

        agg = arcface_results.get("aggregate", {})
        print(f"  Mean max similarity: {agg.get('mean_max_similarity', 'N/A'):.4f}")
        print(f"  Threshold: > 0.45 = identity preserved")
        print(f"  Result: {'IDENTITY PRESERVED ✓' if arcface_results.get('identity_preserved') else 'IDENTITY NOT PRESERVED ✗'}")
    except Exception as e:
        print(f"  Error: {e}")
        results["arcface"] = {"error": str(e)}

    # 2. CLIP similarity
    print(f"\n{'=' * 60}")
    print("CLIP Similarity")
    print("=" * 60)
    try:
        clip_results = compute_clip_similarity(generated_dir, reference_dir)
        results["clip"] = clip_results

        print(f"  Overall mean similarity: {clip_results.get('overall_mean_similarity', 'N/A'):.4f}")
        print(f"  Threshold: > 0.25 = identity preserved")
        print(f"  Result: {'IDENTITY PRESERVED ✓' if clip_results.get('identity_preserved') else 'IDENTITY NOT PRESERVED ✗'}")
    except Exception as e:
        print(f"  Error: {e}")
        results["clip"] = {"error": str(e)}

    # 3. FID score
    if not skip_fid:
        print(f"\n{'=' * 60}")
        print("FID Score")
        print("=" * 60)
        try:
            fid_results = compute_fid(generated_dir, reference_dir)
            results["fid"] = fid_results

            print(f"  FID Score: {fid_results.get('fid_score', 'N/A'):.2f}")
            print(f"  Threshold: < 35 = high quality")
            print(f"  Result: {'HIGH QUALITY ✓' if fid_results.get('quality_acceptable') else 'LOW QUALITY ✗'}")
        except Exception as e:
            print(f"  Error: {e}")
            results["fid"] = {"error": str(e)}
    else:
        print("\nFID computation skipped (--skip_fid)")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    arcface_preserved = results.get("arcface", {}).get("identity_preserved", False)
    clip_preserved = results.get("clip", {}).get("identity_preserved", False)
    fid_ok = results.get("fid", {}).get("quality_acceptable", None)

    results["summary"] = {
        "arcface_identity_preserved": arcface_preserved,
        "clip_identity_preserved": clip_preserved,
        "fid_quality_acceptable": fid_ok,
        "cloaking_bypassed": arcface_preserved,  # Primary conclusion
    }

    if arcface_preserved:
        print("  ⚠️  CLOAKING BYPASSED: Identity was preserved despite cloaking")
        print("     The adversarial perturbations did not prevent LoRA from")
        print("     learning the subject's identity.")
    else:
        print("  ✓  Cloaking held: Identity was NOT preserved")
        print("     The adversarial perturbations successfully prevented identity")
        print("     transfer through LoRA fine-tuning.")

    print(f"\n  ArcFace identity: {'✓ preserved' if arcface_preserved else '✗ not preserved'}")
    print(f"  CLIP similarity:  {'✓ preserved' if clip_preserved else '✗ not preserved'}")
    if fid_ok is not None:
        print(f"  Image quality:    {'✓ high' if fid_ok else '✗ low'}")

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_file}")

    return results


def evaluate_comparison(
    generated_dirs: list[str],
    reference_dir: str,
    output_path: str,
    skip_fid: bool = False,
) -> dict:
    """Evaluate and compare multiple conditions.

    Args:
        generated_dirs: List of generated image directories.
        reference_dir: Reference images directory.
        output_path: Path to save comparison JSON.
        skip_fid: Skip FID computation.

    Returns:
        Comparison dict.
    """
    comparison = {
        "reference_dir": reference_dir,
        "conditions": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for gen_dir in generated_dirs:
        condition_name = Path(gen_dir).name
        print(f"\n{'#' * 60}")
        print(f"# Evaluating: {condition_name}")
        print(f"{'#' * 60}\n")

        results = evaluate_single(gen_dir, reference_dir, skip_fid=skip_fid)
        comparison["conditions"][condition_name] = results

    # Comparative summary
    print(f"\n{'#' * 60}")
    print("# COMPARATIVE SUMMARY")
    print(f"{'#' * 60}")
    print(f"\n{'Condition':<35} {'ArcFace':>10} {'CLIP':>10} {'Bypassed?':>12}")
    print("-" * 70)

    for name, result in comparison["conditions"].items():
        arcface_sim = result.get("arcface", {}).get("aggregate", {}).get("mean_max_similarity", float("nan"))
        clip_sim = result.get("clip", {}).get("overall_mean_similarity", float("nan"))
        bypassed = result.get("summary", {}).get("cloaking_bypassed", False)

        print(f"{name:<35} {arcface_sim:>10.4f} {clip_sim:>10.4f} {'YES ⚠️' if bypassed else 'NO ✓':>12}")

    # Save comparison
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved to: {output_file}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate identity similarity between generated and reference images"
    )
    parser.add_argument(
        "--generated", nargs="+", required=True,
        help="Directory/directories of generated images"
    )
    parser.add_argument(
        "--reference", required=True,
        help="Directory of reference (original) images"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for results"
    )
    parser.add_argument(
        "--skip_fid", action="store_true",
        help="Skip FID computation (faster evaluation)"
    )
    args = parser.parse_args()

    if len(args.generated) == 1:
        evaluate_single(
            args.generated[0], args.reference,
            output_path=args.output,
            skip_fid=args.skip_fid,
        )
    else:
        evaluate_comparison(
            args.generated, args.reference,
            output_path=args.output,
            skip_fid=args.skip_fid,
        )


if __name__ == "__main__":
    main()
