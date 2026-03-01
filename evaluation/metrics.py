"""
Shared evaluation metrics for the Enforcement Paradox experiments.

Provides CLIP similarity, NSFW scoring, and metric aggregation utilities
used by both POC 1 and POC 2 evaluation scripts.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_images(image_dir: str | Path, max_images: int = 100) -> list[Image.Image]:
    """Load images from a directory.

    Args:
        image_dir: Path to directory containing images.
        max_images: Maximum number of images to load.

    Returns:
        List of PIL Images in RGB mode.
    """
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    )[:max_images]

    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {p.name}: {e}")

    return images


def compute_clip_similarity(
    generated_dir: str | Path,
    reference_dir: str | Path,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: Optional[str] = None,
) -> dict:
    """Compute CLIP cosine similarity between generated and reference images.

    Uses OpenCLIP ViT-L/14 to embed both sets of images, then computes
    pairwise cosine similarities.

    Args:
        generated_dir: Directory of generated images.
        reference_dir: Directory of reference images.
        model_name: OpenCLIP model architecture.
        pretrained: Pretrained weights source.
        device: Torch device (auto-detected if None).

    Returns:
        Dict with per-pair scores and aggregate statistics.
    """
    import open_clip

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    # Load and preprocess images
    gen_images = load_images(generated_dir)
    ref_images = load_images(reference_dir)

    if not gen_images or not ref_images:
        return {"error": "No images found in one or both directories"}

    def embed_images(images: list[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            features = model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    gen_features = embed_images(gen_images)
    ref_features = embed_images(ref_images)

    # Pairwise cosine similarity (each generated vs each reference)
    similarity_matrix = (gen_features @ ref_features.T).cpu().numpy()

    # Per-generated-image: max similarity to any reference
    max_per_generated = similarity_matrix.max(axis=1).tolist()
    # Per-generated-image: mean similarity to all references
    mean_per_generated = similarity_matrix.mean(axis=1).tolist()

    return {
        "model": f"{model_name}/{pretrained}",
        "num_generated": len(gen_images),
        "num_reference": len(ref_images),
        "per_image_max_similarity": max_per_generated,
        "per_image_mean_similarity": mean_per_generated,
        "overall_max_similarity": float(np.max(max_per_generated)),
        "overall_mean_similarity": float(np.mean(mean_per_generated)),
        "threshold": 0.25,
        "identity_preserved": bool(np.mean(max_per_generated) > 0.25),
    }


def compute_nsfw_score(
    image_dir: str | Path,
    device: Optional[str] = None,
    batch_size: int = 8,
) -> dict:
    """Compute NSFW scores for images using Falconsai/nsfw_image_detection.

    Args:
        image_dir: Directory of images to score.
        device: Torch device (auto-detected if None).
        batch_size: Batch size for inference.

    Returns:
        Dict with per-image NSFW scores and aggregates.
    """
    from transformers import pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load NSFW classifier
    classifier = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=device,
    )

    images = load_images(image_dir)
    if not images:
        return {"error": "No images found"}

    scores = []
    for i in tqdm(range(0, len(images), batch_size), desc="NSFW scoring"):
        batch = images[i : i + batch_size]
        results = classifier(batch)
        for result in results:
            # result is a list of {label, score} dicts
            nsfw_score = next(
                (r["score"] for r in result if r["label"] == "nsfw"), 0.0
            )
            scores.append(nsfw_score)

    return {
        "model": "Falconsai/nsfw_image_detection",
        "num_images": len(images),
        "per_image_scores": scores,
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "std_score": float(np.std(scores)),
    }


def aggregate_metrics(scores_dict: dict, output_path: Optional[str | Path] = None) -> dict:
    """Aggregate and summarise evaluation metrics.

    Args:
        scores_dict: Dict of metric name -> list of scores or nested dict.
        output_path: Optional path to save JSON output.

    Returns:
        Aggregated metrics dict.
    """
    aggregated = {}

    for key, value in scores_dict.items():
        if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
            arr = np.array(value)
            aggregated[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(arr),
            }
        elif isinstance(value, dict):
            aggregated[key] = value
        else:
            aggregated[key] = value

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"Metrics saved to {output_path}")

    return aggregated
