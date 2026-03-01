"""
FID (Fréchet Inception Distance) computation for image quality evaluation.

Computes FID between a directory of generated images and a directory of
reference images using InceptionV3 features.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a directory."""

    def __init__(self, image_dir: str | Path, transform=None, max_images: int = 1000):
        self.image_dir = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in extensions
        )[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_inception_features(
    image_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 16,
    dims: int = 2048,
) -> np.ndarray:
    """Extract InceptionV3 features from images in a directory.

    Args:
        image_dir: Path to image directory.
        device: Torch device.
        batch_size: Batch size for feature extraction.
        dims: Feature dimensionality (2048 for pool3).

    Returns:
        Feature array of shape (N, dims).
    """
    from torchvision.models import inception_v3, Inception_V3_Weights

    # Load InceptionV3
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # Remove final FC layer → 2048-d features
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"InceptionV3 features ({image_dir})"):
            batch = batch.to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_fid(
    generated_dir: str | Path,
    reference_dir: str | Path,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> dict:
    """Compute Fréchet Inception Distance between two image directories.

    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^½)

    Lower FID = more similar distributions = better quality.

    Args:
        generated_dir: Directory of generated images.
        reference_dir: Directory of reference images.
        device: Torch device (auto-detected if None).
        batch_size: Batch size for InceptionV3 inference.

    Returns:
        Dict with FID score and metadata.
    """
    from scipy import linalg

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract features
    print("Extracting features from generated images...")
    gen_features = get_inception_features(generated_dir, device, batch_size)
    print(f"  Shape: {gen_features.shape}")

    print("Extracting features from reference images...")
    ref_features = get_inception_features(reference_dir, device, batch_size)
    print(f"  Shape: {ref_features.shape}")

    if gen_features.shape[0] < 2 or ref_features.shape[0] < 2:
        return {"error": "Need at least 2 images in each directory for FID"}

    # Compute statistics
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    mu_ref = np.mean(ref_features, axis=0)
    sigma_ref = np.cov(ref_features, rowvar=False)

    # FID formula
    diff = mu_gen - mu_ref
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_ref, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean))

    return {
        "fid_score": fid,
        "num_generated": gen_features.shape[0],
        "num_reference": ref_features.shape[0],
        "threshold": 35.0,
        "quality_acceptable": fid < 35.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute FID between two image directories")
    parser.add_argument("--generated", required=True, help="Generated images directory")
    parser.add_argument("--reference", required=True, help="Reference images directory")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = compute_fid(args.generated, args.reference)
    print(f"\nFID Score: {result['fid_score']:.2f}")
    print(f"Quality acceptable (< 35): {result['quality_acceptable']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
