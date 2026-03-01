#!/usr/bin/env python3
"""
Download face images for POC 1 experiments.

Uses Labeled Faces in the Wild (LFW) dataset via scikit-learn, which provides
a reliable download source with multiple images per person. Selects a single
identity with enough images (15-30) for LoRA fine-tuning.

Usage:
    python data/download_faces.py --output data/consenting_subjects/subject_001 --num_images 25
    python data/download_faces.py --person "George W Bush" --output data/consenting_subjects/subject_001
    python data/download_faces.py --list_people  # Show available people with 15+ images
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Please install Pillow: pip install Pillow")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Please install numpy: pip install numpy")
    sys.exit(1)

from tqdm import tqdm


def get_lfw_people(min_faces: int = 15):
    """Download LFW dataset and return people with enough images.

    Args:
        min_faces: Minimum number of face images per person.

    Returns:
        Dict mapping person name -> list of face image arrays.
    """
    from sklearn.datasets import fetch_lfw_people

    print("Downloading LFW dataset (first run may take a few minutes)...")
    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=1.0,  # Keep original resolution
        color=True,
    )

    # Group by person
    people: dict[str, list[np.ndarray]] = {}
    for img, label_idx in zip(lfw.images, lfw.target):
        name = lfw.target_names[label_idx]
        people.setdefault(name, []).append(img)

    return people


def list_available_people(min_faces: int = 15):
    """Print all people in LFW with enough images."""
    people = get_lfw_people(min_faces=min_faces)
    print(f"\nPeople with {min_faces}+ images in LFW:")
    print("-" * 50)
    for name, images in sorted(people.items(), key=lambda x: -len(x[1])):
        print(f"  {name:<30} {len(images)} images")
    print(f"\nTotal: {len(people)} people")


def save_face_images(
    images: list[np.ndarray],
    output_dir: Path,
    target_size: int = 512,
    max_images: int = 30,
):
    """Save face images to directory, upscaled to target resolution.

    Args:
        images: List of numpy image arrays (H, W, C), float [0, 1].
        output_dir: Directory to save images.
        target_size: Target resolution (images are upscaled for LoRA).
        max_images: Maximum images to save.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images = images[:max_images]
    saved = []

    for idx, img_array in enumerate(tqdm(images, desc="Saving images")):
        # Convert from float [0, 1] to uint8 [0, 255] if needed
        if img_array.dtype in (np.float32, np.float64):
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        img = Image.fromarray(img_array)

        # Upscale to target resolution (LFW images are 250×250 or 125×94)
        # Use LANCZOS for high-quality upscaling
        img = img.resize((target_size, target_size), Image.LANCZOS)

        filename = f"face_{idx:03d}.jpg"
        img.save(output_dir / filename, quality=95)
        saved.append(filename)

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Download face images for POC 1 experiments"
    )
    parser.add_argument(
        "--output", type=str, default="data/consenting_subjects/subject_001",
        help="Output directory for downloaded images"
    )
    parser.add_argument(
        "--person", type=str, default=None,
        help="Person name to download (default: auto-select best candidate)"
    )
    parser.add_argument(
        "--num_images", type=int, default=25,
        help="Number of images to download (default: 25)"
    )
    parser.add_argument(
        "--resize", type=int, default=512,
        help="Resize images to this resolution (default: 512)"
    )
    parser.add_argument(
        "--min_faces", type=int, default=15,
        help="Minimum faces per person (default: 15)"
    )
    parser.add_argument(
        "--list_people", action="store_true",
        help="List available people and exit"
    )
    args = parser.parse_args()

    if args.list_people:
        list_available_people(min_faces=args.min_faces)
        return

    output_dir = Path(args.output)

    # Download LFW
    people = get_lfw_people(min_faces=args.min_faces)
    print(f"Found {len(people)} people with {args.min_faces}+ images")

    # Select person
    if args.person:
        # Fuzzy match
        matches = [
            name for name in people
            if args.person.lower() in name.lower()
        ]
        if not matches:
            print(f"Error: No match for '{args.person}'")
            print("Use --list_people to see available names")
            sys.exit(1)
        person_name = matches[0]
        if len(matches) > 1:
            print(f"Multiple matches: {matches}. Using '{person_name}'.")
    else:
        # Auto-select: pick person closest to target num_images
        person_name = min(
            people.keys(),
            key=lambda k: abs(len(people[k]) - args.num_images)
        )

    images = people[person_name]
    print(f"\nSelected: {person_name} ({len(images)} images available)")

    # Save images
    saved = save_face_images(
        images, output_dir,
        target_size=args.resize,
        max_images=args.num_images,
    )

    print(f"\n{'=' * 60}")
    print(f"SUCCESS: {len(saved)} face images ready at {output_dir}/")
    print(f"Person: {person_name}")
    print(f"Resolution: {args.resize}×{args.resize}")
    print(f"{'=' * 60}")
    print(f"\nNext step: Apply cloaking")
    print(f"  python poc1_shield_bypass/01_cloak_images.py \\")
    print(f"    --input {output_dir} \\")
    print(f"    --output poc1_shield_bypass/cloaked_images/subject_001 \\")
    print(f"    --method fawkes")


if __name__ == "__main__":
    main()
