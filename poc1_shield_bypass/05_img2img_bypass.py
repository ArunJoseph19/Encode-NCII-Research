#!/usr/bin/env python3
"""
POC 1 — Experiment: img2img bypass of cloaked images.

Takes a single cloaked image and passes it through SDXL's img2img pipeline
with a modified prompt. No LoRA training required — this tests whether
even a basic image-to-image transformation strips the adversarial cloak.

Usage:
    python poc1_shield_bypass/05_img2img_bypass.py \
        --input poc1_shield_bypass/cloaked_images/george_w_bush_fawkes_high/face_000.jpg \
        --output poc1_shield_bypass/results/img2img_bypass \
        --prompt "a photo of this person wearing a cowboy hat"
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


def img2img_generate(input_path, output_dir, prompts, strength=0.6,
                     guidance_scale=7.5, num_inference_steps=30, seed=42):
    """Run img2img on a cloaked image with various prompts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the cloaked image
    source_img = Image.open(input_path).convert("RGB").resize((512, 512))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading SDXL img2img pipeline...")
    from diffusers import StableDiffusionXLImg2ImgPipeline

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    print(f"Source image: {input_path}")
    print(f"Strength: {strength} (0=exact copy, 1=fully reimagined)")
    print(f"Prompts: {len(prompts)}")
    print("-" * 60)

    # Save a copy of the source for comparison
    source_img.save(output_path / "source_cloaked.png")

    generator = torch.Generator(device=device)
    all_metadata = []

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] \"{prompt}\"")
        generator.manual_seed(seed + idx)

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=source_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
        
        image = result.images[0]
        filename = f"img2img_{idx:02d}_seed{seed + idx}.png"
        image.save(output_path / filename)
        print(f"  → Saved {filename}")

        all_metadata.append({
            "filename": filename,
            "prompt": prompt,
            "seed": seed + idx,
            "strength": strength,
        })

    # Save metadata
    meta = {
        "source_image": str(input_path),
        "method": "img2img (no LoRA, no training)",
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "images": all_metadata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path / "img2img_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {len(all_metadata)} images saved to {output_path}")
    print(f"\nNext: Run ArcFace to check if identity was preserved:")
    print(f"  python poc1_shield_bypass/04_arcface_similarity.py \\")
    print(f"    --generated {output_path} \\")
    print(f"    --reference data/consenting_subjects/george_w_bush \\")
    print(f"    --output {output_path / 'similarity_scores.json'} --skip_fid")


def main():
    parser = argparse.ArgumentParser(description="img2img bypass of cloaked images")
    parser.add_argument("--input", required=True, help="Path to a single cloaked image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prompts", nargs="*", default=[
        "a photo of this person wearing a cowboy hat, portrait",
        "a photo of this person wearing sunglasses, cool style",
        "an oil painting of this person in the style of van gogh",
    ])
    parser.add_argument("--strength", type=float, default=0.6,
                        help="Denoising strength (0=no change, 1=fully reimagined)")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    img2img_generate(
        input_path=args.input,
        output_dir=args.output,
        prompts=args.prompts,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
