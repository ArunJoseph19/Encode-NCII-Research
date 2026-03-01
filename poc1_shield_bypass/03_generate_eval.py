#!/usr/bin/env python3
"""
POC 1 — Step 3: Generate images using the fine-tuned LoRA adapter.

Loads the base model with the trained LoRA adapter and generates images
from text prompts. Supports SD 1.5 (local) and Flux.1-dev (GPU).

Usage:
    python poc1_shield_bypass/03_generate_eval.py \
        --lora poc1_shield_bypass/loras/subject_001_fgsm \
        --output poc1_shield_bypass/results/subject_001_fgsm \
        --num_images 5
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm


DEFAULT_PROMPTS = [
    "a photo of {trigger}, professional headshot, studio lighting",
    "a portrait of {trigger}, natural outdoor lighting",
    "a close-up photo of {trigger}, sharp focus, high quality",
]


def generate_images(
    lora_path: str,
    output_dir: str,
    prompts: list[str] | None = None,
    base_model: str | None = None,
    num_images: int = 3,
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    width: int = 512,
    height: int = 512,
    trigger_word: str = "ohwx person",
):
    """Generate images using base model + LoRA adapter."""
    from diffusers import StableDiffusionPipeline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lora_dir = Path(lora_path)

    # Load training metadata to get the base model
    metadata_file = lora_dir / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            train_meta = json.load(f)
        if base_model is None:
            base_model = train_meta.get("base_model", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        trigger_word = train_meta.get("trigger_word", trigger_word)
    elif base_model is None:
        base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    # Resolve prompts
    prompt_list = prompts if prompts else DEFAULT_PROMPTS
    prompt_list = [p.replace("{trigger}", trigger_word) for p in prompt_list]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Load LoRA
    print(f"Loading LoRA: {lora_path}")
    from peft import PeftModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

    print(f"\nGenerating {num_images} images × {len(prompt_list)} prompts")
    print(f"  Trigger: '{trigger_word}'")
    print("-" * 60)

    all_metadata = []
    generator = torch.Generator(device=device)

    for prompt_idx, prompt in enumerate(prompt_list):
        print(f"\nPrompt {prompt_idx + 1}: \"{prompt}\"")

        for img_idx in tqdm(range(num_images), desc="Generating"):
            current_seed = seed + prompt_idx * num_images + img_idx
            generator.manual_seed(current_seed)

            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    width=width, height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )

            image = result.images[0]
            filename = f"gen_{prompt_idx:02d}_{img_idx:03d}_seed{current_seed}.png"
            image.save(output_path / filename)

            all_metadata.append({
                "filename": filename, "prompt": prompt,
                "seed": current_seed, "guidance_scale": guidance_scale,
            })

    # Save metadata
    gen_meta = {
        "base_model": base_model,
        "lora_path": lora_path,
        "trigger_word": trigger_word,
        "total_images": len(all_metadata),
        "prompts": prompt_list,
        "images": all_metadata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path / "generation_metadata.json", "w") as f:
        json.dump(gen_meta, f, indent=2)

    print(f"\nGenerated {len(all_metadata)} images → {output_path}")
    print(f"\nNext: python poc1_shield_bypass/04_arcface_similarity.py \\")
    print(f"  --generated {output_path} \\")
    print(f"  --reference data/consenting_subjects/subject_001 \\")
    print(f"  --output {output_path / 'similarity_scores.json'}")


def main():
    parser = argparse.ArgumentParser(description="Generate images with LoRA adapter")
    parser.add_argument("--lora", required=True, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--base_model", default=None, help="Base model (auto-detected from LoRA)")
    parser.add_argument("--num_images", type=int, default=3, help="Images per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--trigger_word", default="ohwx person")
    args = parser.parse_args()

    generate_images(
        lora_path=args.lora, output_dir=args.output,
        prompts=args.prompts, base_model=args.base_model,
        num_images=args.num_images, seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        width=args.width, height=args.height,
        trigger_word=args.trigger_word,
    )


if __name__ == "__main__":
    main()
