#!/usr/bin/env python3
"""
POC 1 — Step 2: Train a LoRA adapter on face images (cloaked or uncloaked).

Supports both Stable Diffusion 1.5 (CPU-feasible, ~4GB) and Flux.1-dev
(GPU required, ~24GB). Defaults to SD 1.5 for local testing.

Usage:
    # Local CPU (SD 1.5 — fast enough for proof of concept)
    python poc1_shield_bypass/02_train_lora.py \
        --images poc1_shield_bypass/cloaked_images/subject_001_fgsm \
        --output poc1_shield_bypass/loras/subject_001_fgsm \
        --steps 100 --rank 4

    # GPU (Flux.1-dev — full experiment)
    python poc1_shield_bypass/02_train_lora.py \
        --images poc1_shield_bypass/cloaked_images/subject_001_fgsm \
        --output poc1_shield_bypass/loras/subject_001_fgsm \
        --base_model black-forest-labs/FLUX.1-dev \
        --steps 1500 --rank 16
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FaceLoRADataset(Dataset):
    """Dataset for LoRA fine-tuning on face images."""

    def __init__(
        self,
        image_dir: str | Path,
        trigger_word: str = "ohwx person",
        caption_template: str = "a photo of {trigger}",
        resolution: int = 512,
    ):
        self.image_dir = Path(image_dir)
        self.trigger_word = trigger_word
        self.caption = caption_template.format(trigger=trigger_word)
        self.resolution = resolution

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in extensions
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        print(f"Dataset: {len(self.image_paths)} images from {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(img)
        return {"pixel_values": pixel_values, "caption": self.caption}


# ---------------------------------------------------------------------------
# SD 1.5 Training
# ---------------------------------------------------------------------------

def train_sd15_lora(
    images_dir: str,
    output_dir: str,
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    steps: int = 200,
    rank: int = 4,
    learning_rate: float = 1e-4,
    resolution: int = 512,
    trigger_word: str = "ohwx person",
    seed: int = 42,
    log_every: int = 10,
):
    """Train LoRA on Stable Diffusion 1.5 (CPU-feasible).

    This is the local-testing path. SD 1.5 is ~4GB and can run on CPU
    (slowly, but feasibly). For the actual experiment, use Flux.1-dev on GPU.
    """
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {base_model}")
    print(f"  Device: {device}, Dtype: {dtype}")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=dtype,
    )

    # Extract components
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    unet = pipe.unet.to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Configure LoRA on UNet
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Dataset
    dataset = FaceLoRADataset(images_dir, trigger_word=trigger_word, resolution=resolution)

    # Optimizer
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

    # Encode the caption once (same for all images)
    text_input = tokenizer(
        dataset.caption, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0]

    # Training loop
    print(f"\nStarting training: {steps} steps")
    print("-" * 60)
    unet.train()
    losses = []
    progress = tqdm(total=steps, desc="Training LoRA")

    global_step = 0
    while global_step < steps:
        for idx in range(len(dataset)):
            if global_step >= steps:
                break

            sample = dataset[idx % len(dataset)]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device, dtype=dtype)

            # Encode to latent
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (1,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses.append(loss.item())

            if global_step % log_every == 0:
                avg = sum(losses[-log_every:]) / min(len(losses), log_every)
                progress.set_postfix(loss=f"{avg:.4f}")

            progress.update(1)

    progress.close()

    # Save
    print(f"\nSaving LoRA to {output_path}")
    unet.save_pretrained(output_path)

    metadata = {
        "base_model": base_model,
        "steps": steps,
        "rank": rank,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "trigger_word": trigger_word,
        "seed": seed,
        "num_training_images": len(dataset),
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "images_dir": images_dir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete! Final loss: {losses[-1]:.4f}" if losses else "Done")
    print(f"\nNext: python poc1_shield_bypass/03_generate_eval.py \\")
    print(f"  --lora {output_path} --output poc1_shield_bypass/results/subject_001_fgsm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for face identity")
    parser.add_argument("--images", required=True, help="Training images directory")
    parser.add_argument("--output", required=True, help="Output directory for LoRA")
    parser.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        help="Base model (default: SD 1.5)")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--trigger_word", default="ohwx person", help="Trigger word")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_8bit", action="store_true", help="8-bit quantisation (GPU)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    if "flux" in args.base_model.lower() or "FLUX" in args.base_model:
        print("Flux.1-dev requires a GPU with >=16GB VRAM.")
        print("Use SD 1.5 for local testing:")
        print(f"  python {sys.argv[0]} --images {args.images} --output {args.output} --steps 100")
        if not torch.cuda.is_available():
            print("\nNo GPU detected. Aborting Flux training.")
            sys.exit(1)

    train_sd15_lora(
        images_dir=args.images,
        output_dir=args.output,
        base_model=args.base_model,
        steps=args.steps,
        rank=args.rank,
        learning_rate=args.lr,
        resolution=args.resolution,
        trigger_word=args.trigger_word,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
