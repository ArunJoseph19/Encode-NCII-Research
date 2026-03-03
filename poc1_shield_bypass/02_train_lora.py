#!/usr/bin/env python3
"""
POC 1 — Step 2: Train a LoRA adapter on face images (cloaked or uncloaked).

Supports Stable Diffusion XL (SDXL — default, ~6.5GB, best quality),
SD 1.5 (smaller, lower quality), and Flux.1-dev (GPU required).

Usage:
    # Local CPU (SDXL — good quality, no NSFW filter)
    python poc1_shield_bypass/02_train_lora.py \
        --images poc1_shield_bypass/cloaked_images/subject_001_fawkes \
        --output poc1_shield_bypass/loras/subject_001_sdxl \
        --steps 1500 --rank 4

    # GPU (Flux.1-dev — full experiment)
    python poc1_shield_bypass/02_train_lora.py \
        --images poc1_shield_bypass/cloaked_images/subject_001_fawkes \
        --output poc1_shield_bypass/loras/subject_001_flux \
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
# SDXL Training
# ---------------------------------------------------------------------------

def train_sdxl_lora(
    images_dir: str,
    output_dir: str,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    steps: int = 1500,
    rank: int = 4,
    learning_rate: float = 1e-4,
    resolution: int = 512,
    trigger_word: str = "ohwx person",
    seed: int = 42,
    log_every: int = 10,
):
    """Train LoRA on Stable Diffusion XL.

    SDXL produces much better face images than SD 1.5 and has no
    built-in NSFW safety checker (no more black images).
    """
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {base_model}")
    print(f"  Device: {device}, Dtype: {dtype}")

    # Load pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=dtype, variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )

    # Extract components
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    unet = pipe.unet.to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

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

    # Encode the caption once with BOTH text encoders (SDXL requirement)
    caption = dataset.caption

    # Text encoder 1
    text_input_1 = tokenizer(
        caption, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)

    # Text encoder 2
    text_input_2 = tokenizer_2(
        caption, padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        prompt_embeds_1 = text_encoder(text_input_1.input_ids, output_hidden_states=True)
        pooled_prompt_embeds_1 = prompt_embeds_1[0]
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]

        prompt_embeds_2 = text_encoder_2(text_input_2.input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

        # Concatenate embeddings from both encoders (SDXL standard)
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    # SDXL additional conditioning: time ids
    # (original_size, crops_coords_top_left, target_size)
    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=dtype, device=device,
    )

    # Training loop
    print(f"\nStarting training: {steps} steps (SDXL)")
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

            # SDXL UNet requires added_cond_kwargs
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            # Predict noise
            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
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
    print(f"  --lora {output_path} --output poc1_shield_bypass/results/subject_001_sdxl")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for face identity")
    parser.add_argument("--images", required=True, help="Training images directory")
    parser.add_argument("--output", required=True, help="Output directory for LoRA")
    parser.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Base model (default: SDXL)")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps")
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
        print("Use SDXL for local testing:")
        print(f"  python {sys.argv[0]} --images {args.images} --output {args.output} --steps 1500")
        if not torch.cuda.is_available():
            print("\nNo GPU detected. Aborting Flux training.")
            sys.exit(1)

    train_sdxl_lora(
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
