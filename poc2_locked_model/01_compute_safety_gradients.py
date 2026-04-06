#!/usr/bin/env python3
"""
POC 2 — Step 1: Compute safety gradient subspace from NCII-adjacent prompts.

For each safety prompt, computes the gradient of the denoising loss with
respect to the model's attention parameters. Stacking these vectors and
taking the top-k right singular vectors (SVD) produces a "safety subspace"
S — the directions in weight-space that encode NCII-related generation
capability. Downstream steps project fine-tuning gradients away from S so
the safety capability cannot be removed.

Supports:
  - black-forest-labs/FLUX.1-dev (default, requires 40GB+ VRAM)
  - stabilityai/stable-diffusion-xl-base-1.0 (T4 / 16GB VRAM)

Usage:
    # SDXL — runs on T4 (16GB)
    python poc2_locked_model/01_compute_safety_gradients.py \\
        --model stabilityai/stable-diffusion-xl-base-1.0 \\
        --safety_prompts data/safety_prompts/ncii_adjacent.txt \\
        --output poc2_locked_model/safety_gradients/sdxl_safety.pt \\
        --top_k 50

    # Flux.1-dev — requires A100 / H100
    python poc2_locked_model/01_compute_safety_gradients.py \\
        --safety_prompts data/safety_prompts/ncii_adjacent.txt \\
        --output poc2_locked_model/safety_gradients/flux_dev_safety.pt \\
        --top_k 50
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Must be set before any CUDA context is initialised.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm


SUPPORTED_MODELS = {
    "black-forest-labs/FLUX.1-dev": "flux",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
}

# Attention projection modules targeted by LoRA — safety gradients must cover
# exactly the same parameter set that an attacker would fine-tune.
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_safety_prompts(path: Path) -> list[str]:
    """Load and filter prompts from a text file (skip blank lines and comments)."""
    lines = path.read_text().splitlines()
    prompts = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    if not prompts:
        print(f"Error: No prompts found in {path}")
        sys.exit(1)
    print(f"Loaded {len(prompts)} safety prompts from {path}")
    return prompts


# ---------------------------------------------------------------------------
# Gradient collection helpers
# ---------------------------------------------------------------------------

def _collect_target_params(module, target_modules: list[str]):
    """Return (name, param) pairs for all attention projection layers."""
    names, params = [], []
    for name, param in module.named_parameters():
        if any(t in name for t in target_modules):
            names.append(name)
            params.append(param)
    return names, params


def _grad_vector(params: list[torch.Tensor]) -> torch.Tensor:
    """Flatten and concatenate per-parameter gradients into one vector."""
    vecs = []
    for p in params:
        if p.grad is not None:
            vecs.append(p.grad.flatten())
        else:
            vecs.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    return torch.cat(vecs)


def _zero_grads(params: list[torch.Tensor]) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


# ---------------------------------------------------------------------------
# SDXL gradient computation
# ---------------------------------------------------------------------------

def compute_gradients_sdxl(
    model_id: str,
    prompts: list[str],
    top_k: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    num_timesteps: int,
) -> dict:
    """Compute safety gradient matrix for SDXL (UNet attention layers)."""
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler

    print(f"\nLoading SDXL: {model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    del pipe  # free pipeline wrapper memory

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(True)

    # Recompute activations during backward instead of storing them — trades
    # compute for memory, essential on T4 where the UNet activations alone
    # exceed available VRAM at full resolution.
    unet.enable_gradient_checkpointing()

    param_names, params = _collect_target_params(unet, LORA_TARGET_MODULES)
    print(f"Tracking gradients for {len(params)} attention parameters "
          f"({sum(p.numel() for p in params):,} values)")

    torch.manual_seed(seed)
    # Use a 32×32 latent spatial size (equivalent to a 256×256 image) rather
    # than the full 64×64 (512×512).  Attention memory scales quadratically
    # with sequence length, so halving each spatial dimension cuts peak
    # activation memory by ~4× — enough to fit on a T4 with gradients enabled.
    latent_channels = unet.config.in_channels
    latent_h = latent_w = 32

    # time_ids encode (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
    # in pixel space — keep at 512 so SDXL's aesthetic conditioning is valid
    # even though we sample smaller latents.
    add_time_ids = torch.tensor(
        [[512, 512, 0, 0, 512, 512]],
        dtype=dtype, device=device,
    )

    all_grad_vectors = []

    for prompt in tqdm(prompts, desc="Safety gradients (SDXL)"):
        # Free any allocator-cached blocks from the previous iteration before
        # building the new computation graph.
        if device == "cuda":
            torch.cuda.empty_cache()
        _zero_grads(params)

        # Encode with both SDXL text encoders
        tok1 = tokenizer(
            prompt, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        tok2 = tokenizer_2(
            prompt, padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out1 = text_encoder(tok1.input_ids, output_hidden_states=True)
            emb1 = out1.hidden_states[-2]

            out2 = text_encoder_2(tok2.input_ids, output_hidden_states=True)
            pooled2 = out2[0]
            emb2 = out2.hidden_states[-2]

            prompt_embeds = torch.cat([emb1, emb2], dim=-1)

        added_cond = {"text_embeds": pooled2, "time_ids": add_time_ids}

        # Accumulate denoising loss gradient over several timesteps
        for _ in range(num_timesteps):
            latents = torch.randn(
                1, latent_channels, latent_h, latent_w, dtype=dtype, device=device,
            )
            noise = torch.randn_like(latents)
            t = torch.randint(
                0, scheduler.config.num_train_timesteps, (1,), device=device,
            ).long()
            noisy = scheduler.add_noise(latents, noise, t)

            noise_pred = unet(
                noisy, t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond,
            ).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

        all_grad_vectors.append(_grad_vector(params).detach().cpu().float())
        _zero_grads(params)

    return _compute_svd_subspace(all_grad_vectors, top_k, param_names)


# ---------------------------------------------------------------------------
# Flux gradient computation
# ---------------------------------------------------------------------------

def compute_gradients_flux(
    model_id: str,
    prompts: list[str],
    top_k: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    num_timesteps: int,
) -> dict:
    """Compute safety gradient matrix for Flux.1-dev (DiT attention layers)."""
    from diffusers import FluxPipeline

    print(f"\nLoading Flux: {model_id}")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    transformer = pipe.transformer.to(device)
    text_encoder = pipe.text_encoder.to(device)   # CLIP
    text_encoder_2 = pipe.text_encoder_2.to(device)  # T5
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    del pipe

    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(True)

    param_names, params = _collect_target_params(transformer, LORA_TARGET_MODULES)
    print(f"Tracking gradients for {len(params)} attention parameters "
          f"({sum(p.numel() for p in params):,} values)")

    torch.manual_seed(seed)
    resolution = 512
    latent_channels = transformer.config.in_channels
    latent_h = latent_w = resolution // 8

    all_grad_vectors = []

    for prompt in tqdm(prompts, desc="Safety gradients (Flux)"):
        _zero_grads(params)

        # Flux uses CLIP (77 tokens) + T5 (512 tokens)
        tok1 = tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).to(device)
        tok2 = tokenizer_2(
            prompt, padding="max_length", max_length=512,
            truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            pooled_prompt_embeds = text_encoder(
                tok1.input_ids, output_hidden_states=False,
            )[1]
            prompt_embeds = text_encoder_2(tok2.input_ids)[0]

        # Flux uses flow matching — timestep is a continuous value in [0, 1]
        for _ in range(num_timesteps):
            latents = torch.randn(
                1, latent_channels, latent_h, latent_w, dtype=dtype, device=device,
            )
            t = torch.rand(1, device=device)
            noise = torch.randn_like(latents)
            noisy = (1 - t) * latents + t * noise

            noise_pred = transformer(
                hidden_states=noisy,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
            ).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

        all_grad_vectors.append(_grad_vector(params).detach().cpu().float())
        _zero_grads(params)

    return _compute_svd_subspace(all_grad_vectors, top_k, param_names)


# ---------------------------------------------------------------------------
# SVD subspace computation (shared)
# ---------------------------------------------------------------------------

def _compute_svd_subspace(
    grad_vectors: list[torch.Tensor],
    top_k: int,
    param_names: list[str],
) -> dict:
    """Stack gradient vectors and extract the top-k safety subspace via SVD.

    The safety subspace S (shape [k, D]) is the matrix of top-k right
    singular vectors of the [num_prompts × D] gradient matrix G.
    Projecting a fine-tuning gradient g away from S:
        g_safe = g - S.T @ (S @ g)
    removes the component that would erode safety capability.
    """
    print(f"\nStacking {len(grad_vectors)} gradient vectors...")
    G = torch.stack(grad_vectors)  # [num_prompts, D]
    print(f"Gradient matrix: {G.shape}  ({G.numel() * 4 / 1e6:.1f} MB)")

    mean_grad = G.mean(dim=0)

    k = min(top_k, G.shape[0], G.shape[1])
    print(f"Computing SVD (retaining top {k} components)...")

    try:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        safety_subspace = Vh[:k]   # [k, D]
        singular_values = S[:k]
        explained = (singular_values ** 2).sum() / (S ** 2).sum()
        print(f"  Top-{k} components explain {explained:.1%} of gradient variance")
    except Exception as e:
        print(f"  SVD failed ({e}); falling back to mean gradient as 1-D subspace")
        safety_subspace = mean_grad.unsqueeze(0)
        singular_values = torch.tensor([mean_grad.norm()])

    return {
        "mean_gradient": mean_grad,
        "safety_subspace": safety_subspace,
        "singular_values": singular_values,
        "param_names": param_names,
        "num_prompts": len(grad_vectors),
        "top_k": k,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute NCII safety gradient subspace for SafeGrad / SaLoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.1-dev",
        choices=list(SUPPORTED_MODELS.keys()),
        help=(
            "Base model to compute gradients against. "
            "Use stabilityai/stable-diffusion-xl-base-1.0 for T4 / 16GB VRAM."
        ),
    )
    parser.add_argument(
        "--safety_prompts", required=True,
        help="Path to NCII-adjacent safety prompts file (one per line)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .pt file for safety gradient subspace",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Number of SVD components retained for the safety subspace",
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=5,
        help="Number of random timesteps to average gradients over per prompt",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override (default: auto-detect cuda/cpu)",
    )
    args = parser.parse_args()

    prompts_path = Path(args.safety_prompts)
    if not prompts_path.exists():
        print(f"Error: Safety prompts file not found: {prompts_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_type = SUPPORTED_MODELS[args.model]

    print("=" * 60)
    print("POC 2 — Safety Gradient Computation")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Architecture:   {model_type.upper()}")
    print(f"Prompts:        {prompts_path}")
    print(f"Output:         {output_path}")
    print(f"Top-k SVD:      {args.top_k}")
    print(f"Timesteps/prompt: {args.num_timesteps}")
    print(f"Device:         {device} ({dtype})")
    print("=" * 60)

    prompts = load_safety_prompts(prompts_path)

    if model_type == "sdxl":
        result = compute_gradients_sdxl(
            args.model, prompts, args.top_k, device, dtype,
            args.seed, args.num_timesteps,
        )
    else:
        result = compute_gradients_flux(
            args.model, prompts, args.top_k, device, dtype,
            args.seed, args.num_timesteps,
        )

    # Save gradient subspace
    timestamp = datetime.now(timezone.utc).isoformat()
    torch.save({
        "model_id": args.model,
        "model_type": model_type,
        "mean_gradient": result["mean_gradient"],
        "safety_subspace": result["safety_subspace"],   # [k, D]
        "singular_values": result["singular_values"],
        "param_names": result["param_names"],
        "num_prompts": result["num_prompts"],
        "top_k": result["top_k"],
        "num_timesteps": args.num_timesteps,
        "seed": args.seed,
        "prompts_file": str(prompts_path),
        "timestamp": timestamp,
    }, output_path)

    subspace_mb = result["safety_subspace"].numel() * 4 / 1e6
    print(f"\nSaved safety gradient subspace → {output_path}")
    print(f"  Subspace shape:    {list(result['safety_subspace'].shape)}")
    print(f"  Parameters tracked: {len(result['param_names'])}")
    print(f"  File size (subspace): ~{subspace_mb:.1f} MB")

    # Human-readable metadata alongside the .pt file
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_id": args.model,
            "model_type": model_type,
            "num_prompts": result["num_prompts"],
            "top_k": result["top_k"],
            "subspace_shape": list(result["safety_subspace"].shape),
            "num_params_tracked": len(result["param_names"]),
            "singular_values_top10": result["singular_values"][:10].tolist(),
            "num_timesteps": args.num_timesteps,
            "seed": args.seed,
            "prompts_file": str(prompts_path),
            "timestamp": timestamp,
        }, f, indent=2)
    print(f"  Metadata → {meta_path}")

    print(f"\nNext: python poc2_locked_model/02_apply_safegrad.py \\")
    print(f"  --model {args.model} \\")
    print(f"  --safety_gradients {output_path} \\")
    print(f"  --output poc2_locked_model/locked_model/"
          f"{'flux_locked' if model_type == 'flux' else 'sdxl_locked'}")


if __name__ == "__main__":
    main()
