"""
Fine-tune Flux edit on the (source, ensemble-brief, target) pairs produced by
generate_flux_pairs.py.

Two stages:
  1. **Supervised LoRA fine-tune (SFT)**
     Rank-32 LoRA on the cross-attention + DiT blocks. Standard MSE-on-velocity
     loss against the Nano-Banana-rendered targets. Equivalent to a
     DreamBooth-style refinement of the base Flux edit.

  2. **Reward-weighted DPO** (optional, --dpo)
     For each pair we have post_edit_health and pre_edit_health. We build
     preference triples: (source, brief, winner = teacher target,
     loser = a noised reconstruction of the source). The reward is the
     observed health-score lift. Loss = DPO with β=0.1 on the LM-style
     velocity preference, gated by the reward magnitude.

Run:
    cd models && PYTHONPATH=$PWD python3 scripts/finetune_flux_edit.py \\
        --pairs outputs/flux_pairs/manifest.jsonl \\
        --epochs 4 --batch 1 --grad_accum 16 --lr 1e-4 --rank 32

If you don't have an H100/A100 you can run the script with --dry-run to
validate the loop end-to-end on a tiny subset (CPU-friendly).

References:
- Diffusion-DPO: https://arxiv.org/abs/2311.12092
- ImageReward:  https://arxiv.org/abs/2305.16381
- Flux:         https://github.com/black-forest-labs/flux
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs/models/flux_edit_finetuned"
DEFAULT_PAIRS = ROOT / "outputs/flux_pairs/manifest.jsonl"


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Manifest missing at {path}. Run generate_flux_pairs.py first.")
    pairs = []
    with open(path) as f:
        for line in f:
            try:
                pairs.append(json.loads(line))
            except Exception:
                continue
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str,
                        default="to_q,to_k,to_v,to_out.0",
                        help="Comma-separated target module names for LoRA injection")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--checkpoint", action="store_true", default=True)
    parser.add_argument("--dpo", action="store_true",
                        help="Also run the reward-weighted DPO stage after SFT")
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate the loop on 4 samples, no real training")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    if args.dry_run:
        pairs = pairs[:4]
    print(f"Loaded {len(pairs)} pairs from {args.pairs.relative_to(REPO)}")

    # ── Lazy import so the script's --help works without diffusers installed.
    try:
        import torch
        from diffusers import FluxPipeline   # type: ignore
        from peft import LoraConfig, get_peft_model   # type: ignore
        from PIL import Image
    except ImportError as e:
        print(f"\nMissing deps. Install with:\n"
              f"  pip install diffusers transformers peft accelerate bitsandbytes pillow\n"
              f"\n(actual error: {e})")
        sys.exit(1)

    print("\nLoading Flux edit base model — this will pull weights on first run.")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    if args.checkpoint:
        try: pipe.transformer.gradient_checkpointing_enable()
        except Exception: pass

    # Inject rank-r LoRA into the DiT cross-attention + attention output blocks.
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=[s.strip() for s in args.target_modules.split(",")],
        lora_dropout=args.dropout,
        bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_cfg)
    n_trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    print(f"LoRA injected. Trainable transformer params: {n_trainable / 1e6:.1f}M")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: supervised LoRA training loop ────────────────────────────
    pipe.transformer.train()
    optim = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    log_path = OUTPUT_DIR / "train_log.jsonl"
    log = open(log_path, "a")

    print("\n=== Stage 1: SFT (Nano-Banana targets) ===")
    step = 0
    for epoch in range(args.epochs):
        for i, pair in enumerate(pairs):
            src_img = Image.open(REPO / pair["source"]).convert("RGB")
            tgt_img = Image.open(REPO / pair["target"]).convert("RGB")
            brief = pair["brief"]

            # Encode source + target into latent space; predict velocity for target
            # conditioned on (source latent, text-embed of brief). Standard Flux
            # edit objective.
            with torch.no_grad():
                src_lat = pipe.vae.encode(_to_tensor(src_img, args.bf16)).latent_dist.sample()
                tgt_lat = pipe.vae.encode(_to_tensor(tgt_img, args.bf16)).latent_dist.sample()
                # Flux uses rectified flow — sample t in [0, 1] and interpolate.
                t = torch.rand(1, device=src_lat.device).clamp(min=0.05, max=0.95)
                noisy = (1 - t) * src_lat + t * tgt_lat
                target_velocity = tgt_lat - src_lat

            text_emb = pipe.encode_prompt(brief)[0]
            pred_velocity = pipe.transformer(
                hidden_states=noisy,
                timestep=t * 1000,
                encoder_hidden_states=text_emb,
            ).sample

            loss = (pred_velocity - target_velocity).pow(2).mean()
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1

            if i % 10 == 0:
                msg = {"epoch": epoch, "i": i, "step": step, "loss": float(loss.detach()),
                       "lift": pair.get("lift", 0)}
                print("  " + json.dumps(msg))
                log.write(json.dumps(msg) + "\n"); log.flush()

            if args.dry_run and i >= 3:
                break

    # Save LoRA adapter only (the base Flux weights stay on disk).
    pipe.transformer.save_pretrained(str(OUTPUT_DIR / "lora_sft"))
    print(f"\nStage 1 done. Adapter saved → {OUTPUT_DIR.relative_to(REPO)}/lora_sft")

    # ── Stage 2: reward-weighted DPO (optional) ───────────────────────────
    if args.dpo:
        print("\n=== Stage 2: reward-weighted DPO ===")
        # The "winner" is the Nano-Banana target; the "loser" is the source
        # itself (the unedited weak version). Reward = observed lift.
        for epoch in range(max(1, args.epochs // 2)):
            for i, pair in enumerate(pairs):
                lift = float(pair.get("lift", 0))
                if lift <= 0:
                    continue
                # weighted-DPO: weight each preference by the lift magnitude
                w = max(0.1, min(1.0, lift / 20.0))

                src_img = Image.open(REPO / pair["source"]).convert("RGB")
                tgt_img = Image.open(REPO / pair["target"]).convert("RGB")
                with torch.no_grad():
                    src_lat = pipe.vae.encode(_to_tensor(src_img, args.bf16)).latent_dist.sample()
                    tgt_lat = pipe.vae.encode(_to_tensor(tgt_img, args.bf16)).latent_dist.sample()
                    t = torch.rand(1, device=src_lat.device).clamp(min=0.05, max=0.95)

                text_emb = pipe.encode_prompt(pair["brief"])[0]
                pred_w = pipe.transformer(hidden_states=(1-t)*src_lat + t*tgt_lat,
                                          timestep=t*1000, encoder_hidden_states=text_emb).sample
                pred_l = pipe.transformer(hidden_states=src_lat,
                                          timestep=t*1000, encoder_hidden_states=text_emb).sample

                # DPO log-ratio loss on velocities (proxy for log-prob).
                logp_w = -(pred_w - (tgt_lat - src_lat)).pow(2).mean()
                logp_l = -(pred_l - (tgt_lat - src_lat)).pow(2).mean()
                logits = args.dpo_beta * (logp_w - logp_l)
                loss = -torch.nn.functional.logsigmoid(logits) * w
                (loss / args.grad_accum).backward()
                if (i + 1) % args.grad_accum == 0:
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                if i % 10 == 0:
                    msg = {"stage": "dpo", "epoch": epoch, "i": i, "loss": float(loss.detach()),
                           "lift": lift, "w": w}
                    print("  " + json.dumps(msg))
                    log.write(json.dumps(msg) + "\n"); log.flush()

                if args.dry_run and i >= 3:
                    break

        pipe.transformer.save_pretrained(str(OUTPUT_DIR / "lora_dpo"))
        print(f"Stage 2 done. Adapter saved → {OUTPUT_DIR.relative_to(REPO)}/lora_dpo")

    log.close()
    print("\nAll done. Load the adapter at inference with:")
    print(f"  pipe.load_lora_weights('{OUTPUT_DIR}/lora_dpo'  # or lora_sft)")


def _to_tensor(img: "Image.Image", bf16: bool):
    import torch
    import numpy as np
    arr = np.asarray(img.resize((512, 512))).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(torch.bfloat16 if bf16 else torch.float32)


if __name__ == "__main__":
    main()
