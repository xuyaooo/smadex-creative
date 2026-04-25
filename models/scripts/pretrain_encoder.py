"""
Domain-adaptive contrastive fine-tuning of SigLIP-2 on the Smadex creatives.

For each creative we synthesize a text caption from its metadata (vertical,
theme, dominant color, headline, CTA, emotional tone) and run a sigmoid
contrastive loss between (image, caption) pairs in a batch. This nudges the
visual encoder to organize ad imagery along marketer-relevant axes (vertical,
theme, color), which the off-the-shelf SigLIP-2 isn't trained for.

After fine-tuning, re-encode all 1,080 PNGs with the new encoder and re-run
`scripts/train_all.py` to measure the lift.

Usage:
    python3 scripts/pretrain_encoder.py
    python3 scripts/pretrain_encoder.py --epochs 5 --batch-size 32

References:
    - SigLIP loss: Zhai et al. ICCV 2023
    - SigLIP-2: Tschannen et al. 2025 (arXiv 2502.14786)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.loader import DataLoader as MasterLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_for(row) -> str:
    """Build a marketer-readable caption from creative metadata."""
    parts = []
    if row.get("vertical"):
        parts.append(f"a {row['vertical']} ad")
    if row.get("format"):
        parts.append(f"in {row['format'].replace('_', ' ')} format")
    if row.get("dominant_color"):
        parts.append(f"with a {row['dominant_color']} dominant color")
    if row.get("theme"):
        parts.append(f"using a {row['theme'].replace('-', ' ').replace('_', ' ')} theme")
    if row.get("emotional_tone"):
        parts.append(f"in a {row['emotional_tone']} tone")
    text = ", ".join(parts) + "."
    headline = str(row.get("headline", "") or "").strip()
    if headline:
        text += f" Headline: {headline!r}."
    cta = str(row.get("cta_text", "") or "").strip()
    if cta:
        text += f" CTA: {cta!r}."
    return text


class CreativeDataset(Dataset):
    def __init__(self, master_df, asset_dir: Path, processor):
        self.records = []
        for _, row in master_df.iterrows():
            img_path = asset_dir / f"creative_{int(row['creative_id'])}.png"
            if img_path.exists():
                self.records.append({
                    "creative_id": int(row["creative_id"]),
                    "img_path": str(img_path),
                    "caption": caption_for(row),
                })
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return r


def collate(batch, processor):
    images = [Image.open(b["img_path"]).convert("RGB") for b in batch]
    captions = [b["caption"] for b in batch]
    inputs = processor(
        images=images, text=captions,
        padding="max_length", max_length=64, truncation=True,
        return_tensors="pt",
    )
    return inputs


def sigmoid_contrastive_loss(image_embeds, text_embeds, t: torch.nn.Parameter,
                              b: torch.nn.Parameter):
    """SigLIP loss: per-pair sigmoid log-likelihood, batch-balanced.

    Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training", ICCV 2023.
    """
    logits = torch.matmul(image_embeds, text_embeds.T) * torch.exp(t) + b
    n = logits.size(0)
    targets = 2 * torch.eye(n, device=logits.device) - 1  # +1 on diagonal, -1 elsewhere
    loss = -F.logsigmoid(targets * logits).sum() / n
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", default="outputs/models/siglip2_finetuned")
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor
    BASE = "google/siglip2-base-patch16-256"
    print(f"Loading {BASE}...")
    processor = AutoProcessor.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE)

    loader = MasterLoader("config.yaml")
    master = loader.load_master_table()
    asset_dir = (Path("config.yaml").parent / "../assets").resolve()
    ds = CreativeDataset(master, asset_dir, processor)
    print(f"  dataset size: {len(ds)} creatives with images")

    sample_caption = ds.records[0]["caption"]
    print(f"  sample caption: {sample_caption}")

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=lambda b: collate(b, processor),
    )

    # Trainable temperature + bias for SigLIP loss
    t_param = torch.nn.Parameter(torch.tensor(np.log(10.0), device=DEVICE,
                                              dtype=torch.float32))
    b_param = torch.nn.Parameter(torch.tensor(-10.0, device=DEVICE,
                                              dtype=torch.float32))

    optim = torch.optim.AdamW(
        list(model.parameters()) + [t_param, b_param],
        lr=args.lr, weight_decay=1e-4,
    )

    print(f"\nTraining {args.epochs} epochs on {DEVICE}, batch={args.batch_size}, lr={args.lr}")
    model.train()
    t0 = time.time()
    for ep in range(args.epochs):
        ep_loss = 0.0
        n_batches = 0
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                img_e = F.normalize(outputs.image_embeds.float(), dim=-1)
                txt_e = F.normalize(outputs.text_embeds.float(), dim=-1)
            loss = sigmoid_contrastive_loss(img_e, txt_e, t_param, b_param)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        print(f"  epoch {ep+1}/{args.epochs}  avg loss = {avg:.4f}  "
              f"t={t_param.item():.2f}  b={b_param.item():.2f}  "
              f"elapsed={time.time()-t0:.0f}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    print(f"\nSaved fine-tuned encoder to {out}")
    print("Next steps:")
    print(f"  1. Update config.yaml: embeddings.clip_model = '{out}'")
    print("  2. Re-run scripts/recompute_embeddings.py")
    print("  3. Re-run scripts/train_all.py")
    print("  4. Re-run scripts/build_artifacts.py")


if __name__ == "__main__":
    main()
