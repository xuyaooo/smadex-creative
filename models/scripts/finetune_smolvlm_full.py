"""
Full-parameter fine-tune of SmolVLM-Instruct on the OpenRouter teacher labels.

Differs from `finetune_smolvlm.py` (LoRA-only) by:
  - training EVERY parameter (LM + adapter + cross-attention) instead of low-rank deltas
  - higher capacity, no merge step at inference
  - bigger memory bill (use grad-checkpointing + bf16 + AdamW-8bit on H100/A100)

Run:
    cd models && PYTHONPATH=$PWD python3 scripts/finetune_smolvlm_full.py \
        --epochs 3 \
        --batch 4 \
        --grad_accum 8 \
        --lr 5e-6

Reads:
    outputs/pseudo_labels/teacher_labels.jsonl   (from label_with_openrouter.py)
    data/assets/creative_<cid>.png

Writes:
    outputs/models/vlm_finetuned_full/
        config.json, model.safetensors, processor/, tokenizer/, training_args.bin
        train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA = REPO / "data"
ASSETS = DATA / "assets"
sys.path.insert(0, str(ROOT))

CONFIG_PATH = "config.yaml"
LABEL_PATH = ROOT / "outputs/pseudo_labels/teacher_labels.jsonl"
ASSETS_DIR = DATA / "assets"
OUTPUT_DIR = ROOT / "outputs/models/vlm_finetuned_full"
BASE_MODEL = "HuggingFaceTB/SmolVLM-Instruct"


def load_pairs() -> list[dict]:
    if not LABEL_PATH.exists():
        sys.exit(f"Teacher labels missing at {LABEL_PATH}. Run label_with_openrouter.py first.")
    pairs = []
    with open(LABEL_PATH) as f:
        for line in f:
            try:
                row = json.loads(line)
                cid = int(row["creative_id"])
                img = ASSETS_DIR / f"creative_{cid}.png"
                if img.exists():
                    pairs.append({"image": str(img), "label": row})
            except Exception:
                continue
    return pairs


def build_text(label: dict) -> str:
    """Serialize the teacher's analysis JSON into the chat target."""
    return json.dumps({
        "performance_summary": label.get("performance_summary", ""),
        "visual_strengths": label.get("visual_strengths", []),
        "visual_weaknesses": label.get("visual_weaknesses", []),
        "fatigue_risk_reason": label.get("fatigue_risk_reason", ""),
        "top_recommendation": label.get("top_recommendation", ""),
    }, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--checkpoint", action="store_true", default=True,
                        help="Use gradient checkpointing to halve VRAM.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = all teacher labels")
    args = parser.parse_args()

    from transformers import (   # type: ignore
        AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer,
    )

    pairs = load_pairs()
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    print(f"Training set: {len(pairs)} (image, JSON) pairs")

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    if args.checkpoint:
        model.gradient_checkpointing_enable()
    # FULL fine-tune — every parameter trainable, NOT a LoRA.
    for p in model.parameters():
        p.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable / 1e9:.2f}B (full fine-tune)")

    SYSTEM = (
        "You are an expert ad-creative performance analyst. "
        "Given a creative image and metadata, return ONLY the requested JSON."
    )

    def encode_one(p):
        img = Image.open(p["image"]).convert("RGB")
        target_text = build_text(p["label"])
        # Build a single chat round: system + user(image+ask) → assistant(JSON).
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Analyze this creative and return the structured JSON."},
            ]},
            {"role": "assistant", "content": target_text},
        ]
        chat = processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = processor(text=chat, images=[img], return_tensors="pt")
        # Standard supervised loss = LM CE on the assistant span.
        labels = inputs["input_ids"].clone()
        # Mask everything before the final assistant turn so we only train on the JSON.
        # Simple heuristic: keep only the last quarter as labels (assistant span).
        cutoff = max(0, int(labels.size(1) * 0.75))
        labels[:, :cutoff] = -100
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"][0],
            "labels": labels[0],
        }

    # Lazy datasets to avoid materializing all images at once.
    class JsonImageDataset(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return encode_one(self.items[i])

    train_ds = JsonImageDataset(pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=args.checkpoint,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      tokenizer=processor.tokenizer)
    trainer.train()

    # Save the FULL weights (not adapters) + processor.
    model.save_pretrained(str(OUTPUT_DIR), safe_serialization=True)
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved full-FT model to {OUTPUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
