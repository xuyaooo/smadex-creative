"""LoRA fine-tune SmolVLM-Instruct on the OpenRouter teacher annotations.

The cached teacher labels in `outputs/pseudo_labels/teacher_labels.jsonl` are
the supervision signal — for each (image, metadata) we already have a JSON
analysis (performance_summary, visual_strengths, visual_weaknesses,
fatigue_risk_reason, top_recommendation). We train SmolVLM via LoRA to emit
that same JSON given (image, metadata).

This is a one-time offline step. After it runs, `outputs/models/vlm_finetuned/`
holds a small adapter that can produce explanations for new creatives without
calling OpenRouter.

Hardware: tested on a single RTX 4090 (24 GB). LoRA r=16 fits comfortably.

Usage:
    python3 scripts/finetune_smolvlm.py
    python3 scripts/finetune_smolvlm.py --epochs 1 --batch-size 1   # smoke test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.loader import DataLoader

CONFIG_PATH = "config.yaml"
LABEL_PATH = "outputs/pseudo_labels/teacher_labels.jsonl"
ASSETS_DIR = "../data/assets"
OUTPUT_DIR = "outputs/models/vlm_finetuned"

PROMPT_TEMPLATE = (
    "You are an ad creative analyst. Given the image and metadata below, "
    "produce a JSON analysis with keys: performance_summary (1-2 sentences), "
    "visual_strengths (list), visual_weaknesses (list), fatigue_risk_reason, "
    "top_recommendation.\n\n"
    "Metadata: vertical={vertical}, format={format}, theme={theme}, "
    "dominant_color={dominant_color}, has_discount_badge={has_discount_badge}, "
    "headline=\"{headline}\", cta=\"{cta_text}\".\n"
)


def load_labels(jsonl_path: Path) -> dict:
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                cid = int(rec["creative_id"])
                # Strip metadata fields not part of the target
                target = {
                    "performance_summary": rec.get("performance_summary", ""),
                    "visual_strengths": rec.get("visual_strengths", []),
                    "visual_weaknesses": rec.get("visual_weaknesses", []),
                    "fatigue_risk_reason": rec.get("fatigue_risk_reason", ""),
                    "top_recommendation": rec.get("top_recommendation", ""),
                }
                out[cid] = target
            except Exception:
                pass
    return out


class CreativeAnnotationDataset(Dataset):
    def __init__(self, master_df, labels_by_cid, asset_dir: Path, processor):
        self.records = []
        for _, row in master_df.iterrows():
            cid = int(row["creative_id"])
            if cid not in labels_by_cid:
                continue
            img_path = asset_dir / f"creative_{cid}.png"
            if not img_path.exists():
                continue
            self.records.append({
                "creative_id": cid,
                "image_path": str(img_path),
                "metadata": {
                    "vertical": str(row.get("vertical", "")),
                    "format": str(row.get("format", "")),
                    "theme": str(row.get("theme", "")),
                    "dominant_color": str(row.get("dominant_color", "")),
                    "has_discount_badge": int(row.get("has_discount_badge", 0)),
                    "headline": str(row.get("headline", "")),
                    "cta_text": str(row.get("cta_text", "")),
                },
                "target": labels_by_cid[cid],
            })
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = Image.open(r["image_path"]).convert("RGB")
        prompt = PROMPT_TEMPLATE.format(**r["metadata"])
        target_json = json.dumps(r["target"])
        messages = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": target_json}
            ]},
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=text, images=[image], return_tensors="pt", padding=True)
        # Labels: same as input_ids; we'll mask the prompt portion via -100 below
        # Simpler: train on full sequence (model already learns to copy the prompt).
        inputs["labels"] = inputs["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in inputs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap on training rows (for smoke tests).")
    args = parser.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    print(f"Loading processor + model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Applying LoRA adapter...")
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"Loading teacher labels from {LABEL_PATH}")
    labels = load_labels(Path(LABEL_PATH))
    print(f"  {len(labels)} labeled creatives")

    loader = DataLoader(CONFIG_PATH)
    master = loader.load_master_table()
    asset_dir = (Path(CONFIG_PATH).parent / ASSETS_DIR).resolve()

    ds = CreativeAnnotationDataset(master, labels, asset_dir, processor)
    if args.max_samples:
        ds.records = ds.records[: args.max_samples]
    print(f"  training samples: {len(ds)}")

    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
    )

    def collate(batch):
        """SmolVLM produces variable image-patch counts per sample (image-splitting),
        so we use batch_size=1 and just unsqueeze. effective_batch = grad_accum."""
        if len(batch) != 1:
            raise ValueError(
                f"This collator only supports batch_size=1 (got {len(batch)}). "
                "Use --batch-size 1 --grad-accum N for effective batch size N."
            )
        b = batch[0]
        return {k: v.unsqueeze(0) for k, v in b.items()}

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved finetuned LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
