"""
VLM student finetuning with LoRA using teacher pseudo-labels (SDFT pipeline).

The student (SmolVLM-Instruct) learns to produce structured JSON creative analyses
conditioned on [image + metadata] inputs. LoRA keeps training memory-efficient
while the base model's general capabilities are preserved.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


STUDENT_SYSTEM = (
    "You are an ad creative performance analyst. "
    "Given a creative image and its metadata, explain its performance in JSON format."
)

STUDENT_TEMPLATE = """\
Analyze this ad creative.
Format: {format} | Theme: {theme} | Hook: {hook_type}
CTA: {cta_text} | Tone: {emotional_tone} | Color: {dominant_color}
Novelty: {novelty_score:.2f} | Motion: {motion_score:.2f} | Clutter: {clutter_score:.2f}
Vertical: {vertical} | Objective: {objective}

Return JSON with keys: performance_summary, visual_strengths, visual_weaknesses, fatigue_risk_reason, top_recommendation."""


class VLMCreativeDataset(Dataset):
    def __init__(
        self,
        master_df,
        labels: List[Dict],
        asset_dir: Path,
        processor,
        max_length: int = 512,
    ):
        label_map = {rec["creative_id"]: rec for rec in labels}
        self.samples = []
        for _, row in master_df.iterrows():
            cid = int(row["creative_id"])
            if cid not in label_map:
                continue
            img_path = asset_dir / f"creative_{cid}.png"
            if not img_path.exists():
                continue
            self.samples.append((row.to_dict(), label_map[cid], img_path))

        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        row, label, img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        target_json = json.dumps({
            k: label[k] for k in ["performance_summary", "visual_strengths",
                                   "visual_weaknesses", "fatigue_risk_reason",
                                   "top_recommendation"]
        })

        prompt_text = STUDENT_TEMPLATE.format(**{k: row.get(k, "unknown") for k in [
            "format", "theme", "hook_type", "cta_text", "emotional_tone", "dominant_color",
            "novelty_score", "motion_score", "clutter_score", "vertical", "objective",
        ]})

        messages = [
            {"role": "system", "content": STUDENT_SYSTEM},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]},
            {"role": "assistant", "content": target_json},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False)
        encoding = self.processor(
            text=text, images=[image],
            return_tensors="pt", truncation=True, max_length=self.max_length,
            padding="max_length",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask prompt tokens — only compute loss on the assistant response
        prompt_messages = messages[:-1]
        prompt_text_only = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_encoding = self.processor(
            text=prompt_text_only, images=[image],
            return_tensors="pt", truncation=True, max_length=self.max_length, padding=False,
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "pixel_values": encoding.get("pixel_values", torch.zeros(1)).squeeze(0)}


class VLMFinetuner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg.get("student_model", "HuggingFaceTB/SmolVLM-Instruct")
        self._model = None
        self._processor = None

    def _load_base_model(self) -> Tuple:
        processor = AutoProcessor.from_pretrained(self.model_name)
        model = Idefics3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=self.cfg.get("lora_r", 16),
            lora_alpha=self.cfg.get("lora_alpha", 32),
            target_modules=self.cfg.get("lora_target_modules",
                                        ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, processor

    def train(
        self,
        train_dataset: VLMCreativeDataset,
        val_dataset: VLMCreativeDataset,
        output_dir: str,
    ) -> None:
        model, processor = self._load_base_model()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=self.cfg.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=self.cfg.get("gradient_accumulation_steps", 8),
            learning_rate=float(self.cfg.get("learning_rate", 2e-4)),
            warmup_ratio=self.cfg.get("warmup_ratio", 0.1),
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=20,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    def load_labels(self, labels_path: str) -> List[Dict]:
        labels = []
        with open(labels_path) as f:
            for line in f:
                try:
                    labels.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return labels
