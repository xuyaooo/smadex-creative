"""
Inference wrapper for the finetuned student VLM.
Loads the LoRA-merged checkpoint and provides structured creative analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


INFERENCE_SYSTEM = (
    "You are an ad creative performance analyst. "
    "Given a creative image and its metadata, explain its performance in JSON format."
)

INFERENCE_TEMPLATE = """\
Analyze this ad creative.
Format: {format} | Theme: {theme} | Hook: {hook_type}
CTA: {cta_text} | Tone: {emotional_tone} | Color: {dominant_color}
Novelty: {novelty_score:.2f} | Motion: {motion_score:.2f} | Clutter: {clutter_score:.2f}
Vertical: {vertical} | Objective: {objective}

Return JSON with keys: performance_summary, visual_strengths, visual_weaknesses, fatigue_risk_reason, top_recommendation."""


class VLMCreativeAnalyzer:
    def __init__(self, checkpoint: str, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None   # AutoModelForCausalLM, loaded lazily
        self._processor = None  # AutoProcessor, loaded lazily

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        self._processor = AutoProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()

    def analyze(self, image_path: str | Path, row: Dict) -> Dict:
        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")

        prompt_text = INFERENCE_TEMPLATE.format(**{k: row.get(k, "unknown") for k in [
            "format", "theme", "hook_type", "cta_text", "emotional_tone", "dominant_color",
            "novelty_score", "motion_score", "clutter_score", "vertical", "objective",
        ]})

        messages = [
            {"role": "system", "content": INFERENCE_SYSTEM},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=text, images=[image], return_tensors="pt"
        ).to(self._model.device)

        output_ids = self._model.generate(
            **inputs, max_new_tokens=256, do_sample=False, temperature=1.0
        )
        generated = self._processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        try:
            start = generated.find("{")
            end = generated.rfind("}") + 1
            return json.loads(generated[start:end]) if start >= 0 else {"raw": generated}
        except json.JSONDecodeError:
            return {"raw": generated}

    def analyze_batch(self, items: List[Dict]) -> List[Dict]:
        return [self.analyze(item["image_path"], item["row"]) for item in items]

    @classmethod
    def load(cls, cfg: dict) -> "VLMCreativeAnalyzer":
        checkpoint = cfg.get("student_checkpoint", "outputs/models/vlm_finetuned")
        return cls(checkpoint=checkpoint, device=cfg.get("device", "auto"))
