"""
Teacher pseudo-label generation for SDFT (Self-Distillation Fine-Tuning).

The teacher VLM (Qwen2-VL-2B or larger) generates structured JSON explanations
for each creative. These become the training targets for the student VLM.

SDFT key idea: the teacher conditions on expert demonstrations (in-context examples
of high/low performers), then generates its own training signal on-policy — allowing
the student to learn domain knowledge without catastrophic forgetting of base capabilities.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


TEACHER_SYSTEM = (
    "You are an expert ad creative performance analyst. "
    "Given a creative image and its performance data, produce a concise JSON analysis."
)

TEACHER_TEMPLATE = """\
Analyze this ad creative.

Creative attributes:
- Format: {format}
- Theme: {theme}
- Hook: {hook_type}
- CTA: {cta_text}
- Emotional tone: {emotional_tone}
- Dominant color: {dominant_color}
- Has discount badge: {has_discount_badge}
- Text density: {text_density:.2f}
- Readability: {readability_score:.2f}
- Novelty: {novelty_score:.2f}
- Motion: {motion_score:.2f}
- Brand visibility: {brand_visibility_score:.2f}
- Clutter: {clutter_score:.2f}
- Faces: {faces_count}, Products: {product_count}

Performance:
- Status: {creative_status}
- Overall CTR: {overall_ctr:.4f}
- Overall IPM: {overall_ipm:.3f}
- CTR decay: {ctr_decay_pct:.1%}
- Vertical: {vertical}
- Objective: {objective}

Respond ONLY with valid JSON in this exact schema:
{{
  "performance_summary": "<1-2 sentence explanation of why this creative performs as it does>",
  "visual_strengths": ["<strength 1>", "<strength 2>"],
  "visual_weaknesses": ["<weakness 1>"],
  "fatigue_risk_reason": "<why this creative is/isn't at risk of fatigue>",
  "top_recommendation": "<single most impactful change to improve this creative>"
}}"""


class TeacherLabeler:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model: Optional[Qwen2VLForConditionalGeneration] = None
        self._processor: Optional[AutoProcessor] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _build_prompt(self, row: Dict, image_path: Path) -> List[Dict]:
        text = TEACHER_TEMPLATE.format(**{k: row.get(k, "unknown") for k in [
            "format", "theme", "hook_type", "cta_text", "emotional_tone", "dominant_color",
            "has_discount_badge", "text_density", "readability_score", "novelty_score",
            "motion_score", "brand_visibility_score", "clutter_score", "faces_count",
            "product_count", "creative_status", "overall_ctr", "overall_ipm",
            "ctr_decay_pct", "vertical", "objective",
        ]})
        return [
            {"role": "system", "content": TEACHER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": text},
                ],
            },
        ]

    @torch.no_grad()
    def generate_label(self, row: Dict, image_path: Path) -> Optional[Dict]:
        self._load_model()
        messages = self._build_prompt(row, image_path)

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            text=[text_input], images=[image], return_tensors="pt"
        ).to(self._model.device)

        output_ids = self._model.generate(
            **inputs, max_new_tokens=256, temperature=0.1, do_sample=False
        )
        generated = self._processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        try:
            # Extract JSON even if surrounded by markdown code fences
            start = generated.find("{")
            end = generated.rfind("}") + 1
            return json.loads(generated[start:end]) if start >= 0 else None
        except json.JSONDecodeError:
            return None

    def label_all_creatives(
        self,
        master_df,
        asset_dir: Path,
        output_path: Path,
        resume: bool = True,
    ) -> List[Dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids: set = set()

        if resume and output_path.exists():
            with open(output_path) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        existing_ids.add(obj["creative_id"])
                    except Exception:
                        pass

        results = []
        with open(output_path, "a") as out_f:
            for _, row in master_df.iterrows():
                cid = int(row["creative_id"])
                if cid in existing_ids:
                    continue
                image_path = asset_dir / f"creative_{cid}.png"
                if not image_path.exists():
                    continue

                label = self.generate_label(row.to_dict(), image_path)
                if label is not None:
                    record = {"creative_id": cid, **label}
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    results.append(record)

        return results

    def validate_label(self, label: Dict) -> bool:
        required = ["performance_summary", "visual_strengths", "visual_weaknesses",
                    "fatigue_risk_reason", "top_recommendation"]
        return (
            all(k in label for k in required)
            and len(label.get("performance_summary", "")) > 20
            and isinstance(label.get("visual_strengths"), list)
        )
