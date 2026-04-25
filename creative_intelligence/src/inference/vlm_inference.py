"""
SmolVLM inference for fresh annotations on creatives that don't have a
precomputed teacher annotation (e.g., a brand-new creative uploaded to the demo).

Loads the LoRA adapter on top of the base SmolVLM. Lazy-loaded — the demo only
pays the GPU init cost when this is actually called.
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image


PROMPT_TEMPLATE = (
    "You are an ad creative analyst. Given the image and metadata below, "
    "produce a JSON analysis with keys: performance_summary (1-2 sentences), "
    "visual_strengths (list), visual_weaknesses (list), fatigue_risk_reason, "
    "top_recommendation.\n\n"
    "Metadata: vertical={vertical}, format={format}, theme={theme}, "
    "dominant_color={dominant_color}, has_discount_badge={has_discount_badge}, "
    "headline=\"{headline}\", cta=\"{cta_text}\".\n"
)


class SmolVLMAnalyzer:
    """Wraps the LoRA-finetuned SmolVLM for on-demand annotation generation."""

    def __init__(self, base_model: str, adapter_dir: Path, device: str = "auto"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel

        self.device = (
            "cuda" if torch.cuda.is_available() and device in ("auto", "cuda")
            else "cpu"
        )
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(base_model)
        base = AutoModelForImageTextToText.from_pretrained(
            base_model, dtype=dtype, device_map=self.device,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()

    @torch.inference_mode()
    def annotate(self, image_path: str, metadata: Dict, max_new_tokens: int = 350) -> Dict:
        image = Image.open(image_path).convert("RGB")
        prompt = PROMPT_TEMPLATE.format(**{
            k: metadata.get(k, "")
            for k in ["vertical", "format", "theme", "dominant_color",
                     "has_discount_badge", "headline", "cta_text"]
        })
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)

        t0 = time.time()
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0

        # Decode only the new tokens
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        decoded = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Try to parse JSON from the decoded text
        parsed = _try_parse_json(decoded)
        if parsed is None:
            parsed = {
                "performance_summary": decoded.strip()[:400],
                "visual_strengths": [],
                "visual_weaknesses": [],
                "fatigue_risk_reason": "",
                "top_recommendation": "",
            }
        parsed["_inference_seconds"] = round(elapsed, 2)
        parsed["source_model"] = "SmolVLM-Instruct + LoRA (local)"
        return parsed


def _try_parse_json(text: str) -> Optional[Dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
