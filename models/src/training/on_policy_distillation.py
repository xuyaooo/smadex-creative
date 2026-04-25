"""
On-Policy Self-Distillation (SDFT implementation).

The key distinction from offline SFT:
  - OFFLINE SFT: train student on teacher labels directly
    → student learns teacher's distribution but diverges on its own outputs
  - ON-POLICY SDFT: student generates → teacher scores/corrects → student trains on its own
    corrected outputs → no distribution mismatch

This implements two interleaved phases:

Phase A (Offline warm-up):
  Student is finetuned on teacher (OpenRouter) labels for N epochs.
  Gives the student a domain-adapted starting point.

Phase B (On-policy rounds):
  1. Student generates completions for a batch of creatives (temperature > 0)
  2. Teacher (OpenRouter) scores each student output:
     - If valid JSON and "good enough" → keep as-is (student self-confirms)
     - Otherwise → teacher provides corrected label
  3. Student finetunes on the mix of:
     - self-confirmed outputs (on-policy, student's own distribution)
     - teacher-corrected outputs (targeted fixes)
     - rehearsal examples (prevent catastrophic forgetting)
  4. Repeat with next batch

This is equivalent to the SDFT paper's on-policy distillation where the model
generates its own training signal with occasional teacher correction.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel


# Scoring rubric for teacher validation of student outputs
JUDGE_SYSTEM = (
    "You are evaluating an AI-generated ad creative analysis. "
    "Check if the analysis is accurate given the metadata."
)

JUDGE_TEMPLATE = """\
Creative metadata: {metadata_summary}
Actual status: {creative_status}, CTR: {overall_ctr:.4f}, CTR decay: {ctr_decay_pct:.1%}

Student analysis:
{student_output}

Is this analysis accurate and specific? Reply with JSON:
{{"score": <0-10>, "issues": ["<issue 1 if any>"], "corrected_summary": "<corrected 1-sentence summary if score < 7, else null>"}}"""


class OnPolicyDistiller:
    """
    Runs on-policy SDFT rounds using OpenRouter teacher + local student.
    """

    def __init__(
        self,
        student_checkpoint: str,
        teacher,          # OpenRouterTeacher instance
        asset_dir: Path,
        output_dir: Path,
        device: str = "auto",
        rehearsal_fraction: float = 0.2,
        confidence_threshold: float = 7.0,   # teacher score out of 10
        on_policy_batch_size: int = 50,
    ):
        self.student_checkpoint = student_checkpoint
        self.teacher = teacher
        self.asset_dir = asset_dir
        self.output_dir = output_dir
        self.device = device
        self.rehearsal_fraction = rehearsal_fraction
        self.confidence_threshold = confidence_threshold
        self.batch_size = on_policy_batch_size

        self._all_training_records: List[Dict] = []
        self._round = 0

    def _load_student(self) -> Tuple:
        processor = AutoProcessor.from_pretrained(self.student_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            self.student_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        model.eval()
        return model, processor

    @torch.no_grad()
    def _student_generate(
        self,
        model,
        processor,
        row: Dict,
        image_path: Path,
        temperature: float = 0.7,
    ) -> str:
        from src.training.train_vlm import STUDENT_SYSTEM, STUDENT_TEMPLATE

        image = Image.open(image_path).convert("RGB")
        prompt_text = STUDENT_TEMPLATE.format(**{
            k: row.get(k, "unknown") for k in [
                "format", "theme", "hook_type", "cta_text", "emotional_tone", "dominant_color",
                "novelty_score", "motion_score", "clutter_score", "vertical", "objective",
            ]
        })

        messages = [
            {"role": "system", "content": STUDENT_SYSTEM},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )
        return processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def _teacher_score(self, row: Dict, student_output: str) -> Dict:
        """Ask OpenRouter teacher to score and optionally correct a student output."""
        metadata_summary = (
            f"format={row.get('format')}, vertical={row.get('vertical')}, "
            f"theme={row.get('theme')}, novelty={row.get('novelty_score', 0):.2f}, "
            f"clutter={row.get('clutter_score', 0):.2f}"
        )
        prompt = JUDGE_TEMPLATE.format(
            metadata_summary=metadata_summary,
            creative_status=row.get("creative_status", "?"),
            overall_ctr=float(row.get("overall_ctr", 0)),
            ctr_decay_pct=float(row.get("ctr_decay_pct", 0)),
            student_output=student_output[:600],
        )

        for attempt in range(3):
            try:
                response = self.teacher.client.chat.completions.create(
                    model=self.teacher.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=150,
                    temperature=0.0,
                )
                text = response.choices[0].message.content or ""
                start = text.find("{")
                end = text.rfind("}") + 1
                return json.loads(text[start:end]) if start >= 0 else {"score": 0}
            except Exception as e:
                time.sleep(2 ** attempt)
        return {"score": 0}

    def run_on_policy_round(
        self,
        master_df,
        labels_path: Path,
        train_fn,           # callable: (records, checkpoint_path) → None
    ) -> Dict:
        """
        One on-policy SDFT round:
        1. Student generates outputs for a random batch
        2. Teacher scores → keep self-confirmed or correct
        3. Merge with rehearsal
        4. Finetune student
        """
        self._round += 1
        round_dir = self.output_dir / f"round_{self._round:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Sample batch
        batch = master_df.sample(min(self.batch_size, len(master_df)), random_state=self._round)

        model, processor = self._load_student()
        on_policy_records = []
        teacher_corrected = 0
        self_confirmed = 0

        print(f"\n[Round {self._round}] On-policy generation for {len(batch)} creatives...")
        for _, row in batch.iterrows():
            cid = int(row["creative_id"])
            img_path = self.asset_dir / f"creative_{cid}.png"
            if not img_path.exists():
                continue

            # Student generates (on-policy, with temperature)
            student_output = self._student_generate(model, processor, row.to_dict(), img_path,
                                                     temperature=0.7)

            # Teacher scores
            score_result = self._teacher_score(row.to_dict(), student_output)
            score = float(score_result.get("score", 0))

            if score >= self.confidence_threshold:
                # Student output is good — use it as training target (on-policy)
                student_json = self._parse_json(student_output)
                if student_json:
                    on_policy_records.append({
                        "creative_id": cid,
                        "source": "student_self",
                        "score": score,
                        **student_json,
                    })
                    self_confirmed += 1
            else:
                # Teacher corrects — re-label with full teacher call
                corrected = self.teacher.label_one(row.to_dict(), img_path)
                if corrected:
                    on_policy_records.append({
                        "creative_id": cid,
                        "source": "teacher_corrected",
                        "score": 10.0,
                        **corrected,
                    })
                    teacher_corrected += 1

            time.sleep(self.teacher._delay)

        del model  # free GPU

        # Rehearsal: mix in old training data
        n_rehearsal = max(1, int(len(self._all_training_records) * self.rehearsal_fraction))
        rehearsal = random.sample(
            self._all_training_records,
            min(n_rehearsal, len(self._all_training_records))
        )

        merged = on_policy_records + rehearsal
        self._all_training_records.extend(on_policy_records)

        # Save merged labels for this round
        round_labels = round_dir / "on_policy_labels.jsonl"
        with open(round_labels, "w") as f:
            for rec in merged:
                f.write(json.dumps(rec) + "\n")

        stats = {
            "round": self._round,
            "self_confirmed": self_confirmed,
            "teacher_corrected": teacher_corrected,
            "rehearsal": len(rehearsal),
            "total_merged": len(merged),
        }
        print(f"  Self-confirmed: {self_confirmed}, Teacher-corrected: {teacher_corrected}, "
              f"Rehearsal: {len(rehearsal)}")

        # Finetune student on merged labels
        new_checkpoint = str(round_dir / "student")
        train_fn(merged, new_checkpoint)
        self.student_checkpoint = new_checkpoint

        return stats

    def load_existing_records(self, labels_path: Path) -> None:
        if not labels_path.exists():
            return
        with open(labels_path) as f:
            for line in f:
                try:
                    self._all_training_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(self._all_training_records)} existing training records.")

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end]) if start >= 0 else None
        except json.JSONDecodeError:
            return None
