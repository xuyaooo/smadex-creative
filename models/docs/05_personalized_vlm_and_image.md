# 05 · Personalized VLM + Flux edit

> [← 04 · Visual intelligence](04_visual_intelligence.md) · [↑ Index](../README.md) · [06 · Evaluation →](06_evaluation.md)

Models **2** and **3** of the chain. Both are personalised on this
dataset; both ride on top of Model 1's outputs. Both have a third-party
fallback (Gemini 2.5 Flash Lite / Nano Banana on OpenRouter) that the
front-end uses by default when the local fine-tunes are not mounted.

| | Model 2 (analysis) | Model 3 (image edit) |
|---|---|---|
| **Base** | `HuggingFaceTB/SmolVLM-Instruct` (2.2B) | `black-forest-labs/FLUX.1-dev` |
| **Strategy** | Full fine-tune + SDFT | Rank-32 LoRA + reward-weighted DPO |
| **Supervision** | Teacher pseudo-labels (Gemma / Gemini Flash) | Nano-Banana-rendered targets driven by the tabular ensemble's lift brief |
| **Train script** | [`scripts/finetune_smolvlm_full.py`](../scripts/finetune_smolvlm_full.py) | [`scripts/finetune_flux_edit.py`](../scripts/finetune_flux_edit.py) |
| **Data builder** | [`scripts/label_with_openrouter.py`](../scripts/label_with_openrouter.py) | [`scripts/generate_flux_pairs.py`](../scripts/generate_flux_pairs.py) |
| **Output dir** | `outputs/models/vlm_finetuned_full/` | `outputs/models/flux_edit_finetuned/` |
| **Fallback** | Gemini 2.5 Flash Lite via OpenRouter | Gemini 2.5 Flash Image (Nano Banana) via OpenRouter |

## Quickstart — full personalisation chain

```bash
# 0. Make sure Model 1 (tabular ensemble) exists.
PYTHONPATH=models python3 models/scripts/build_clean_dataset.py
PYTHONPATH=models python3 models/scripts/train_clean.py --final

# 1. Per-vertical color palettes from real top-performer images.
PYTHONPATH=models python3 models/scripts/build_palette_lookup.py

# 2. Gemma teacher labels (analysis JSON per creative).
export OPENROUTER_API_KEY=sk-or-...
PYTHONPATH=models python3 models/scripts/label_with_openrouter.py

# 3. Full fine-tune SmolVLM on the teacher labels.
PYTHONPATH=models python3 models/scripts/finetune_smolvlm_full.py \
    --epochs 3 --batch 4 --grad_accum 8 --lr 5e-6

# 4. Nano Banana → (source, ensemble-brief, target) triples.
#    Each pair is filtered against the ensemble's post-edit health re-prediction.
PYTHONPATH=models python3 models/scripts/generate_flux_pairs.py \
    --max 200 --threshold 75

# 5. LoRA + DPO fine-tune on the kept pairs.
PYTHONPATH=models python3 models/scripts/finetune_flux_edit.py \
    --epochs 4 --batch 1 --grad_accum 16 --rank 32 --dpo
```

Wall-clock on a single H100 80 GB:
- SmolVLM full FT: ~4 h
- Flux edit LoRA + DPO: ~6 h

Both can be skipped at inference — the front-end falls back to the
OpenRouter equivalents transparently (same prompts, same output schema).

---

## Model 2 — Personalised VLM (analysis)

### Goal

Read an ad creative + its launch metadata, return structured JSON:

```
performance_summary · visual_strengths · visual_weaknesses ·
fatigue_risk_reason · top_recommendation
+ richer color_recommendations · layout_recommendations · copy_recommendations
```

### Training data

[`scripts/label_with_openrouter.py`](../scripts/label_with_openrouter.py)
sends every creative + its real performance metrics to a Gemma-family
teacher and asks for the JSON above.

Output: `outputs/pseudo_labels/teacher_labels.jsonl` (1,080 rows).

### Full fine-tune

[`scripts/finetune_smolvlm_full.py`](../scripts/finetune_smolvlm_full.py)
trains **every** parameter (no LoRA, no adapter), using:

- bf16 + gradient checkpointing (~22 GB VRAM at batch=4)
- supervised LM cross-entropy on the assistant span only
- 3 epochs at 5e-6 LR with 8× grad-accum
- adamw_torch optimiser, 5% warmup

The masking heuristic is simple: keep the last 25% of token positions as
labels (the assistant JSON span); the rest are masked with `-100`. For
production you'd want a proper chat-template tokeniser that returns the
exact assistant span — this script ships the pragmatic version.

### Self-distillation (optional)

After full FT, a second pass runs the SDFT loop in
[`src/training/on_policy_distillation.py`](../src/training/on_policy_distillation.py):
the student generates → teacher demonstrates the corrected output →
student fine-tunes on the demonstration. arXiv:2601.19897-style.

### Legacy LoRA path

[`scripts/finetune_smolvlm.py`](../scripts/finetune_smolvlm.py) still
ships the older r=16 LoRA recipe (q/k/v/o_proj). Not recommended —
the full FT runs strictly better at the same wall-clock — but kept for
A6000-class GPUs that can't fit the full FT.

---

## Model 3 — Image edit (Flux edit)

### Goal

Apply the **tabular ensemble's lift brief** to a creative and return a
ready-to-ship rebuild. Crucial point: the supervision is the ensemble's,
not the LLM's. We're not asking a VLM to guess what "better" means — we
hand it the ensemble's specific lift recommendations and say "apply
these".

### Training data

[`scripts/generate_flux_pairs.py`](../scripts/generate_flux_pairs.py)
writes one row per kept pair to `outputs/flux_pairs/manifest.jsonl`:

```json
{
  "creative_id": 500144,
  "vertical": "ecommerce",
  "format": "interstitial",
  "source": "models/outputs/flux_pairs/images/500144__source.png",
  "target": "models/outputs/flux_pairs/images/500144__target.png",
  "brief": "# Creative-rebuild brief (ensemble-driven)\n…",
  "pre_health": 38.0,
  "post_health": 81.0,
  "lift": 43.0
}
```

The brief is built from:
- Tabular ensemble output (predicted_status, health_score, class probs)
- Top-3 single-feature counterfactual lifts
- The data-grounded palette for the creative's vertical

The teacher (Nano Banana on OpenRouter) renders the improved variant.
**We re-score the result with the tabular ensemble** and only keep
pairs with `post_health ≥ threshold` (75 by default). Garbage-in pairs
would teach the student bad habits, so we drop them.

### LoRA + DPO fine-tune

[`scripts/finetune_flux_edit.py`](../scripts/finetune_flux_edit.py) runs
two stages:

**Stage 1 · supervised LoRA (SFT)** — rank-32 LoRA on the DiT
cross-attention blocks (`to_q, to_k, to_v, to_out.0`). Loss = MSE on
the rectified-flow velocity between source and target latents,
conditioned on the brief. ~6 h on a single H100.

**Stage 2 · reward-weighted DPO** (with `--dpo`) — for each pair we
construct preference triples (winner = teacher target, loser = source),
weight the preference by the observed health-score lift, and optimise
the standard DPO log-sigmoid loss with β=0.1. Refines the SFT adapter
without burning the base model's general capability.

Grounded in [Diffusion-DPO (Wallace et al. 2023)](https://arxiv.org/abs/2311.12092)
and [ImageReward (Xu et al. 2023)](https://arxiv.org/abs/2305.16381).
Our twist: the preference signal comes from our own ensemble's
health-score predictions, not from human raters.

### Inference

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.load_lora_weights("models/outputs/models/flux_edit_finetuned/lora_dpo")

result = pipe(prompt=ensemble_brief, image=source_image).images[0]
result.save("rebuilt.png")
```

In production the front-end skips the local Flux pipeline entirely and
calls the **Gemini 2.5 Flash Image** endpoint via OpenRouter — same
behaviour shape, zero GPU footprint. The local fine-tune is the swap-in
for offline / on-device / zero-per-request scenarios.

---

## Hardware notes

- **SmolVLM full FT**: ~22 GB VRAM at batch=4 with bf16 + grad-checkpointing.
  Fits on a single H100 80 GB or A100 80 GB. Drop to A6000 (48 GB) by
  reducing to batch=2.
- **Flux edit LoRA**: ~38 GB VRAM with batch=1 + bf16 + grad-checkpointing.
  Single H100 only. CPU dry-run with `--dry_run` validates the loop end-to-end
  on 4 samples without the full model footprint.
- **Synthetic-positive generation**: cost-bound, not compute-bound. About
  $0.10–$1.50 per 1,000 calls to Nano Banana on OpenRouter.

## File map (artefacts)

What's committed in `outputs/` vs what each trainer creates locally:

```
outputs/                                                              committed?
├── models/
│   ├── final/                    Model 1 — tabular ensemble           ✓ yes
│   ├── vlm_finetuned/            Model 2 — SmolVLM LoRA (legacy)      ✓ yes
│   ├── vlm_finetuned_full/       Model 2 — SmolVLM full FT            ✗ created by finetune_smolvlm_full.py
│   └── flux_edit_finetuned/      Model 3 — Flux edit LoRA + DPO       ✗ created by finetune_flux_edit.py
├── pseudo_labels/teacher_labels.jsonl   Gemma teacher pseudo-labels   ✓ yes
├── rubric/rubric_scores.parquet         15-dim LLM rubric             ✓ yes
└── flux_pairs/                          (src, brief, target) triples  ✗ created by generate_flux_pairs.py
    ├── manifest.jsonl
    └── images/<cid>__{source,target}.png
```

The two ✗-marked model dirs and `flux_pairs/` are written by their
trainers; we don't commit them because the SmolVLM full FT alone is
~9 GB and the Flux DiT adapter + 200 image pairs is another ~6 GB.

## Design decisions

- **Full fine-tune over LoRA for SmolVLM.** A small VLM on a small,
  consistent annotation task benefits more from updating every
  parameter than from a low-rank adapter. The legacy LoRA recipe is
  kept around for GPUs that can't fit the full FT.

- **Reward signal from our own ensemble, not human raters.** The
  ensemble already encodes "what works" as a numeric score, and we
  trust it. Using its health score as the DPO reward keeps the loop
  self-consistent and removes the cost of human labelling.

- **Rejection-sampling Nano Banana outputs.** The teacher is good but
  not infallible. We re-score every generated target with the same
  ensemble that built the brief and drop the pairs that don't actually
  improve. Bad pairs would teach the student bad habits.

- **Keep the third-party fallback path live.** The local Flux LoRA is
  the swap-in for offline / on-device scenarios; for the standalone
  SPA demo, the OpenRouter path renders the same flow at zero GPU
  cost and the same output schema.

- **Bricks the rebuild loop into Model 1's universe.** Anywhere we
  could've leaned on the VLM to grade quality, we leaned on the
  ensemble instead. It keeps Model 2 and Model 3 honest by giving
  them a single, stable source of truth.

---

[← 04 · Visual intelligence](04_visual_intelligence.md) · [↑ Index](../README.md) · [06 · Evaluation →](06_evaluation.md)
