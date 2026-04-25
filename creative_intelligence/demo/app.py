"""
Smadex Creative Intelligence Demo — Gradio App.

Tabs:
  1. Creative Analyzer: score + SHAP + VLM explanation
  2. Fatigue Monitor: campaign-level fatigue dashboard
  3. Creative Recommender: brief generation + similar top-performers
  4. Campaign Dashboard: portfolio performance overview
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from PIL import Image

from src.data.loader import DataLoader
from src.inference.pipeline import CreativeIntelligencePipeline

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

print("Loading data and CLIP embeddings...")
pipeline = CreativeIntelligencePipeline(CONFIG_PATH)
pipeline._ensure_data()
print("Data ready.")

master_df = pipeline._master_df
daily_df = pipeline._daily_df
loader = pipeline.data_loader

STATUS_COLORS = {
    "top_performer": "#22c55e",
    "stable": "#3b82f6",
    "fatigued": "#f59e0b",
    "underperformer": "#ef4444",
}


def get_creative_ids() -> list:
    return sorted(master_df["creative_id"].tolist())


def get_campaign_ids() -> list:
    return sorted(master_df["campaign_id"].unique().tolist())


# ── Tab 1: Creative Analyzer ──────────────────────────────────────────────────

def analyze_creative(creative_id: int):
    try:
        pipeline._ensure_models()
        report = pipeline.analyze_creative(int(creative_id))
    except Exception as e:
        return None, f"Error: {e}", "{}", "{}", ""

    # Creative image
    img_path = loader.get_asset_path(int(creative_id))
    img = Image.open(img_path) if img_path.exists() else None

    # SHAP bar chart
    shap = report.shap_top_features
    feat_names = list(shap.keys())
    shap_vals = list(shap.values())
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in shap_vals]
    fig_shap, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_names[::-1], shap_vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Creative {creative_id} — SHAP Feature Contributions")
    ax.set_xlabel("SHAP value (impact on performance score)")
    plt.tight_layout()

    status_color = STATUS_COLORS.get(report.predicted_status, "#6b7280")
    summary_md = f"""
### Creative {creative_id} Analysis

**Performance Score:** `{report.perf_score:.3f}` / 1.0

**Predicted Status:** <span style="color:{status_color}; font-weight:bold">{report.predicted_status.upper()}</span>

**Status Probabilities:**
{chr(10).join(f"- {k}: {v:.1%}" for k, v in report.status_probabilities.items())}

**Fatigue Risk:**
- Probability: `{report.fatigue_risk.get('fatigue_probability', 0):.1%}`
- Current fatigue score: `{report.fatigue_risk.get('current_fatigue_score', 0):.2f}`
- Days active: `{report.fatigue_risk.get('days_active', 0)}`
"""

    vlm_md = ""
    if report.vlm_analysis:
        v = report.vlm_analysis
        vlm_md = f"""
**Performance Summary:** {v.get('performance_summary', 'N/A')}

**Visual Strengths:** {', '.join(v.get('visual_strengths', []))}

**Visual Weaknesses:** {', '.join(v.get('visual_weaknesses', []))}

**Fatigue Risk Reason:** {v.get('fatigue_risk_reason', 'N/A')}

**Top Recommendation:** _{v.get('top_recommendation', 'N/A')}_
"""
    else:
        vlm_md = "_VLM model not loaded. Run scripts/run_pipeline.py to finetune._"

    recs_md = "\n".join(
        f"- **{r['feature']}**: {r['rationale']}" for r in report.recommendations[:5]
    ) or "_No feature recommendations available._"

    return img, summary_md, fig_shap, vlm_md, recs_md


# ── Tab 2: Fatigue Monitor ────────────────────────────────────────────────────

def monitor_campaign(campaign_id: int):
    pipeline._ensure_models()
    df = pipeline.monitor_campaign(int(campaign_id))

    fig = go.Figure()
    for _, row in df.iterrows():
        color = STATUS_COLORS.get(row["creative_status"], "#6b7280")
        fig.add_bar(
            x=[str(row["creative_id"])],
            y=[row["fatigue_probability"]],
            name=str(row["creative_id"]),
            marker_color=color,
            showlegend=False,
        )
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Fatigue threshold")
    fig.update_layout(
        title=f"Campaign {campaign_id} — Creative Fatigue Risk",
        xaxis_title="Creative ID", yaxis_title="Fatigue Probability", yaxis_range=[0, 1],
        height=400,
    )

    summary = df[["creative_id", "creative_status", "overall_ctr",
                  "fatigue_probability", "current_fatigue_score", "days_active"]]
    return fig, summary


def show_fatigue_curve(campaign_id: int, creative_id: int):
    pipeline._ensure_models()
    curve = pipeline._fatigue_detector.compute_fatigue_curve(int(creative_id), pipeline._daily_df)
    if curve.empty:
        return go.Figure().update_layout(title="No daily data available")

    fig = go.Figure()
    fig.add_scatter(x=curve["days_since_launch"], y=curve["ctr"],
                    name="Daily CTR", line=dict(color="#3b82f6"))
    fig.add_scatter(x=curve["days_since_launch"], y=curve["rolling_ctr_3d"],
                    name="3-day rolling CTR", line=dict(color="#6366f1", dash="dot"))
    fig.add_scatter(x=curve["days_since_launch"], y=curve["fatigue_score"],
                    name="Fatigue Score", line=dict(color="#ef4444"), yaxis="y2")
    fig.update_layout(
        title=f"Creative {creative_id} — CTR Decay & Fatigue",
        xaxis_title="Days since launch",
        yaxis_title="CTR",
        yaxis2=dict(title="Fatigue Score", overlaying="y", side="right", range=[0, 1]),
        height=400,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ── Tab 3: Recommender ────────────────────────────────────────────────────────

def recommend(creative_id: int):
    pipeline._ensure_models()
    try:
        report = pipeline.analyze_creative(int(creative_id))
    except Exception as e:
        return f"Error: {e}", None, []

    brief = report.creative_brief
    brief_md = f"""
### Creative Brief for replacement of Creative {creative_id}

**Keep:**
{chr(10).join(f"- {k}" for k in brief.get('keep', []))}

**Change:**
{chr(10).join(f"- {c}" for c in brief.get('change', []))}

**Top Recommendation:** _{brief.get('top_recommendation', 'N/A')}_

**VLM Rationale:** {brief.get('vlm_rationale', 'N/A')}

**Visual Weaknesses to Fix:** {', '.join(brief.get('visual_weaknesses', []))}
"""

    similar = pipeline._recommender.retrieve_similar_top_performers(int(creative_id))
    gallery_images = []
    for s in similar[:5]:
        p = loader.get_asset_path(s["creative_id"])
        if p.exists():
            img = Image.open(p)
            gallery_images.append((img, f"#{s['creative_id']} | CTR={s['ctr']:.4f} | {s['status']}"))

    similar_md = "\n".join(
        f"- **{s['creative_id']}** — similarity={s['similarity']:.3f}, CTR={s['ctr']:.4f}, status={s['status']}"
        for s in similar
    )

    return brief_md, gallery_images, similar_md


# ── Tab 4: Campaign Dashboard ─────────────────────────────────────────────────

def campaign_dashboard(campaign_id: int):
    pipeline._ensure_data()
    creatives = master_df[master_df["campaign_id"] == int(campaign_id)].copy()
    if creatives.empty:
        return go.Figure(), go.Figure()

    creatives["status_color"] = creatives["creative_status"].map(STATUS_COLORS)

    # Bar chart: perf_score by creative
    engineer = pipeline.feature_engineer
    y_perf = engineer.get_perf_scores(creatives)
    creatives = creatives.copy()
    creatives["computed_perf_score"] = y_perf

    fig_bar = px.bar(
        creatives, x="creative_id", y="computed_perf_score",
        color="creative_status",
        color_discrete_map=STATUS_COLORS,
        title=f"Campaign {campaign_id} — Creative Performance Scores",
        labels={"computed_perf_score": "Performance Score", "creative_id": "Creative ID"},
        height=400,
    )

    # Scatter: CTR vs IPM
    fig_scatter = px.scatter(
        creatives, x="overall_ctr", y="overall_ipm",
        color="creative_status",
        color_discrete_map=STATUS_COLORS,
        hover_data=["creative_id", "format", "theme"],
        title=f"Campaign {campaign_id} — CTR vs IPM",
        labels={"overall_ctr": "CTR", "overall_ipm": "IPM"},
        height=400,
    )

    return fig_bar, fig_scatter


# ── Gradio App ────────────────────────────────────────────────────────────────

creative_ids = get_creative_ids()
campaign_ids = get_campaign_ids()

with gr.Blocks(title="Smadex Creative Intelligence", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Smadex Creative Intelligence Platform")
    gr.Markdown(
        "Powered by CLIP + XGBoost + SmolVLM with Self-Distillation Fine-Tuning (SDFT) for continual learning."
    )

    with gr.Tabs():

        with gr.Tab("Creative Analyzer"):
            with gr.Row():
                cid_input = gr.Dropdown(
                    choices=creative_ids, label="Select Creative ID", value=creative_ids[0]
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Row():
                img_out = gr.Image(label="Creative Asset", width=360)
                summary_out = gr.Markdown()
            shap_out = gr.Plot(label="SHAP Feature Contributions")
            vlm_out = gr.Markdown(label="VLM Analysis")
            recs_out = gr.Markdown(label="Feature Recommendations")

            analyze_btn.click(
                fn=analyze_creative,
                inputs=[cid_input],
                outputs=[img_out, summary_out, shap_out, vlm_out, recs_out],
            )

        with gr.Tab("Fatigue Monitor"):
            with gr.Row():
                camp_input_fat = gr.Dropdown(
                    choices=campaign_ids, label="Select Campaign ID", value=campaign_ids[0]
                )
                monitor_btn = gr.Button("Monitor Campaign", variant="primary")
            fatigue_bar_out = gr.Plot(label="Fatigue Risk by Creative")
            fatigue_table_out = gr.Dataframe(label="Creative Fatigue Summary")

            with gr.Row():
                cid_input_curve = gr.Dropdown(
                    choices=creative_ids, label="Select Creative for Decay Curve"
                )
                curve_btn = gr.Button("Show Decay Curve")
            fatigue_curve_out = gr.Plot(label="CTR Decay & Fatigue Score")

            monitor_btn.click(
                fn=monitor_campaign,
                inputs=[camp_input_fat],
                outputs=[fatigue_bar_out, fatigue_table_out],
            )
            curve_btn.click(
                fn=show_fatigue_curve,
                inputs=[camp_input_fat, cid_input_curve],
                outputs=[fatigue_curve_out],
            )

        with gr.Tab("Creative Recommender"):
            with gr.Row():
                cid_rec_input = gr.Dropdown(
                    choices=creative_ids, label="Select Creative to Replace", value=creative_ids[0]
                )
                rec_btn = gr.Button("Generate Brief", variant="primary")
            brief_out = gr.Markdown(label="Creative Brief")
            gallery_out = gr.Gallery(label="Similar Top Performers", columns=5)
            similar_md_out = gr.Markdown(label="Similar Creatives (text)")

            rec_btn.click(
                fn=recommend,
                inputs=[cid_rec_input],
                outputs=[brief_out, gallery_out, similar_md_out],
            )

        with gr.Tab("Campaign Dashboard"):
            with gr.Row():
                camp_input_dash = gr.Dropdown(
                    choices=campaign_ids, label="Select Campaign ID", value=campaign_ids[0]
                )
                dash_btn = gr.Button("Load Dashboard", variant="primary")
            with gr.Row():
                perf_bar_out = gr.Plot(label="Performance Scores")
                scatter_out = gr.Plot(label="CTR vs IPM")

            dash_btn.click(
                fn=campaign_dashboard,
                inputs=[camp_input_dash],
                outputs=[perf_bar_out, scatter_out],
            )


if __name__ == "__main__":
    demo_cfg = CFG.get("demo", {})
    demo.launch(
        server_name=demo_cfg.get("host", "0.0.0.0"),
        server_port=demo_cfg.get("port", 7860),
        share=demo_cfg.get("share", False),
    )
