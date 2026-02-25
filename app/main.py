"""
Main Gradio Application

Entry point for DermScreen AI. Defines the three-stage workflow UI
(Image Upload → Targeted Q&A → Triage Decision) and wires it to the inference engine.
"""
import logging

import gradio as gr
from dotenv import load_dotenv

from app.inference import MedGemmaInference, URGENCY_LABELS
from app.ui_components import (
    format_triage_badge,
    format_assessment_summary,
    format_referral_note,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Initialise model at startup ───────────────────────────────────────────────
try:
    model = MedGemmaInference()
except Exception as exc:
    logger.error("Failed to initialise MedGemma: %s", exc)
    model = None


# ── Stage handlers ────────────────────────────────────────────────────────────

def handle_stage_1(image, chw_notes: str):
    """
    Stage 1: assess the uploaded image and surface the 3 follow-up questions.

    Args:
        image: PIL image from the Gradio Image component.
        chw_notes: Free-text notes entered by the community health worker.

    Returns:
        Tuple of Gradio updates: assessment markdown, stage-2 visibility,
        three question-box label updates, and the assessment dict for state.
    """
    if model is None:
        msg = "*⚠️ Model failed to load — check console logs.*"
        return msg, gr.update(visible=False), gr.update(), gr.update(), gr.update(), None

    if image is None:
        return "*Please upload an image first.*", gr.update(visible=False), gr.update(), gr.update(), gr.update(), None

    assessment = model.assess_image(image, chw_notes or "No additional notes.")
    summary_md = format_assessment_summary(assessment)

    qs = assessment.get("clarifying_questions", [])
    q1 = qs[0] if len(qs) > 0 else "How long has this lesion been present?"
    q2 = qs[1] if len(qs) > 1 else "Does it itch, bleed, or crust?"
    q3 = qs[2] if len(qs) > 2 else "Has its size or colour changed recently?"

    return (
        summary_md,
        gr.update(visible=True),
        gr.update(label=q1, interactive=True),
        gr.update(label=q2, interactive=True),
        gr.update(label=q3, interactive=True),
        assessment,
    )


def handle_stage_3(assessment: dict, ans1: str, ans2: str, ans3: str):
    """
    Stage 3: synthesise Stage 1 assessment + patient answers into a triage note.

    Args:
        assessment: Parsed dict from Stage 1 (stored in gr.State).
        ans1/2/3: Patient answers to the three follow-up questions.

    Returns:
        Tuple: stage-3 column visibility update, badge HTML, referral note markdown.
    """
    if model is None or not assessment:
        return gr.update(visible=True), "<p>Model error — cannot generate triage.</p>", ""

    qs = assessment.get("clarifying_questions", ["Q1", "Q2", "Q3"])
    followup_answers = {
        qs[0] if len(qs) > 0 else "Q1": ans1,
        qs[1] if len(qs) > 1 else "Q2": ans2,
        qs[2] if len(qs) > 2 else "Q3": ans3,
    }

    triage_result = model.generate_triage_decision(assessment, followup_answers)

    urgency_raw  = triage_result.get("final_urgency", "yellow").lower()
    urgency_text = URGENCY_LABELS.get(urgency_raw, URGENCY_LABELS["yellow"])
    badge_html   = format_triage_badge(urgency_raw, urgency_text)
    referral_md  = format_referral_note(triage_result)

    return gr.update(visible=True), badge_html, referral_md


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks layout and wire up event handlers."""
    with gr.Blocks(title="DermScreen AI") as interface:

        gr.Markdown("# 🩺 DermScreen AI — Dermatology Triage for Community Health Workers")
        gr.Markdown(
            "> **⚠️ For clinical decision support only. Not a diagnostic device.**  \n"
            "> This tool guides community health workers through a structured "
            "three-stage triage pipeline powered by Google's MedGemma model."
        )

        assessment_state = gr.State(None)

        with gr.Row():
            # ── Left: Inputs ──────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## Stage 1 — Image Capture")
                image_input = gr.Image(
                    type="pil",
                    label="Skin Lesion Photo (Upload a file or use webcam)",
                )
                chw_notes = gr.Textbox(
                    lines=3,
                    placeholder="e.g. Patient has had this dark spot for 2 months, no pain.",
                    label="CHW Notes (optional context)",
                )
                assess_btn = gr.Button("🔍 Assess Image", variant="primary", size="lg")

                with gr.Column(visible=False, variant="panel") as stage_2_col:
                    gr.Markdown("---\n## Stage 2 — Targeted Follow-up")
                    gr.Markdown(
                        "*Ask the patient each question below and enter their answers.*"
                    )
                    q1_ans = gr.Textbox(label="Question 1", interactive=True)
                    q2_ans = gr.Textbox(label="Question 2", interactive=True)
                    q3_ans = gr.Textbox(label="Question 3", interactive=True)
                    triage_btn = gr.Button(
                        "📋 Generate Triage Decision", variant="secondary", size="lg"
                    )

            # ── Right: Outputs ────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## Findings")
                stage_1_output = gr.Markdown(
                    "*Upload an image and press **Assess Image** to see initial findings.*"
                )

                with gr.Column(visible=False) as stage_3_col:
                    gr.Markdown("---\n## Stage 3 — Triage Recommendation")
                    urgency_badge     = gr.HTML()
                    referral_note_out = gr.Markdown()

        # ── Events ───────────────────────────────────────────────────────────
        assess_btn.click(
            fn=handle_stage_1,
            inputs=[image_input, chw_notes],
            outputs=[
                stage_1_output,
                stage_2_col,
                q1_ans,
                q2_ans,
                q3_ans,
                assessment_state,
            ],
        )

        triage_btn.click(
            fn=handle_stage_3,
            inputs=[assessment_state, q1_ans, q2_ans, q3_ans],
            outputs=[stage_3_col, urgency_badge, referral_note_out],
        )

    return interface


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        theme=gr.themes.Soft(),
    )
