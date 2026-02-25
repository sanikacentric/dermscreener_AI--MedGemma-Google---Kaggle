"""
App Inference Module

This module serves as the primary wrapper for the MedGemma model.
It handles multimodal zero-shot classification, feature extraction,
and question generation, as well as text-only triage determination.

MedGemma uses the PaliGemma processor / Gemma chat template.
Images must be passed in a structured messages dict, not inline text tokens.
"""
import os
import json
import logging
import re
from typing import Dict, Any, Optional

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Demo / Mock Mode ──────────────────────────────────────────────────────────
# Set DEMO_MODE=true in .env to skip live model inference and return
# realistic mock responses instantly. Great for CPU-only demo runs.
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "t")

_MOCK_ASSESSMENT = {
    "condition_category": "melanoma",
    "confidence": "high",
    "key_features_observed": [
        "Irregular, asymmetric border",
        "Multiple shades of brown and black",
        "Diameter >6 mm",
        "Surface irregularity with satellite lesions",
    ],
    "initial_urgency": "red",
    "clarifying_questions": [
        "How long have you had this lesion, and has it changed in shape or color?",
        "Does the lesion bleed, itch, or crust spontaneously?",
        "Do you have a personal or family history of melanoma or skin cancer?",
    ],
}

def _mock_triage_from_answers(initial_assessment: dict, followup_answers: dict) -> dict:
    """
    Rule-based triage for DEMO_MODE.
    Scores the patient's free-text answers to produce a realistic urgency level.
    """
    answers_blob = " ".join(followup_answers.values()).lower()

    # ── Duration signals ──────────────────────────────────────────────────────
    short_duration = any(w in answers_blob for w in [
        "day", "days", "week", "a week", "1 week", "one week", "few days",
        "yesterday", "recently", "just", "new",
    ])
    long_duration = any(w in answers_blob for w in [
        "month", "months", "year", "years", "long time", "long",
    ])

    # ── Symptom signals ───────────────────────────────────────────────────────
    has_bleed    = any(w in answers_blob for w in ["bleed", "bleeding", "blood"])
    has_itch     = any(w in answers_blob for w in ["itch", "itching", "itchy"])
    has_crust    = any(w in answers_blob for w in ["crust", "crusting", "scab"])
    has_growth   = any(w in answers_blob for w in ["grow", "growing", "larger", "bigger", "spread", "changed"])
    no_symptoms  = any(w in answers_blob for w in ["no ", "none", "nothing", "nope", "haven't", "not"])

    # ── Family / personal history ─────────────────────────────────────────────
    has_history  = any(w in answers_blob for w in [
        "family", "mother", "father", "parent", "sibling", "history",
        "cancer", "melanoma", "yes",
    ])
    no_history   = any(w in answers_blob for w in ["no family", "no history", "no cancer", "no"])

    # ── Score → urgency ───────────────────────────────────────────────────────
    score = 0
    score += 3 if has_bleed  else 0
    score += 2 if has_crust  else 0
    score += 1 if has_itch   else 0
    score += 2 if has_growth else 0
    score += 2 if has_history and not no_history else 0
    score += 3 if long_duration and not short_duration else 0
    score -= 2 if short_duration else 0
    score -= 1 if no_symptoms else 0
    score -= 1 if no_history and not has_history else 0

    initial_urgency = initial_assessment.get("initial_urgency", "yellow").lower()
    if initial_urgency == "red":
        score += 2  # image already flagged high-risk

    if score >= 5:
        urgency = "red"
        rationale = (
            "Multiple high-risk factors confirmed: the lesion shows ABCDE features "
            "and the patient's history (duration, symptoms, or family background) "
            "reinforces urgent concern. Immediate dermatologist review is required."
        )
        instructions = (
            "Do NOT apply any creams or cover the lesion. Avoid sun exposure. "
            "Seek urgent evaluation at a dermatology or skin cancer clinic within 24–48 hours."
        )
        chief = "Suspicious pigmented lesion with high-risk clinical and historical features."
        suspicion = f"{initial_assessment.get('condition_category', 'Melanoma').replace('_',' ').title()} — elevated risk based on combined image and history."
        timeline = "Duration and symptom profile indicate active progression."
        action = "Urgent dermatology referral within 24–48 hours. Consider excisional biopsy."
    elif score >= 2:
        urgency = "yellow"
        rationale = (
            "The lesion has some concerning visual features, but the patient's reported "
            "history suggests a moderate risk profile. A clinic review within the week "
            "is recommended for professional assessment."
        )
        instructions = (
            "Monitor the lesion and avoid picking or scratching it. Apply sunscreen when outdoors. "
            "Book a clinic appointment within 7 days for a professional skin check."
        )
        chief = "Pigmented skin lesion requiring clinical evaluation."
        suspicion = f"Possible {initial_assessment.get('condition_category', 'lesion').replace('_',' ')} — moderate concern; clinical correlation needed."
        timeline = "Short to moderate duration with limited symptomatic progression reported."
        action = "Non-urgent dermatology or GP referral within 7 days. Dermoscopy recommended."
    else:
        urgency = "green"
        rationale = (
            "The lesion is visually notable but the patient reports a short duration, "
            "no significant symptoms, and no relevant family history. Current evidence "
            "supports watchful waiting with routine follow-up."
        )
        instructions = (
            "Keep an eye on the lesion using the ABCDE rule (Asymmetry, Border, Colour, "
            "Diameter, Evolution). Take a photo today and compare in 2 weeks. "
            "Return if any changes are noticed."
        )
        chief = "Incidental pigmented lesion, low-risk profile based on history."
        suspicion = f"Likely benign; {initial_assessment.get('condition_category', 'lesion').replace('_',' ')} cannot be fully excluded without clinical exam."
        timeline = "Short duration, no symptoms, no family history of skin cancer."
        action = "Routine monitoring. Reassess in 2 weeks. Refer to GP if any change noted."

    return {
        "final_urgency": urgency,
        "triage_rationale": rationale,
        "patient_instructions": instructions,
        "referral_note": {
            "chief_complaint": chief,
            "clinical_suspicion": suspicion,
            "timeline_notes": timeline,
            "recommended_action": action,
        },
    }

# ── Constants ─────────────────────────────────────────────────────────────────
CONDITIONS = [
    "melanoma", "nevus", "basal_cell_carcinoma", "actinic_keratosis",
    "benign_keratosis", "dermatofibroma", "vascular_lesion", "squamous_cell_carcinoma"
]

URGENCY_LABELS = {
    "green":  "Monitor at home — reassess in 2 weeks",
    "yellow": "Clinic visit recommended within 7 days",
    "red":    "Urgent referral — seek care within 24–48 hours",
}

# Text of the system / user turns — image object is injected separately in the
# messages list, NOT embedded as a string token.
ASSESSMENT_USER_TEXT = (
    "You are a dermatology AI assistant helping a community health worker.\n"
    "Analyze the skin lesion image above together with the health worker's description:\n"
    "\"{chw_description}\"\n\n"
    "Reply ONLY with a JSON object using exactly these keys:\n"
    "- condition_category: one of [{conditions}]\n"
    "- confidence: high | medium | low\n"
    "- key_features_observed: list of 3-5 short visual observations\n"
    "- initial_urgency: green | yellow | red\n"
    "- clarifying_questions: list of exactly 3 targeted follow-up questions\n\n"
    "Output the raw JSON only — no markdown, no commentary."
)

TRIAGE_USER_TEXT = (
    "You are a dermatology AI assistant.\n"
    "Given the initial assessment and follow-up answers below, produce a final triage note.\n\n"
    "Initial Assessment:\n{initial_assessment}\n\n"
    "Follow-up Answers:\n{followup_answers}\n\n"
    "Reply ONLY with a JSON object using exactly these keys:\n"
    "- final_urgency: green | yellow | red\n"
    "- triage_rationale: short explanation\n"
    "- patient_instructions: plain-language instructions\n"
    "- referral_note: object with chief_complaint, clinical_suspicion, "
    "timeline_notes, recommended_action\n\n"
    "Output the raw JSON only — no markdown, no commentary."
)


class MedGemmaInference:
    """Wrapper for Google's MedGemma-4b-it model for dermatology triage."""

    def __init__(self) -> None:
        """Load processor + model. Supports 4-bit quantisation and CPU fallback."""
        if DEMO_MODE:
            logger.info("DEMO_MODE active — skipping model load.")
            self.model = None
            self.processor = None
            self.device = "cpu"
            return

        self.model_id = os.getenv("MODEL_ID", "google/medgemma-4b-it")
        self.use_4bit = os.getenv("USE_4BIT", "true").lower() in ("true", "1", "t")
        hf_token = os.getenv("HF_TOKEN")

        # Honour DEVICE env var but fall back to CUDA → CPU detection
        requested = os.getenv("DEVICE", "")
        if requested:
            self.device = requested
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading %s on %s (4-bit: %s)", self.model_id, self.device, self.use_4bit)

        quant_cfg: Optional[BitsAndBytesConfig] = None
        if self.use_4bit and self.device == "cuda":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, token=hf_token
            )

            model_kwargs: Dict[str, Any] = {"token": hf_token}
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            if quant_cfg:
                model_kwargs["quantization_config"] = quant_cfg

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **model_kwargs
            )

            if self.device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """
        Safely extract a JSON dict from the model's raw text output.

        Tries direct parse, then strips markdown fences, then falls back to
        brace-scanning. Never raises — returns an error dict on total failure.

        Args:
            raw: Raw decoded text from the model.

        Returns:
            Parsed dict, or {"error": ..., "raw": ...} on failure.
        """
        candidates = [raw.strip()]

        # Strip ```json ... ``` or ``` ... ``` fences
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            candidates.append(m.group(1).strip())

        # Brace scanning
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            candidates.append(raw[start : end + 1])

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue

        logger.warning("Could not parse JSON from model output.")
        return {"error": "Failed to parse model response", "raw": raw}

    def _build_messages(
        self, user_text: str, image: Optional[Image.Image] = None
    ) -> list:
        """
        Build a messages list in the Gemma chat-template format.

        For multimodal calls the image object is the first element of the
        user content list, followed by the text dict — this is what the
        MedGemma processor expects.

        Args:
            user_text: The text portion of the user turn.
            image: Optional PIL image for multimodal calls.

        Returns:
            List of message dicts compatible with apply_chat_template.
        """
        if image is not None:
            if image.mode != "RGB":
                image = image.convert("RGB")
            content = [{"type": "image", "image": image}, {"type": "text", "text": user_text}]
        else:
            content = [{"type": "text", "text": user_text}]

        return [{"role": "user", "content": content}]

    def _run_inference(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Core inference call — handles both multimodal (image+text) and text-only.

        Args:
            prompt: The user-turn text to send.
            image: Optional PIL image for multimodal calls.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Raw decoded string from the model (may contain JSON or plain text).
        """
        try:
            messages = self._build_messages(prompt, image)

            # apply_chat_template may or may not exist depending on transformers version
            if hasattr(self.processor, "apply_chat_template"):
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                # Fallback: use the processor directly (PaliGemma-style)
                if image is not None:
                    inputs = self.processor(
                        text=prompt, images=image, return_tensors="pt"
                    )
                else:
                    inputs = self.processor(text=prompt, return_tensors="pt")

            # Move tensors to the correct device
            target_device = (
                next(self.model.parameters()).device
                if hasattr(self.model, "parameters")
                else self.device
            )
            inputs = {k: v.to(target_device) for k, v in inputs.items()
                      if hasattr(v, "to")}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Slice off the prompt tokens so we only decode the completion
            input_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_len:]
            return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        except Exception as exc:
            logger.error("Inference run failed: %s", exc)
            return json.dumps({"error": "Inference failed", "detail": str(exc)})

    # ── Public API ────────────────────────────────────────────────────────────

    def assess_image(
        self, image: Image.Image, chw_description: str
    ) -> Dict[str, Any]:
        """
        Stage 1 — visual assessment of a skin lesion.

        Args:
            image: PIL image of the skin lesion.
            chw_description: CHW's free-text notes about the patient.

        Returns:
            Parsed assessment dict with condition, confidence, features,
            initial urgency, and 3 clarifying questions.
        """
        logger.info("Stage 1: Image Assessment")
        if DEMO_MODE:
            logger.info("DEMO_MODE active — returning mock assessment instantly.")
            return _MOCK_ASSESSMENT.copy()
        user_text = ASSESSMENT_USER_TEXT.format(
            chw_description=chw_description,
            conditions=", ".join(CONDITIONS),
        )
        raw = self._run_inference(user_text, image=image, max_new_tokens=512)
        return self._parse_json_response(raw)

    def generate_triage_decision(
        self,
        initial_assessment: Dict[str, Any],
        followup_answers: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Stage 3 — synthesise assessment + patient answers into a referral note.

        Args:
            initial_assessment: Parsed dict from Stage 1.
            followup_answers: Dict mapping each question to the patient's answer.

        Returns:
            Parsed triage dict with final urgency, rationale, instructions,
            and a structured referral note.
        """
        logger.info("Stage 3: Triage Decision Generation")
        if DEMO_MODE:
            logger.info("DEMO_MODE active — returning rule-based triage decision.")
            return _mock_triage_from_answers(initial_assessment, followup_answers)
        user_text = TRIAGE_USER_TEXT.format(
            initial_assessment=json.dumps(initial_assessment, indent=2),
            followup_answers=json.dumps(followup_answers, indent=2),
        )
        raw = self._run_inference(user_text, image=None, max_new_tokens=512)
        return self._parse_json_response(raw)
