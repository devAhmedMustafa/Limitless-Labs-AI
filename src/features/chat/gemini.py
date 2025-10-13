from __future__ import annotations

from dataclasses import dataclass

import google.generativeai as genai

from ...core.models import JointMetrics, PatientFeedback, QuestionSet
from ...config.settings import DEFAULT_GEMINI_MODEL


@dataclass
class GeminiConfig:
    api_key: str
    model_name: str = DEFAULT_GEMINI_MODEL


class GeminiChatService:
    def __init__(self, config: GeminiConfig) -> None:
        if not config.api_key:
            raise ValueError("Gemini API key is required")
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_name)

    def generate_questions(self, metrics: JointMetrics) -> QuestionSet:
        prompt = (
            "You are a conversational physical therapy intake assistant. "
            "Generate three concise questions to ask a patient about their discomfort after completing a range of motion assessment. "
            "Each question should reference the joint and the motion metrics provided. Return the questions as a numbered list."
        )
        context = self._metrics_summary(metrics)
        try:
            response = self.model.generate_content(f"{prompt}\n\n{context}")
        except Exception as exc:  # pragma: no cover - external dependency
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc
        text = getattr(response, "text", None) or "Describe any discomfort you experienced."
        return QuestionSet(prompt=text.strip())

    def summarize_feedback(self, metrics: JointMetrics, patient_response: str) -> PatientFeedback:
        prompt = (
            "Summarize the patient's feedback for the clinician with three short bullet points covering pain characteristics, triggers, and functional limitations."
        )
        context = self._metrics_summary(metrics)
        try:
            response = self.model.generate_content(
                f"{prompt}\n\nJoint: {metrics.joint_name}\nMetrics: {context}\nPatient response: {patient_response}"
            )
        except Exception as exc:  # pragma: no cover - external dependency
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc
        summary = getattr(response, "text", None) or patient_response
        return PatientFeedback(response=patient_response.strip(), summary=summary.strip())

    @staticmethod
    def _metrics_summary(metrics: JointMetrics) -> str:
        return (
            f"Joint assessed: {metrics.joint_name}. "
            f"Samples captured: {metrics.samples}. "
            f"Minimum angle: {metrics.min_angle}. Maximum angle: {metrics.max_angle}. Average angle: {metrics.mean_angle}."
        )
