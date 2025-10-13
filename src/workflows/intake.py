from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..core.models import JointMetrics, PatientFeedback, QuestionSet, SessionRecord
from ..features.chat.gemini import GeminiChatService
from ..features.reporting.pdf import PDFReportBuilder
from ..features.skeleton_scan.analyzer import PoseAnalyzer
from ..features.skeleton_scan.joints import DEFAULT_JOINTS, JointDefinition


@dataclass
class IntakeConfig:
    patient_name: str
    output_dir: Path
    camera_index: int
    display: bool
    capture_duration: float = 0.0


class IntakeWorkflow:
    def __init__(
        self,
        chat_service: GeminiChatService,
        analyzer_factory=lambda **kwargs: PoseAnalyzer(**kwargs),
    ) -> None:
        self.chat_service = chat_service
        self.analyzer_factory = analyzer_factory

    def run(self, config: IntakeConfig) -> Path:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{self._slugify(config.patient_name)}_assessment.pdf"

        session_records: List[SessionRecord] = []
        with self.analyzer_factory(camera_index=config.camera_index, display=config.display) as analyzer:
            for joint in DEFAULT_JOINTS:
                record = self._assess_joint(analyzer, joint, config.capture_duration)
                session_records.append(record)

        builder = PDFReportBuilder(report_path)
        builder.build(config.patient_name, session_records)
        return report_path

    def _assess_joint(self, analyzer: PoseAnalyzer, joint: JointDefinition, duration: float) -> SessionRecord:
        input(
            f"\n--- {joint.name} ---\nPress Enter when the patient is ready to begin the movement. "
            "When finished, press 'n' in the preview window to continue..."
        )
        metrics = analyzer.capture_joint_range(joint, duration_seconds=duration)

        if not metrics.samples:
            questions = QuestionSet(prompt="Describe any discomfort you experienced during the movement.")
            patient_feedback = PatientFeedback(response="", summary="No motion data captured.")
            return SessionRecord(joint_metrics=metrics, questions=questions, patient_feedback=patient_feedback)

        try:
            questions = self.chat_service.generate_questions(metrics)
        except RuntimeError as exc:
            print(f"Gemini question generation failed: {exc}")
            questions = QuestionSet(prompt="Describe any discomfort you experienced during the movement.")

        print("\nChatbot questions:")
        print(questions.prompt)
        patient_response = input("\nPatient response: ").strip()

        if not patient_response:
            patient_feedback = PatientFeedback(response="", summary="No patient feedback provided.")
        else:
            try:
                patient_feedback = self.chat_service.summarize_feedback(metrics, patient_response)
            except RuntimeError as exc:
                print(f"Gemini summarization failed: {exc}")
                patient_feedback = PatientFeedback(response=patient_response, summary=patient_response)

        return SessionRecord(joint_metrics=metrics, questions=questions, patient_feedback=patient_feedback)

    @staticmethod
    def _slugify(value: str) -> str:
        sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
        return sanitized.strip("_") or "patient"
