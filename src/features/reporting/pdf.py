from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from ...core.models import PatientFeedback, SessionRecord


class PDFReportBuilder:
    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, patient_name: str, records: Iterable[SessionRecord]) -> Path:
        generated_at = datetime.now()
        c = canvas.Canvas(str(self.output_path), pagesize=letter)
        _, height = letter

        cursor_y = height - inch
        c.setFont("Helvetica-Bold", 18)
        c.drawString(inch, cursor_y, "Musculoskeletal Assessment Report")

        cursor_y -= 0.4 * inch
        c.setFont("Helvetica", 12)
        c.drawString(inch, cursor_y, f"Patient: {patient_name}")
        cursor_y -= 0.25 * inch
        c.drawString(inch, cursor_y, f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M')}")

        cursor_y -= 0.4 * inch
        for record in records:
            cursor_y = self._ensure_space(c, cursor_y, inch)
            cursor_y = self._draw_record(c, record, cursor_y)

        c.showPage()
        c.save()
        return self.output_path

    @staticmethod
    def _draw_record(c: canvas.Canvas, record: SessionRecord, cursor_y: float) -> float:
        metrics = record.joint_metrics
        feedback: PatientFeedback = record.patient_feedback

        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, cursor_y, metrics.joint_name)

        cursor_y -= 0.25 * inch
        c.setFont("Helvetica", 11)
        c.drawString(
            inch,
            cursor_y,
            f"Range of Motion (deg) | Min: {metrics.min_angle} | Max: {metrics.max_angle} | Avg: {metrics.mean_angle}",
        )

        if metrics.samples:
            cursor_y -= 0.2 * inch
            c.drawString(
                inch,
                cursor_y,
                f"Samples captured: {metrics.samples} | Duration: {metrics.duration:0.1f}s",
            )

        if feedback.response:
            cursor_y -= 0.25 * inch
            c.setFont("Helvetica-Oblique", 10)
            text = c.beginText(inch, cursor_y)
            text.textLines(f"Patient notes:\n{feedback.response}")
            c.drawText(text)
            cursor_y = text.getY() - 0.15 * inch

        if feedback.summary:
            c.setFont("Helvetica", 10)
            text = c.beginText(inch, cursor_y)
            text.textLines(f"Clinician summary:\n{feedback.summary}")
            c.drawText(text)
            cursor_y = text.getY() - 0.15 * inch

        return cursor_y - 0.3 * inch

    @staticmethod
    def _ensure_space(c: canvas.Canvas, cursor_y: float, margin: float) -> float:
        if cursor_y < margin * 2:
            c.showPage()
            cursor_y = letter[1] - margin
        return cursor_y
