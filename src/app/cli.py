from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config.settings import DEFAULT_GEMINI_MODEL, Settings
from ..features.chat.gemini import GeminiChatService, GeminiConfig
from ..workflows.intake import IntakeConfig, IntakeWorkflow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SkeletonScan assessment with Gemini chatbot support.")
    parser.add_argument("--patient-name", required=True, help="Patient full name.")
    parser.add_argument(
        "--capture-duration",
        type=float,
        default=0.0,
        help="Seconds to capture motion per joint (set 0 to wait for manual completion).",
    )
    parser.add_argument("--output-dir", default="reports", help="Directory for generated PDF reports.")
    parser.add_argument("--camera-index", type=int, default=0, help="Video capture device index.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV preview windows.")
    parser.add_argument(
        "--model-name",
        help=f"Gemini model identifier (default: {DEFAULT_GEMINI_MODEL}). Overrides environment configuration.",
    )
    parser.add_argument("--api-key", help="Gemini API key override (defaults to GEMINI_API_KEY env variable).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    settings = Settings.from_env()

    api_key = args.api_key or settings.gemini_api_key
    model_name = args.model_name or settings.gemini_model_name

    try:
        chat_service = GeminiChatService(GeminiConfig(api_key=api_key, model_name=model_name))
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    workflow = IntakeWorkflow(chat_service)
    config = IntakeConfig(
        patient_name=args.patient_name,
        capture_duration=args.capture_duration,
        output_dir=Path(args.output_dir),
        camera_index=args.camera_index,
        display=not args.no_display,
    )

    try:
        report_path = workflow.run(config)
    except KeyboardInterrupt:
        print("\nAssessment cancelled.")
        return
    except (RuntimeError, ValueError) as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    print(f"\nReport generated: {report_path}")


if __name__ == "__main__":
    main()
