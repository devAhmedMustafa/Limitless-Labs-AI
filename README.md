# Skeleton Scan Concierge

A small command-line assistant that captures joint range of motion with Mediapipe, asks follow-up questions through the Gemini API, and wraps everything into a clinician-ready PDF report.

---

## Quick start

1. **Clone the repo**
   ```powershell
   git clone https://github.com/devAhmedMustafa/Limitless-Labs-AI.git
   cd Limitless-Labs-AI
   ```

2. **Create a virtual environment** (optional but recommended)
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Add your Gemini API key**
   - Copy `.env.example` to `.env`
   - Fill in `GEMINI_API_KEY=your_real_key`

5. **Run the intake workflow**
   ```powershell
   python -m src.app.cli --patient-name "Jane Doe"
   ```
   - The preview window opens for each joint.
   - When the patient finishes a movement, press **`n`** (or Enter) in the window to move on.
   - Type the patient’s response when prompted in the terminal.

6. **Check the generated report**
   - A PDF is saved under the `reports/` folder, named after the patient (for example `jane_doe_assessment.pdf`).

---

## Common options

- `--camera-index` &mdash; choose a different webcam (default is `0`).
- `--no-display` &mdash; hide the preview window and rely on a fixed duration (set `--capture-duration`).
- `--model-name` &mdash; override the Gemini model (default `gemini-2.5-flash-lite`).

Run `python -m src.app.cli --help` to see every flag.

---

## Project structure

```
src/
  app/          # CLI entry point
  config/       # environment & settings helpers
  core/         # shared math utilities and data models
  features/
    skeleton_scan/  # Mediapipe pose analyzer
    chat/           # Gemini API wrapper
    reporting/      # PDF builder
  workflows/    # intake orchestration
```

---

## Troubleshooting

- **No webcam window**: make sure you run from the project root and the camera isn’t in use by another app.
- **Gemini request failed**: confirm your API key is valid and has access to the model; try `--model-name gemini-2.5-flash-lite` explicitly.
- **PDF looks empty**: ensure you pressed `n` after moving the joint so data is recorded.

Feel free to open issues or tweak the workflow to add more joints, switch models, or integrate a GUI.
