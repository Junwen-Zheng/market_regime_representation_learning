from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.workflow import run_research_pipeline  # noqa: E402


if __name__ == "__main__":
    outputs = run_research_pipeline(output_dir=ROOT / "outputs")
    summary = outputs["research_summary"]
    print("Research pipeline completed.")
    print(summary.to_string(index=False))
