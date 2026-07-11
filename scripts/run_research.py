from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.workflow import run_research_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the market-regime research workflow."
    )

    parser.add_argument(
        "--data-mode",
        choices=["real", "synthetic_smoke_test"],
        default="real",
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "raw" / "prices.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
    )

    parser.add_argument(
        "--regime-mode",
        choices=[
            "walk_forward",
            "full_sample_diagnostic",
        ],
        default="walk_forward",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-assets", type=int, default=120)
    parser.add_argument("--n-days", type=int, default=760)
    parser.add_argument("--n-sectors", type=int, default=8)
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--refit-frequency", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_data_path = (
        args.data_path
        if args.data_mode == "real"
        else None
    )

    outputs = run_research_pipeline(
        output_dir=args.output_dir,
        data_mode=args.data_mode,
        data_path=selected_data_path,
        seed=args.seed,
        n_assets=args.n_assets,
        n_days=args.n_days,
        n_sectors=args.n_sectors,
        n_regimes=args.n_regimes,
        pca_components=args.pca_components,
        regime_mode=args.regime_mode,
        min_train_days=args.min_train_days,
        refit_frequency=args.refit_frequency,
    )

    print("Research pipeline completed.")
    print(
        outputs["research_summary"].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
