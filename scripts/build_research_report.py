from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPOSITORY_ROOT),
)

from src.research_report import build_research_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the reviewer-facing market-regime "
            "research report."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real"),
    )

    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            "docs/reports/"
            "market_regime_research_report.html"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = build_research_report(
        output_dir=args.output_dir,
        destination=args.destination,
    )

    print("Research report generated.")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
