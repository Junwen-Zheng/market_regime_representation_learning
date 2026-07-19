from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_release_audit_passes():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_release.py",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "tracked_text_files_clean: True" in output
    assert "tracked_raw_market_data: False" in output
