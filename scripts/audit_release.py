from __future__ import annotations

from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_FRAGMENTS = (
    "/" + "Users" + "/",
    "junwen" + "zheng",
    "file:" + "//",
    "BEGIN " + "OPENSSH PRIVATE KEY",
    "github_" + "pat_",
    "gh" + "p_",
)


def repository_files() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return [
        line
        for line in result.stdout.splitlines()
        if line
    ]


def find_text_violations(
    filenames: list[str],
) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []

    for filename in filenames:
        path = REPOSITORY_ROOT / filename

        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for fragment in FORBIDDEN_FRAGMENTS:
            if fragment in text:
                violations.append(
                    (filename, fragment)
                )

    return violations


def tracked_raw_data_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "data/raw"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return [
        line
        for line in result.stdout.splitlines()
        if line != "data/raw/.gitkeep"
    ]


def main() -> None:
    violations = find_text_violations(
        repository_files()
    )

    for filename, fragment in violations:
        print(
            f"{filename}: contains "
            f"{fragment!r}"
        )

    raw_files = tracked_raw_data_files()

    for filename in raw_files:
        print(
            "Unexpected tracked raw-data file: "
            f"{filename}"
        )

    if violations or raw_files:
        raise SystemExit("Release audit failed")

    print("tracked_text_files_clean: True")
    print("tracked_raw_market_data: False")


if __name__ == "__main__":
    main()
