"""Export paper markdown to PDF via pandoc.

Usage:
    python scripts/export_paper_pdf.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    """Export `papers/full_paper.md` to `papers/full_paper.pdf`.

    Returns:
        Exit code, 0 on success.
    """
    source = Path("papers/full_paper.md")
    target = Path("papers/full_paper.pdf")
    if not source.exists():
        print(f"Missing source file: {source}")
        return 1

    command = [
        "pandoc",
        str(source),
        "-o",
        str(target),
        "--pdf-engine=xelatex",
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        print("pandoc not found. Install pandoc (and xelatex) to export PDF.")
        return 1

    if result.returncode != 0:
        print(result.stderr.strip() or "Failed to export PDF.")
        return result.returncode

    print(f"Saved: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
