"""Fail if dataset files or data-derived artifacts are about to be shipped.

Used in two places:

* ``python tools/check_no_private_data.py --git`` checks files staged or
  tracked by git (run before ``git commit`` / ``git push``).
* ``python tools/check_no_private_data.py /app`` checks a directory tree
  (run during ``docker build`` so the image build fails on a leak).

Detection is by content as well as name: an NPZ is a ZIP archive, so a copy
saved under a strange name (for example ``data%2E``) is still caught.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

BLOCKED_SUFFIXES = {
    ".npz",
    ".npy",
    ".duckdb",
    ".sqlite3",
    ".joblib",
    ".pkl",
    ".zip",
}
BLOCKED_DIRS = {"data", "output"}
ALLOWED_NAMES = {".gitkeep"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}


def problems_for(path: Path, relative: Path) -> list[str]:
    reasons = []
    if path.name in ALLOWED_NAMES:
        return reasons
    if relative.parts and relative.parts[0] in BLOCKED_DIRS:
        reasons.append(f"inside generated folder '{relative.parts[0]}/'")
    suffix = "".join(path.suffixes[-1:]).lower()
    if suffix in BLOCKED_SUFFIXES or ".sqlite3" in path.name.lower():
        reasons.append(f"blocked file type '{suffix or path.name}'")
    try:
        if path.is_file() and zipfile.is_zipfile(path):
            reasons.append("ZIP/NPZ archive content")
    except OSError:
        pass
    return reasons


def git_files(root: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return [Path(p) for p in output.decode().split("\0") if p]


def tree_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_file():
            files.append(rel)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument(
        "--git", action="store_true", help="check files tracked or staged by git"
    )
    args = parser.parse_args()
    root = args.root.resolve()

    files = git_files(root) if args.git else tree_files(root)
    leaks = []
    for rel in files:
        reasons = problems_for(root / rel, rel)
        if reasons:
            leaks.append(f"  {rel}: {', '.join(reasons)}")

    if leaks:
        print("Refusing to ship possible private data:", file=sys.stderr)
        print("\n".join(leaks), file=sys.stderr)
        return 1

    print(f"OK: {len(files)} files checked, no dataset or derived artifacts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
