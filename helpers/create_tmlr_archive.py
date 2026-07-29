#!/usr/bin/env python3
"""Create an anonymized TMLR code archive from Git-visible files.

The archive contains files that are tracked or untracked-but-not-ignored by
Git. Files matching .gitignore are excluded even if they were committed before
the ignore rule was added. UNO/ is always excluded. The source repository is
never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable
import zipfile


TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cfg",
    ".csv",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# Build project-specific strings in pieces so the script does not flag itself.
DEFAULT_IDENTITY_TERMS = (
    "pin" + "ak",
    "gott" + "wald",
    "sydney" + ".edu",
    "University of " + "Sydney",
    "DP220" + "100931",
    "github.com/" + "pin" + "akm9",
    "/Users" + "/",
    "/home" + "/",
)
IDENTIFYING_REPOSITORY_URL = (
    "https://github.com/" + "pin" + "akm9/forget.git"
)
ALWAYS_EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
}
DATASET_PREFIXES = (
    Path("data/CelebA/dataset"),
    Path("data/CelebA/dataset-reconstructed"),
    Path("data/ImageNet/2012"),
    Path("data/ImageNet/DiT-XL-2"),
    Path("data/MNIST/MNIST"),
    Path("data/MNIST/MNIST-Experiments/MNIST"),
    Path("data/MNIST-138"),
    Path("data/tiny-imagenet-200"),
    Path("notebooks/data"),
    Path("notebooks/MNIST/data"),
)


class ArchiveError(RuntimeError):
    """Raised when the archive cannot safely be created."""


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Create a ZIP from Git-visible files, excluding every .gitignore "
            "match and the UNO directory."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root,
        help=f"Repository root (default: {repo_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ZIP path; defaults to a timestamped file beside the repo.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=20.0,
        help="Maximum size of an archived file in MiB (default: 20)",
    )
    parser.add_argument(
        "--include-notebooks",
        action="store_true",
        help=(
            "Not supported when honoring .gitignore because this repository "
            "ignores *.ipynb. Remove that rule first if notebooks are needed."
        ),
    )
    parser.add_argument(
        "--identity-term",
        action="append",
        default=[],
        metavar="TEXT",
        help="Additional identifying text to reject; may be repeated.",
    )
    parser.add_argument(
        "--skip-identity-check",
        action="store_true",
        help="Skip the identifying-text scan (not recommended).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output if it already exists.",
    )
    return parser.parse_args()


def timestamped_output(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root.parent / f"{root.name}_tmlr_code_{stamp}.zip"


def git_output(root: Path, arguments: list[str]) -> set[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise ArchiveError("git is required but was not found") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise ArchiveError(f"git failed: {message}") from error

    return {
        Path(value.decode("utf-8", errors="surrogateescape"))
        for value in result.stdout.split(b"\0")
        if value
    }


def archive_candidates(root: Path) -> list[Path]:
    # --exclude-standard applies all repository, global, and .git/info excludes
    # to untracked files. The second query removes tracked files that now match
    # an ignore rule, which normal `git ls-files --cached` would still return.
    visible = git_output(
        root,
        ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
    )
    tracked_but_ignored = git_output(
        root,
        ["ls-files", "-z", "-c", "-i", "--exclude-standard"],
    )

    candidates = []
    for relative in visible - tracked_but_ignored:
        if not relative.parts:
            continue
        if relative.parts[0] == "UNO":
            continue
        if any(
            name in ALWAYS_EXCLUDED_DIRECTORY_NAMES
            for name in relative.parts[:-1]
        ):
            continue
        if relative.suffix.lower() in {".pyc", ".pyo"}:
            continue
        if any(
            relative == prefix or prefix in relative.parents
            for prefix in DATASET_PREFIXES
        ):
            continue
        candidates.append(relative)
    return sorted(candidates)


def copy_candidates(
    root: Path,
    staged_root: Path,
    candidates: Iterable[Path],
    max_bytes: int,
) -> tuple[int, list[tuple[Path, str]]]:
    copied = 0
    skipped: list[tuple[Path, str]] = []
    for relative in candidates:
        source = root / relative
        if not source.is_file():
            skipped.append((relative, "not a regular file"))
            continue
        try:
            size = source.stat().st_size
        except OSError as error:
            skipped.append((relative, f"cannot stat: {error}"))
            continue
        if size > max_bytes:
            skipped.append((relative, "larger than size limit"))
            continue

        destination = staged_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied, skipped


def sanitize_repository_url(staged_root: Path) -> None:
    readme = staged_root / "README.md"
    if not readme.is_file():
        return
    text = readme.read_text(encoding="utf-8")
    text = text.replace(
        IDENTIFYING_REPOSITORY_URL,
        "<ANONYMIZED_REPOSITORY_URL>",
    )
    readme.write_text(text, encoding="utf-8")


def scan_for_identity(
    staged_root: Path, identity_terms: Iterable[str]
) -> list[tuple[Path, int, str, str]]:
    terms = [term for term in identity_terms if term]
    if not terms:
        return []
    pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
    findings: list[tuple[Path, int, str, str]] = []

    for path in sorted(staged_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as file:
                for line_number, line in enumerate(file, start=1):
                    match = pattern.search(line)
                    if match:
                        findings.append(
                            (
                                path.relative_to(staged_root),
                                line_number,
                                match.group(0),
                                line.strip()[:240],
                            )
                        )
        except OSError as error:
            raise ArchiveError(f"Cannot scan {path}: {error}") from error
    return findings


def create_zip(staged_parent: Path, staged_root: Path, output: Path) -> int:
    file_count = 0
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(staged_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staged_parent))
                file_count += 1
    return file_count


def format_mib(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MiB"


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise ArchiveError(f"Not a Git repository root: {root}")
    if args.max_size_mb <= 0:
        raise ArchiveError("--max-size-mb must be positive")
    if args.include_notebooks:
        raise ArchiveError(
            "--include-notebooks conflicts with the request to honor every "
            ".gitignore rule: .gitignore contains '*.ipynb'. Remove the flag."
        )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else timestamped_output(root)
    )
    if output.exists() and not args.force:
        raise ArchiveError(f"Output already exists; use --force: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    max_bytes = int(args.max_size_mb * 1024 * 1024)
    candidates = archive_candidates(root)

    with tempfile.TemporaryDirectory(prefix="tmlr_archive_") as temporary:
        staged_parent = Path(temporary)
        staged_root = staged_parent / root.name
        staged_root.mkdir()
        copied, skipped = copy_candidates(
            root, staged_root, candidates, max_bytes
        )
        sanitize_repository_url(staged_root)

        if not (staged_root / "modules").is_dir():
            raise ArchiveError("The staged archive is missing modules/")

        if not args.skip_identity_check:
            findings = scan_for_identity(
                staged_root,
                (*DEFAULT_IDENTITY_TERMS, *args.identity_term),
            )
            if findings:
                print("Identifying text remains in the archive:", file=sys.stderr)
                for path, line_number, match, line in findings[:100]:
                    print(
                        f"  {path}:{line_number}: {match!r}: {line}",
                        file=sys.stderr,
                    )
                if len(findings) > 100:
                    print(
                        f"  ... and {len(findings) - 100} more findings",
                        file=sys.stderr,
                    )
                raise ArchiveError(
                    "Remove or anonymize these values, or explicitly use "
                    "--skip-identity-check."
                )

        archived = create_zip(staged_parent, staged_root, output)

    archive_size = output.stat().st_size
    result_files = sum(
        1
        for path in candidates
        if path.parts and path.parts[0] == "data" and "Experiments" in str(path)
    )
    print(f"Created: {output}")
    print(f"Archive size: {format_mib(archive_size)}")
    if archive_size > 100 * 1024 * 1024:
        print(
            "warning: archive exceeds TMLR's 100 MiB supplementary-material limit",
            file=sys.stderr,
        )
    print(f"Files archived: {archived} (copied: {copied})")
    print(f"Experiment-result files selected: {result_files}")
    print(f"Files skipped by size or type: {len(skipped)}")
    for path, reason in skipped[:20]:
        print(f"  skipped {path}: {reason}")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")
    print("Notebooks excluded because .gitignore contains '*.ipynb'.")
    print("UNO/ excluded unconditionally.")
    print("Verify with:")
    print(f'  unzip -t "{output}"')
    print(f'  unzip -l "{output}" | head -40')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchiveError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
