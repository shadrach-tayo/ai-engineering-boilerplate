"""Run the RAG generator eval with a searchable Confident AI identifier."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

EVAL_FILE = Path(__file__).parent / "tests" / "test_faithfulness.py"


def _slug(value: str) -> str:
    """Normalize identifier components for consistent Confident AI searches."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def _git_branch() -> str:
    """Return the checked-out branch without failing outside a Git worktree."""
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "local"


def build_identifier(
    experiment: str,
    *,
    environ: Mapping[str, str] | None = None,
    branch: str | None = None,
) -> str:
    """Build a stable experiment identifier from an override, PR, or branch."""
    env = os.environ if environ is None else environ
    if override := env.get("DEEPEVAL_IDENTIFIER"):
        return override

    ref = env.get("GITHUB_REF", "")
    pr_match = re.fullmatch(r"refs/pull/(\d+)/merge", ref)
    change = (
        f"pr-{pr_match.group(1)}"
        if pr_match
        else env.get("GITHUB_HEAD_REF")
        or env.get("CI_MERGE_REQUEST_SOURCE_BRANCH_NAME")
        or env.get("BRANCH_NAME")
        or branch
        or _git_branch()
    )
    return f"rag-{_slug(experiment)}-{_slug(change)}"


def main(argv: Sequence[str] | None = None) -> None:
    """Run DeepEval and forward remaining arguments to its test runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="generator-quality",
        help="Stable name for the change or experiment under evaluation.",
    )
    parser.add_argument(
        "--identifier",
        help="Explicit Confident AI run identifier; overrides automatic naming.",
    )
    args, deepeval_args = parser.parse_known_args(argv)

    env = dict(os.environ)
    if args.identifier:
        env["DEEPEVAL_IDENTIFIER"] = args.identifier
    identifier = build_identifier(args.experiment, environ=env)
    command = [
        str(Path(sys.executable).with_name("deepeval")),
        "test",
        "run",
        str(EVAL_FILE),
        "--identifier",
        identifier,
        *deepeval_args,
    ]
    raise SystemExit(subprocess.run(command, check=False, env=env).returncode)


if __name__ == "__main__":
    main()
