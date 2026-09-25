"""Code identity stamped on every artifact: the git state and a content hash of the evaluator.

The hash covers the git-less server copy, where `git_sha` is "unavailable".
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# The source that decides what a number means, hashed by content because `git_sha` is
# "unavailable" on the server's rsync tree. A new top-level dependency must be added here.
EVALUATOR_SOURCES: tuple[str, ...] = (
    "scoring/benchmark", "scoring/core", "scoring/config.py", "toygen", "metrics/rules",
    "metrics/__init__.py", "metrics/coverage.py", "metrics/independence_null.py",
    "metrics/joint_child.py", "metrics/outdegree.py", "metrics/reconstruction.py",
    "metrics/sibling_redundancy.py", "metrics/sres.py", "metrics/token_control.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evaluator_sha256(root: Path | None = None) -> str:
    """SHA-256 over every `.py` in `EVALUATOR_SOURCES`, path and bytes, sorted by path.

    Hashing the path makes a rename a change; `__pycache__` is skipped as machine-dependent.
    """
    r = Path(root) if root is not None else _repo_root()
    h = hashlib.sha256()
    files: list[Path] = []
    for rel in EVALUATOR_SOURCES:
        p = r / rel
        if p.is_dir():
            files += [q for q in p.rglob("*.py") if "__pycache__" not in q.parts]
        elif p.exists():
            files.append(p)
    for q in sorted(files, key=lambda x: str(x.relative_to(r))):
        h.update(str(q.relative_to(r)).encode())
        h.update(b"\0")
        h.update(q.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def git_provenance(cwd: Path) -> dict:
    """`{git_sha, git_dirty}`, or `"unavailable"` / `None` off a git checkout (e.g. the server)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return {"git_sha": "unavailable", "git_dirty": None}
