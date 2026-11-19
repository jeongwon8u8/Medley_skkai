"""Repository-local destinations for private training artifacts."""
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[2]
PRIVATE_ROOTS = tuple(
    base / directory
    for base in (REPOSITORY, REPOSITORY / "medley_pd")
    for directory in ("outputs", "checkpoints")
)


def is_ignored_private_artifact(path: Path) -> bool:
    return (
        any(root in path.parents for root in PRIVATE_ROOTS)
        and subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)], cwd=REPOSITORY,
        ).returncode == 0
    )
