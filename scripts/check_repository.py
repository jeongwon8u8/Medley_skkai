"""Lightweight structural checks; no clinical data access."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = ["AGENTS.md", "START_HERE.md", "docs/ownership.md", "docs/contracts.md", "docs/git_workflow.md", ".github/pull_request_template.md", "pyproject.toml"]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")
print(f"Repository scaffold OK: {len(required)} required files found")
