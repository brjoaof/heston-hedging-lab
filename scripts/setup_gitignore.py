#!/usr/bin/env python3
"""Safely add Git ignore rules without replacing an existing .gitignore.

Idempotent: run once before the first commit and again only if rules change.
The generated .gitignore itself is versioned; experiment outputs are not.
"""
from pathlib import Path

BEGIN = "# >>> Heston hedging lab: generated files (managed) >>>"
END = "# <<< Heston hedging lab: generated files (managed) <<<"
RULES = """# Local Python environments and caches
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.coverage
htmlcov/
*.egg-info/
build/
dist/

# Simulation pools, evaluations, model/recovery snapshots
/results/
/data/
/datasets/
/checkpoints/
/artifacts/
/logs/
*.log
*.npz
*.npy
*.pkl
*.pt
*.pth
*.ckpt
*.zip
*.patch

# Local secrets and editor state
.env
.env.*
!.env.example
.idea/
.vscode/
.DS_Store
Thumbs.db"""


def update_gitignore(path: Path) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{BEGIN}\n{RULES}\n{END}"
    if (BEGIN in existing) != (END in existing):
        raise ValueError("Inconsistent managed block in .gitignore")
    if BEGIN in existing:
        before, remainder = existing.split(BEGIN, 1)
        _, after = remainder.split(END, 1)
        updated = before + block + after
    else:
        updated = existing.rstrip("\n") + ("\n\n" if existing.strip() else "") + block + "\n"
    if updated == existing:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / ".gitignore"
    changed = update_gitignore(target)
    print(f"{'Updated' if changed else 'Already current'}: {target}")
