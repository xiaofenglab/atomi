from pathlib import Path


def create_project(path: Path, code: str) -> Path:
    """Create a calculation directory with common subfolders."""
    path.mkdir(parents=True, exist_ok=True)
    for subdir in ("inputs", "runs", "analysis", "logs"):
        (path / subdir).mkdir(exist_ok=True)

    readme = path / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {path.name}\n\n"
            f"Code: `{code}`\n\n"
            "Suggested layout:\n\n"
            "- `inputs/`: source input files and structures\n"
            "- `runs/`: submitted calculation directories\n"
            "- `analysis/`: notebooks, parsed data, plots\n"
            "- `logs/`: scheduler and workflow logs\n",
            encoding="utf-8",
        )
    return path

