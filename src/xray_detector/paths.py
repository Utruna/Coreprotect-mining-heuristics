from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def expected_directories(config: ProjectConfig) -> list[Path]:
    return [
        config.raw_dir,
        config.interim_dir,
        config.processed_dir,
        config.reports_dir,
        config.figures_dir,
        config.notebooks_dir,
    ]
