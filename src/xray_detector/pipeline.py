from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig, load_config
from .paths import expected_directories


def prepare_workspace(config: ProjectConfig | None = None) -> list[Path]:
    current_config = config or load_config()
    created_directories: list[Path] = []

    for directory in expected_directories(current_config):
        directory.mkdir(parents=True, exist_ok=True)
        created_directories.append(directory)

    return created_directories
