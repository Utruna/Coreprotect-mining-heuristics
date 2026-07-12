from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectConfig:
    workspace_root: Path
    archive_path: Path | None
    source_path: Path | None
    db_path: Path | None
    export_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    notebooks_dir: Path
    session_gap_minutes: int
    min_session_events: int


def _resolve_path(root: Path, raw_value: str | None, default: Path | None = None) -> Path | None:
    if raw_value is None or raw_value.strip() == "":
        return default

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def load_config(workspace_root: Path | None = None) -> ProjectConfig:
    root = (workspace_root or Path.cwd()).resolve()
    export_dir = root / "data" / "raw"
    raw_dir = root / "data" / "raw"
    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    notebooks_dir = root / "notebooks"

    return ProjectConfig(
        workspace_root=root,
        archive_path=_resolve_path(root, os.getenv("COREPROTECT_ARCHIVE_PATH")),
        source_path=_resolve_path(root, os.getenv("COREPROTECT_SOURCE_PATH")),
        db_path=_resolve_path(root, os.getenv("COREPROTECT_DB_PATH")),
        export_dir=export_dir,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        notebooks_dir=notebooks_dir,
        session_gap_minutes=int(os.getenv("COREPROTECT_SESSION_GAP_MINUTES", "15")),
        min_session_events=int(os.getenv("COREPROTECT_MIN_SESSION_EVENTS", "5")),
    )
