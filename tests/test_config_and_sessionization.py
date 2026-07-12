from datetime import datetime

from xray_detector.config import load_config
from xray_detector.sessionization import split_sessions


def test_load_config_uses_workspace_paths(tmp_path):
    config = load_config(tmp_path)

    assert config.workspace_root == tmp_path.resolve()
    assert config.raw_dir == tmp_path / "data" / "raw"
    assert config.session_gap_minutes == 15
    assert config.min_session_events == 5


def test_split_sessions_respects_gaps():
    timestamps = [
        datetime(2026, 1, 1, 12, 0),
        datetime(2026, 1, 1, 12, 5),
        datetime(2026, 1, 1, 12, 30),
    ]

    sessions = split_sessions(timestamps, gap_minutes=10)

    assert len(sessions) == 2
    assert sessions[0].event_count == 2
    assert sessions[1].event_count == 1
