import pandas as pd

from datetime import datetime

from xray_detector.config import load_config
from xray_detector.mining import segment_sessions
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


def test_split_sessions_respects_spatial_gaps():
    timestamps = [
        datetime(2026, 1, 1, 12, 0),
        datetime(2026, 1, 1, 12, 1),
        datetime(2026, 1, 1, 12, 2),
        datetime(2026, 1, 1, 12, 3),
    ]
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (40.0, 0.0, 0.0),
        (41.0, 0.0, 0.0),
    ]

    sessions = split_sessions(
        timestamps,
        gap_minutes=10,
        event_positions=positions,
        spatial_gap_blocks=30.0,
    )

    assert len(sessions) == 1
    assert sessions[0].event_count == 4


def test_split_sessions_can_split_on_combined_gap():
    timestamps = [
        datetime(2026, 1, 1, 12, 0),
        datetime(2026, 1, 1, 12, 5),
        datetime(2026, 1, 1, 12, 10),
    ]
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (61.0, 0.0, 0.0),
    ]

    sessions = split_sessions(
        timestamps,
        gap_minutes=10,
        event_positions=positions,
        spatial_gap_blocks=30.0,
    )

    assert len(sessions) == 2
    assert [session.event_count for session in sessions] == [2, 1]


def test_segment_sessions_splits_on_spatial_gap():
    df = pd.DataFrame(
        {
            "time": [0, 60, 120, 180],
            "pseudo": ["alice"] * 4,
            "wid": [1] * 4,
            "x": [0, 1, 40, 41],
            "y": [0, 0, 0, 0],
            "z": [0, 0, 0, 0],
        }
    )

    segmented, dropped = segment_sessions(df, gap_seconds=600, min_blocks=1, gap_blocks=30.0)

    assert dropped == 0
    assert segmented["session_id"].tolist() == [0, 0, 0, 0]


def test_segment_sessions_splits_on_combined_gap():
    df = pd.DataFrame(
        {
            "time": [0, 60, 420],
            "pseudo": ["alice"] * 3,
            "wid": [1] * 3,
            "x": [0, 1, 61],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
        }
    )

    segmented, dropped = segment_sessions(df, gap_seconds=600, min_blocks=1, gap_blocks=30.0)

    assert dropped == 0
    assert segmented["session_id"].tolist() == [0, 0, 1]
