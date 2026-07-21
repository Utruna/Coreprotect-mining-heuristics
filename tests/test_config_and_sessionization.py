import pandas as pd

from datetime import datetime, timezone

from xray_detector.config import load_config
from xray_detector.mining import (
    filter_cave_like_sessions,
    is_cave_like_session,
    is_cave_shaped_session,
    parse_utc_datetime,
    segment_sessions,
)
from xray_detector.sessionization import split_sessions


def test_parse_utc_datetime_accepts_common_forms():
    expected = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert parse_utc_datetime("2026-06-01") == expected
    assert parse_utc_datetime("2026-6-1") == expected
    assert parse_utc_datetime("2026-06-1T00:00:00Z") == expected
    assert parse_utc_datetime("2026-06-01T02:00:00+02:00") == expected


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


def test_cave_like_session_filter_detects_natural_cavities():
    cave_seg = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4, 5],
            "pseudo": ["alice"] * 6,
            "wid": [1] * 6,
            "session_id": [0] * 6,
            "material": [
                "minecraft:stone",
                "minecraft:calcite",
                "minecraft:smooth_basalt",
                "minecraft:stone",
                "minecraft:moss_block",
                "minecraft:diamond_ore",
            ],
        }
    )
    tunnel_seg = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4, 5],
            "pseudo": ["alice"] * 6,
            "wid": [1] * 6,
            "session_id": [1] * 6,
            "material": ["minecraft:stone", "minecraft:deepslate"] * 3,
        }
    )
    df = pd.concat([cave_seg, tunnel_seg], ignore_index=True)

    assert is_cave_like_session(cave_seg)
    assert not is_cave_like_session(tunnel_seg)

    kept, excluded = filter_cave_like_sessions(df)

    assert excluded == 1
    assert kept["session_id"].tolist() == [1, 1, 1, 1, 1, 1]


def _shape_session(n: int, ore_every: int, walk_every: int | None, seconds_per_block: float):
    """Session synthetique : un bloc par pas, minerai tous les `ore_every` blocs,
    un saut de marche (> 4 blocs) tous les `walk_every` pas si demande."""
    rows = []
    x = 0
    for i in range(n):
        x += 10 if (walk_every and i and i % walk_every == 0) else 1
        rows.append({
            "time": int(i * seconds_per_block), "x": x, "y": 12, "z": 0,
            "ore": "diamond" if i % ore_every == 0 else None,
        })
    return pd.DataFrame(rows)


def test_cave_shaped_session_detects_walked_ore_rich_cavity():
    # Grotte : 1 minerai visible sur 4, un pas de marche sur 8, rythme lent.
    cave = _shape_session(120, ore_every=4, walk_every=8, seconds_per_block=3.0)
    assert is_cave_shaped_session(cave)


def test_cave_shaped_session_keeps_fast_digging_even_with_high_yield():
    # X-ray : rendement eleve aussi, mais creusage rapide et continu (sans
    # marche) sur une grosse session -> ne doit PAS etre ecarte comme grotte.
    xray = _shape_session(600, ore_every=5, walk_every=None, seconds_per_block=0.8)
    assert not is_cave_shaped_session(xray)
