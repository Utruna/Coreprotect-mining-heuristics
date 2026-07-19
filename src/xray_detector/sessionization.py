from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import dist

Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class SessionBoundary:
    session_id: int
    started_at: datetime
    ended_at: datetime
    event_count: int


def split_sessions(
    event_timestamps: Sequence[datetime],
    gap_minutes: float = 15,
    event_positions: Sequence[Point3D] | None = None,
    spatial_gap_blocks: float | None = 30.0,
) -> list[SessionBoundary]:
    if not event_timestamps:
        return []

    if event_positions is not None and len(event_positions) != len(event_timestamps):
        raise ValueError("event_timestamps et event_positions doivent avoir la meme longueur")

    ordered = sorted(
        zip(
            event_timestamps,
            event_positions if event_positions is not None else [None] * len(event_timestamps),
            strict=False,
        ),
        key=lambda item: item[0],
    )
    gap = timedelta(minutes=gap_minutes)
    sessions: list[SessionBoundary] = []

    current_start = ordered[0][0]
    current_end = ordered[0][0]
    current_position = ordered[0][1]
    current_count = 1
    session_id = 1

    for timestamp, position in ordered[1:]:
        elapsed = timestamp - current_end
        split = elapsed > gap
        if (
            not split
            and spatial_gap_blocks is not None
            and current_position is not None
            and position is not None
        ):
            spatial_distance = dist(position, current_position)
            split_score = (elapsed.total_seconds() / gap.total_seconds()) + (
                spatial_distance / spatial_gap_blocks
            )
            split = split_score >= 1.5

        if split:
            sessions.append(
                SessionBoundary(
                    session_id=session_id,
                    started_at=current_start,
                    ended_at=current_end,
                    event_count=current_count,
                )
            )
            session_id += 1
            current_start = timestamp
            current_end = timestamp
            current_position = position
            current_count = 1
            continue

        current_end = timestamp
        current_position = position
        current_count += 1

    sessions.append(
        SessionBoundary(
            session_id=session_id,
            started_at=current_start,
            ended_at=current_end,
            event_count=current_count,
        )
    )
    return sessions
