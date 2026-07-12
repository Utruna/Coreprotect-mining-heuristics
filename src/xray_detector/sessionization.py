from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SessionBoundary:
    session_id: int
    started_at: datetime
    ended_at: datetime
    event_count: int


def split_sessions(event_timestamps: list[datetime], gap_minutes: int = 15) -> list[SessionBoundary]:
    if not event_timestamps:
        return []

    ordered = sorted(event_timestamps)
    gap = timedelta(minutes=gap_minutes)
    sessions: list[SessionBoundary] = []

    current_start = ordered[0]
    current_end = ordered[0]
    current_count = 1
    session_id = 1

    for timestamp in ordered[1:]:
        if timestamp - current_end > gap:
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
            current_count = 1
            continue

        current_end = timestamp
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
