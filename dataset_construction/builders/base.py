from __future__ import annotations

from collections import defaultdict

from stsg import Event


def merge_activity_intervals(
    labels: list[tuple[int, str]],
    entity_uid: str,
    camera: str,
    *,
    buffer_frames: int = 2,
    min_frames: int = 3,
    id_prefix: str = "evt",
) -> list[Event]:
    if not labels:
        return []
    labels = sorted(labels)
    events: list[Event] = []
    cur_act = labels[0][1]
    start = prev = labels[0][0]
    idx = 0

    def flush(a: str, s: int, e: int) -> None:
        nonlocal idx
        if e - s + 1 >= min_frames:
            events.append(
                Event(
                    id=f"{camera}_{entity_uid}_{id_prefix}_{idx}",
                    entity=entity_uid,
                    activity=a,
                    t_start=s,
                    t_end=e,
                    cameras=[camera],
                )
            )
            idx += 1

    for frame, act in labels[1:]:
        if act == cur_act and frame - prev <= buffer_frames + 1:
            prev = frame
        else:
            flush(cur_act, start, prev)
            cur_act, start, prev = act, frame, frame
    flush(cur_act, start, prev)
    return events


def index_events_by_entity(events: list[Event]) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        out[e.entity].append(e)
    return out
