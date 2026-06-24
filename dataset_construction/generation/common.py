from __future__ import annotations

from stsg import STSG, Event


def describe_entity(graph: STSG, uid: str) -> str:
    ent = graph.entity(uid)
    if ent is None:
        return uid
    return ent.description or ent.category or uid


def describe_event(graph: STSG, event: Event) -> str:
    actor = describe_entity(graph, event.entity)
    activity = event.activity.replace("_", " ")
    if actor and actor not in activity:
        return f"{actor}: {activity}"
    return activity


def camera_display(graph: STSG, camera_id: str) -> str:
    cam = graph.camera(camera_id)
    if cam is not None and cam.name:
        return cam.name
    return camera_id


def event_cameras(event: Event) -> list[str]:
    return list(event.cameras) if event.cameras else []


def frame_span(events: list[Event]) -> list[int] | None:
    if not events:
        return None
    return [min(e.t_start for e in events), max(e.t_end for e in events)]


def union_cameras(events: list[Event]) -> list[str]:
    cams: list[str] = []
    for e in events:
        for c in e.cameras:
            if c not in cams:
                cams.append(c)
    return sorted(cams)
