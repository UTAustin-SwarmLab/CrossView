from __future__ import annotations

import json
import os
from typing import Iterator

from stsg import STSG, Camera, Entity, Event

DATASET = "ego-exo4d"


def _camera_name(cam_id: str) -> str:
    if cam_id.startswith("cam"):
        return f"Exocentric Camera {cam_id[3:]}"
    if cam_id.startswith("gp"):
        return f"Exocentric Camera {cam_id[2:]}"
    return "Egocentric Camera"


def _is_ego(cam_id: str) -> bool:
    return not (cam_id.startswith("cam") or cam_id.startswith("gp"))


def _ms(timestamp: float) -> int:
    return int(round(float(timestamp) * 1000))


def list_scenes(compiled_path: str) -> list[str]:
    with open(compiled_path) as f:
        data = json.load(f)
    return list(data.keys())


def iter_stsgs(compiled_path: str) -> Iterator[STSG]:
    with open(compiled_path) as f:
        data = json.load(f)
    for take_name, take in data.items():
        g = _build_one(take_name, take)
        if g is not None:
            yield g


def build(compiled_path: str, scene_id: str) -> STSG | None:
    with open(compiled_path) as f:
        data = json.load(f)
    take = data.get(scene_id)
    return _build_one(scene_id, take) if take else None


def _build_one(take_name: str, take: dict) -> STSG | None:
    video_files = take.get("video_files", [])
    cam_ids = [os.path.basename(p).split(".")[0] for p in video_files]
    cameras = [
        Camera(id=c, name=_camera_name(c), is_indoor=None, attributes={"ego": _is_ego(c)})
        for c in cam_ids
    ]
    exo_cams = [c for c in cam_ids if not _is_ego(c)]


    flat = [a for sub in take.get("annotations", []) for a in sub]
    if not flat:
        return None
    flat.sort(key=lambda a: a.get("timestamp", 0.0))

    ego = Entity(uid="ego", category="ego-actor", description="the camera wearer", cameras=cam_ids)


    objects = []
    for obj in take.get("objects", []) or []:
        raw = obj[0] if isinstance(obj, (list, tuple)) else str(obj)
        name = raw.rsplit("_", 1)[0] if raw.rsplit("_", 1)[-1].isdigit() else raw
        objects.append(
            Entity(uid=f"obj_{raw}", category="object", description=name.replace("_", " "), cameras=cam_ids)
        )

    events: list[Event] = []
    for i, ann in enumerate(flat):
        bc = ann.get("best_camera")
        events.append(
            Event(
                id=f"egoexo_evt_{i}",
                entity="ego",
                activity=ann.get("text", "action"),
                t_start=_ms(ann.get("timestamp", 0.0)),
                t_end=_ms(ann.get("timestamp", 0.0)),
                cameras=cam_ids,
                best_camera=bc if bc in cam_ids else None,
                caption=ann.get("text"),
                attributes={"subject": ann.get("subject")},
            )
        )

    return STSG(
        scene_id=take_name,
        dataset=DATASET,
        cameras=cameras,
        entities=[ego] + objects,
        events=events,
        spatial_relations=[],
        metadata={
            "take_uid": take.get("take_uid"),
            "task_id": take.get("task_id"),
            "scene_best_camera": take.get("best_camera"),
            "exocentric_cameras": exo_cams,
            "video_paths": video_files,
            "keystep_annotations": take.get("keystep_annotations", []),
        },
    )
