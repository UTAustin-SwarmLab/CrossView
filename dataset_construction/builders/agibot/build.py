from __future__ import annotations

import json
import os
from typing import Iterator

from stsg import STSG, Camera, Entity, Event

DATASET = "agibot"


def _camera_name(path: str) -> str:
    stem = os.path.basename(path).split(".")[0]
    return stem.replace("_", " ").title() + " Camera"


def _scene_id(task: int, episode_id) -> str:
    return f"agibot_{task}_{episode_id}"


def list_scenes(compiled_path: str) -> list[str]:
    with open(compiled_path) as f:
        data = json.load(f)
    ids = []
    for task in data:
        for ep in task.get("episodes", []):
            ids.append(_scene_id(task["task"], ep["id"]))
    return ids


def iter_stsgs(compiled_path: str) -> Iterator[STSG]:
    with open(compiled_path) as f:
        data = json.load(f)
    for task in data:
        for ep in task.get("episodes", []):
            g = _build_one(task, ep)
            if g is not None:
                yield g


def build(compiled_path: str, scene_id: str) -> STSG | None:
    with open(compiled_path) as f:
        data = json.load(f)
    for task in data:
        for ep in task.get("episodes", []):
            if _scene_id(task["task"], ep["id"]) == scene_id:
                return _build_one(task, ep)
    return None


def _build_one(task: dict, ep: dict) -> STSG | None:
    annotations = ep.get("annotations", [])
    if not annotations:
        return None
    paths = ep.get("paths", [])
    camera_ids = [os.path.basename(p).split(".")[0] for p in paths]
    cameras = [Camera(id=c, name=_camera_name(p)) for c, p in zip(camera_ids, paths)]

    robot = Entity(
        uid="robot",
        category="robot action",
        description="the robot",
        cameras=camera_ids,
    )

    events: list[Event] = []
    for i, ann in enumerate(annotations):
        events.append(
            Event(
                id=f"agibot_evt_{i}",
                entity="robot",
                activity=ann.get("action_text", "action"),
                t_start=int(ann.get("start_frame", 0)),
                t_end=int(ann.get("end_frame", 0)),
                cameras=camera_ids,
                attributes={"skill": ann.get("skill")},
            )
        )

    return STSG(
        scene_id=_scene_id(task["task"], ep["id"]),
        dataset=DATASET,
        cameras=cameras,
        entities=[robot],
        events=events,
        spatial_relations=[],
        metadata={
            "task": task["task"],
            "task_name": task.get("task_name"),
            "scene_description": task.get("scene_descriptions"),
            "episode_id": ep["id"],
            "video_paths": paths,
        },
    )
