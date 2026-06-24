from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Union

from .schema import STSG, Camera, Entity, Event, SpatialRelation


def stsg_to_dict(graph: STSG) -> dict:
    return asdict(graph)


def stsg_from_dict(data: dict) -> STSG:
    return STSG(
        scene_id=data["scene_id"],
        dataset=data["dataset"],
        fps=data.get("fps"),
        num_frames=data.get("num_frames"),
        cameras=[Camera(**c) for c in data.get("cameras", [])],
        entities=[Entity(**e) for e in data.get("entities", [])],
        events=[Event(**ev) for ev in data.get("events", [])],
        spatial_relations=[
            SpatialRelation(**r) for r in data.get("spatial_relations", [])
        ],
        metadata=data.get("metadata", {}),
    )


def save_stsg(graph: STSG, path: Union[str, "os.PathLike[str]"]) -> None:
    with open(path, "w") as f:
        json.dump(stsg_to_dict(graph), f, indent=2)


def load_stsg(path: Union[str, "os.PathLike[str]"]) -> STSG:
    with open(path) as f:
        return stsg_from_dict(json.load(f))
