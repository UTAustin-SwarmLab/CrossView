from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Iterator

from stsg import STSG, Camera, Entity, Event, SpatialRelation

DATASET = "nuscenes"

_NUSCENES_CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
]


def _resolve(cfg_input) -> tuple[str, str, str | None]:
    if isinstance(cfg_input, dict):
        root = cfg_input.get("root")
        if root:
            sg = cfg_input.get("scene_graphs", os.path.join(root, "scene_graphs"))
            ia = cfg_input.get("instance_annotations", os.path.join(root, "instance_annotations"))
        else:
            sg = cfg_input["scene_graphs"]
            ia = cfg_input["instance_annotations"]
        return sg, ia, cfg_input.get("videos")

    parent = os.path.dirname(cfg_input.rstrip("/"))
    return cfg_input, os.path.join(parent, "instance_annotations"), None


def list_scenes(cfg_input) -> list[str]:
    sg_root, _, _ = _resolve(cfg_input)
    return sorted(
        d for d in glob.glob(os.path.join(sg_root, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "scene_graph.json"))
    )


def iter_stsgs(cfg_input) -> Iterator[STSG]:
    _, ia_root, vid_root = _resolve(cfg_input)
    for scene_dir in list_scenes(cfg_input):
        g = build(scene_dir, ia_root, vid_root)
        if g is not None:
            yield g


def build(scene_dir: str, ia_root: str, video_root: str | None = None) -> STSG | None:
    sg_path = os.path.join(scene_dir, "scene_graph.json")
    if not os.path.exists(sg_path):
        return None
    with open(sg_path) as f:
        scene = json.load(f)
    token = scene.get("scene_token") or os.path.basename(scene_dir)

    activities: dict[str, dict] = {}
    ia_path = os.path.join(ia_root, f"{token}_instance_annotations.json")
    if os.path.exists(ia_path):
        with open(ia_path) as f:
            ia = json.load(f)
        for a in ia.get("annotations", []):
            activities[a["instance_token"]] = a

    return _to_stsg(scene, token, activities, video_root)


def _video_paths(scene_token: str, video_root: str | None) -> tuple[list[str], dict]:
    if not video_root:
        return [], {}
    scene_vid_dir = os.path.join(video_root, scene_token)
    paths, names = [], {}
    for cam in _NUSCENES_CAMERAS:
        p = os.path.join(scene_vid_dir, f"{cam}.mp4")
        if os.path.exists(p):
            paths.append(p)
            names[cam] = cam.replace("CAM_", "").replace("_", " ").title()
    return paths, names


def _to_stsg(scene: dict, token: str, activities: dict, video_root: str | None) -> STSG:
    frames = scene.get("frames", [])

    obj_class: dict[str, str] = {}
    obj_cams: dict[str, set] = defaultdict(set)
    obj_frames: dict[str, list[int]] = defaultdict(list)
    for fr in frames:
        idx = int(fr.get("frame_idx", 0))
        for o in fr.get("objects", []):
            oid = o.get("object_id") or o.get("annotation_token")
            if not oid:
                continue
            obj_class[oid] = o.get("object_class", "object")
            for c in o.get("visible_cameras", []):
                obj_cams[oid].add(c)
            obj_frames[oid].append(idx)

    entities = []
    for oid, cls in obj_class.items():
        ann = activities.get(oid, {})
        desc = ann.get("description") or (cls.split(".")[-1].replace("_", " ") if "." in cls else cls)
        entities.append(Entity(uid=oid, category=cls, description=desc, cameras=sorted(obj_cams[oid])))


    events: list[Event] = []
    for oid, ann in activities.items():
        activity = ann.get("activity")
        if not activity or oid not in obj_frames:
            continue
        fr_list = obj_frames[oid]
        cams = sorted(obj_cams[oid]) or [f["camera"] for f in ann.get("frames_info", [])[:1]]
        events.append(Event(
            id=f"nusc_evt_{oid}",
            entity=oid,
            activity=activity,
            t_start=min(fr_list),
            t_end=max(fr_list),
            cameras=cams,
            caption=activity,
        ))


    spatial: list[SpatialRelation] = []
    if frames:
        fr = frames[len(frames) // 2]
        idx = int(fr.get("frame_idx", 0))
        for r in fr.get("relationships", []):
            spatial.append(SpatialRelation(
                id=f"nusc_rel_{len(spatial)}",
                frame=idx,
                subject=r.get("source_id", ""),
                reference=r.get("target_id", ""),
                predicate=r.get("relationship_type", ""),
                distance=r.get("distance"),
            ))

    video_paths, cam_names = _video_paths(token, video_root)
    cams_present = sorted({c for cs in obj_cams.values() for c in cs}) or _NUSCENES_CAMERAS
    cameras = [
        Camera(id=c, name=cam_names.get(c, c.replace("CAM_", "").replace("_", " ").title()))
        for c in cams_present
    ]

    return STSG(
        scene_id=token,
        dataset=DATASET,
        num_frames=len(frames),
        cameras=cameras,
        entities=entities,
        events=events,
        spatial_relations=spatial,
        metadata={"video_paths": video_paths, "scene_name": scene.get("scene_name")},
    )
