from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from stsg import STSG, Camera, Entity, Event, SpatialRelation
from builders.meva import native

DATASET = "meva"

MEVA_VIDEO_ROOTS = ("mp4_resized", "mp4s")


DEFAULT_SLOT_INDEX = os.environ.get(
    "CROSSVIEW_MEVA_SLOT_INDEX",
    "/nas/mars/dataset/multicam_prereqs/meva/slot_index.json",
)
native.SLOT_INDEX_PATH = Path(DEFAULT_SLOT_INDEX)


def _resolve_video(clip_base: str) -> str | None:
    parts = clip_base.split(".")
    if len(parts) < 5:
        return None
    date, start, site = parts[0], parts[1], parts[-2]
    hh, mm = start.split("-")[0], start.split("-")[1]
    folder = f"{date}.{hh}-{mm}.{site}"
    for root in MEVA_VIDEO_ROOTS:
        d = f"/nas/mars/dataset/MEVA/{root}/{date}/{hh}/{folder}"
        matches = glob.glob(os.path.join(d, clip_base + ".*.mp4"))
        if matches:
            return sorted(matches)[0]
    return None


def slot_videos(slot: str) -> tuple[list[str], dict]:
    paths, names = [], {}
    for c in native.find_clips_for_slot(slot):
        base = Path(c["activities_file"]).name.replace(".activities.yml", "")
        p = _resolve_video(base)
        if p:
            paths.append(p)
            names[Path(p).stem] = f"Camera {c['camera_id']}"
    return sorted(paths), names


def list_scenes(slot_index_path: str = DEFAULT_SLOT_INDEX) -> list[str]:
    with open(slot_index_path) as f:
        index = json.load(f)
    return list(index.keys()) if isinstance(index, dict) else list(index)


def build(slot: str) -> STSG | None:
    events = native.parse_slot_events(slot)
    if not events:
        return None
    sg = native.build_scene_graph(slot, events)
    resolved = native.resolve_entities(sg)
    return _to_stsg(slot, sg, resolved)


def iter_stsgs(slot_index_path: str = DEFAULT_SLOT_INDEX) -> Iterator[STSG]:
    for slot in list_scenes(slot_index_path):
        try:
            g = build(slot)
        except Exception:
            g = None
        if g is not None:
            yield g


def _resolved_uid_map(sg, resolved) -> dict[str, str]:
    uid = {eid: eid for eid in sg.entities}
    for cluster in getattr(resolved, "entity_clusters", []):
        for eid in cluster.entities:
            uid[eid] = cluster.cluster_id
    return uid


def _to_stsg(slot, sg, resolved) -> STSG:
    uid_map = _resolved_uid_map(sg, resolved)

    cameras = [
        Camera(
            id=cam_id,
            name=cam_id,
            is_indoor=getattr(node, "is_indoor", None),
            center=list(node.position_enu) if getattr(node, "position_enu", None) else None,
            attributes={"has_krtd": getattr(node, "has_krtd", None)},
        )
        for cam_id, node in sg.cameras.items()
    ]


    cams_for_uid: dict[str, set] = defaultdict(set)
    type_for_uid: dict[str, str] = {}
    for eid, ent in sg.entities.items():
        u = uid_map[eid]
        cams_for_uid[u].add(ent.camera_id)
        type_for_uid[u] = ent.entity_type
    entities = [
        Entity(
            uid=u,
            category=type_for_uid.get(u, "unknown"),
            description=type_for_uid.get(u, None),
            cameras=sorted(cams_for_uid[u]),
        )
        for u in cams_for_uid
    ]


    events: list[Event] = []
    uid_event_cams: dict[str, Counter] = defaultdict(Counter)
    for ev in sg.events:
        actor_ids = [a["actor_id"] for a in ev.actors] if ev.actors else []
        if actor_ids:
            primary_eid = f"{ev.camera_id}_actor_{actor_ids[0]}"
            ent_uid = uid_map.get(primary_eid, primary_eid)
        else:
            ent_uid = ev.camera_id
        events.append(
            Event(
                id=ev.event_id,
                entity=ent_uid,
                activity=ev.activity,
                t_start=int(ev.start_frame),
                t_end=int(ev.end_frame),
                cameras=[ev.camera_id],
                participants=[uid_map.get(f"{ev.camera_id}_actor_{a}", "") for a in actor_ids[1:]],
            )
        )
        uid_event_cams[ent_uid][ev.camera_id] += 1


    for ev in events:
        cams = uid_event_cams.get(ev.entity)
        if cams and len(cams) > 1:
            ev.best_camera = cams.most_common(1)[0][0]
            ev.camera_proximity = dict(cams)


    acts_by_uid: dict[str, Counter] = defaultdict(Counter)
    for ev in events:
        acts_by_uid[ev.entity][ev.activity] += 1
    for e in entities:
        if e.uid in acts_by_uid:
            e.description = acts_by_uid[e.uid].most_common(1)[0][0].replace("_", " ")

    spatial = _spatial_relations(sg, uid_map)
    video_paths, camera_names = slot_videos(slot)

    return STSG(
        scene_id=slot,
        dataset=DATASET,
        cameras=cameras,
        entities=entities,
        events=events,
        spatial_relations=spatial,
        metadata={
            "annotation_source": "kitware",
            "entity_resolution": "mevid+heuristic",
            "cross_camera_clusters": len(getattr(resolved, "entity_clusters", [])),
            "video_paths": video_paths,
            "camera_names": camera_names,
        },
    )


def _spatial_relations(sg, uid_map) -> list[SpatialRelation]:
    try:
        candidates = native.find_spatial_candidates(sg)
    except Exception:
        return []

    rels: list[SpatialRelation] = []
    for i, c in enumerate(candidates):
        ea, eb = c["entity_a"], c["entity_b"]
        rels.append(
            SpatialRelation(
                id=f"meva_rel_{i}",
                frame=int(c.get("frame", 0)) if isinstance(c.get("frame"), int) else 0,
                subject=uid_map.get(ea, ea),
                reference=uid_map.get(eb, eb),
                predicate=c.get("proximity", ""),
                distance=c.get("distance_m"),
                camera=c.get("camera_a"),
            )
        )
    return rels
