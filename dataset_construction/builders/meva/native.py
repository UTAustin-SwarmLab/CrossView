from __future__ import annotations

import ast
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


MEVA_ROOT = Path("/nas/mars/dataset/MEVA")
ANNOTATION_BASE = MEVA_ROOT / "meva-data-repo" / "annotation" / "DIVA-phase-2" / "MEVA"
KITWARE_ROOT = ANNOTATION_BASE / "kitware"
KITWARE_TRAINING_ROOT = ANNOTATION_BASE / "kitware-meva-training"
KRTD_DIR = MEVA_ROOT / "meva-data-repo" / "metadata" / "camera-models" / "krtd"
MEVID_DATA_DIR = MEVA_ROOT / "mevid_data" / "mevid-v1-annotation-data"
SLOT_INDEX_PATH = Path(os.environ.get(
    "CROSSVIEW_MEVA_SLOT_INDEX",
    "/nas/mars/dataset/multicam_prereqs/meva/slot_index.json",
))

DEFAULT_FRAMERATE = 30.0
INDOOR_CAMERAS = {"G299", "G330"}


@dataclass
class Event:
    event_id: str
    activity: str
    camera_id: str
    site: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    actors: List[Dict[str, Any]]
    video_file: str
    annotation_source: str


@dataclass
class Entity:
    entity_id: str
    camera_id: str
    actor_id: int
    entity_type: str
    first_frame: int
    last_frame: int
    first_sec: float
    last_sec: float
    keyframe_bboxes: Dict[int, List[int]]
    events: List[str]


@dataclass
class CameraNode:
    camera_id: str
    is_indoor: bool
    has_krtd: bool
    position_enu: Optional[Tuple[float, float, float]]


@dataclass
class SceneGraph:
    slot: str
    cameras: Dict[str, CameraNode]
    entities: Dict[str, Entity]
    events: List[Event]
    events_by_camera: Dict[str, List[Event]]


@dataclass
class EntityCluster:
    cluster_id: str
    entities: List[str]
    cameras: List[str]
    mevid_person_id: Optional[int] = None
    link_type: str = "heuristic"


@dataclass
class ResolvedGraph:
    entity_clusters: List[EntityCluster]
    mevid_persons_in_slot: int = 0


def _load_yaml_fast(path: Path) -> list:
    import yaml
    try:
        Loader = yaml.CSafeLoader
    except AttributeError:
        Loader = yaml.SafeLoader
    with open(path) as f:
        return yaml.load(f, Loader=Loader) or []


def _parse_types_yml(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    type_map = {}
    for entry in _load_yaml_fast(path):
        t = entry.get("types", {})
        if t:
            aid = t.get("id1")
            cset = t.get("cset3", {})
            etype = next(iter(cset.keys()), "unknown") if cset else "unknown"
            if aid is not None:
                type_map[aid] = etype
    return type_map


def _parse_activities_yml(path: Path, camera_id: str, site: str,
                          framerate: float, source: str) -> List[Event]:
    if not path.exists():
        return []
    entries = _load_yaml_fast(path)
    events = []
    type_map = _parse_types_yml(
        path.with_name(path.name.replace(".activities.yml", ".types.yml")))

    for entry in entries:
        act = entry.get("act", {})
        if not act:
            continue
        act2 = act.get("act2", {})
        activity_name = next(iter(act2.keys()), "unknown")
        activity_id = act.get("id2", -1)
        timespan = act.get("timespan", [])
        if not timespan:
            continue
        tsr = timespan[0].get("tsr0", [])
        if len(tsr) < 2:
            continue
        start_frame, end_frame = int(tsr[0]), int(tsr[1])
        start_sec = round(start_frame / framerate, 2)
        end_sec = round(end_frame / framerate, 2)

        actors = []
        for actor_entry in act.get("actors", []):
            aid = actor_entry.get("id1")
            if aid is not None:
                actors.append({"actor_id": aid, "entity_type": type_map.get(aid, "unknown")})

        clip_name = path.stem.replace(".activities", "")
        events.append(Event(
            event_id=f"{camera_id}_evt_{activity_id}",
            activity=activity_name, camera_id=camera_id, site=site,
            start_frame=start_frame, end_frame=end_frame,
            start_sec=start_sec, end_sec=end_sec,
            duration_sec=round(end_sec - start_sec, 2),
            actors=actors, video_file=f"{clip_name}.avi", annotation_source=source,
        ))
    return events


def find_clips_for_slot(slot: str) -> List[Dict]:
    if not SLOT_INDEX_PATH.exists():
        raise FileNotFoundError(f"slot_index.json not found at {SLOT_INDEX_PATH}")
    with open(SLOT_INDEX_PATH) as f:
        index = json.load(f)
    if slot not in index:
        raise ValueError(f"Slot '{slot}' not found in index ({len(index)} slots)")

    info = index[slot]
    clips = []
    parts = slot.split(".")
    date, slot_time = parts[0], parts[1]
    site = parts[2] if len(parts) > 2 else "school"
    hour = slot_time[:2]

    source_dirs = {"kitware": KITWARE_ROOT, "kitware-training": KITWARE_TRAINING_ROOT}
    cameras_seen: Set[str] = set()
    for source_name, source_dir in source_dirs.items():
        if source_name not in info.get("sources", {}):
            continue
        ann_dir = source_dir / date / hour
        if not ann_dir.exists():
            continue
        for cam_id in info["cameras"]:
            if cam_id in cameras_seen or cam_id not in info["sources"].get(source_name, []):
                continue
            matches = list(ann_dir.glob(f"{date}.{slot_time}*{cam_id}*.activities.yml"))
            if not matches:
                matches = list(ann_dir.glob(f"{date}.{slot_time[:5]}*{cam_id}*.activities.yml"))
            if matches:
                act_file = matches[0]
                cameras_seen.add(cam_id)
                clips.append({
                    "clip_name": act_file.stem.replace(".activities", ""),
                    "camera_id": cam_id, "site": site,
                    "annotation_dir": str(ann_dir), "annotation_source": source_name,
                    "framerate": DEFAULT_FRAMERATE, "activities_file": str(act_file),
                })
    return clips


def parse_slot_events(slot: str) -> List[Event]:
    all_events: List[Event] = []
    for clip in find_clips_for_slot(slot):
        all_events.extend(_parse_activities_yml(
            Path(clip["activities_file"]), clip["camera_id"], clip["site"],
            clip["framerate"], clip["annotation_source"]))
    all_events.sort(key=lambda e: (e.start_sec, e.camera_id))
    return all_events


def stream_geom_records(path: Path):
    inline_re = re.compile(r"id1:\s*(\d+).*?ts0:\s*(\d+).*?g0:\s*(\d+\s+\d+\s+\d+\s+\d+)")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("- {") or line.startswith("{"):
                try:
                    obj = ast.literal_eval(line[2:] if line.startswith("- ") else line)
                except Exception:
                    obj = None
                g = obj.get("geom") if isinstance(obj, dict) else None
                if g and g.get("g0") is not None and g.get("id1") is not None and g.get("ts0") is not None:
                    g0 = g["g0"]
                    coords = [int(float(x)) for x in g0.split()] if isinstance(g0, str) else [int(x) for x in g0]
                    if len(coords) >= 4:
                        yield {"id1": int(g["id1"]), "ts0": int(g["ts0"]), "g0": coords[:4]}
                    continue
            m = inline_re.search(line)
            if m:
                yield {"id1": int(m.group(1)), "ts0": int(m.group(2)),
                       "g0": [int(x) for x in m.group(3).split()]}


def get_actor_keyframe_bboxes(path: Path, actor_ids: Optional[Set[int]] = None,
                              sample_every: int = 30) -> Dict[int, Dict[int, List[int]]]:
    result: Dict[int, Dict[int, List[int]]] = {}
    for rec in stream_geom_records(path):
        aid = rec["id1"]
        if actor_ids is not None and aid not in actor_ids:
            continue
        if rec["ts0"] % sample_every != 0:
            continue
        result.setdefault(aid, {})[rec["ts0"]] = rec["g0"]
    return result


def get_actor_frame_range(path: Path) -> Dict[int, tuple]:
    ranges: Dict[int, list] = {}
    for rec in stream_geom_records(path):
        aid, frame = rec["id1"], rec["ts0"]
        if aid not in ranges:
            ranges[aid] = [frame, frame]
        else:
            ranges[aid][0] = min(ranges[aid][0], frame)
            ranges[aid][1] = max(ranges[aid][1], frame)
    return {aid: tuple(r) for aid, r in ranges.items()}


def get_bbox_at_frame(path: Path, actor_id: int, target_frame: int,
                      tolerance: int = 5) -> Optional[List[int]]:
    best, best_dist = None, tolerance + 1
    for rec in stream_geom_records(path):
        if rec["id1"] != actor_id:
            continue
        dist = abs(rec["ts0"] - target_frame)
        if dist < best_dist:
            best_dist, best = dist, rec["g0"]
        if dist == 0:
            break
    return best


class CameraModel:
    def __init__(self, krtd_path: Path):
        with open(krtd_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.K = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
        self.R = np.array([[float(x) for x in lines[i].split()] for i in range(3, 6)])
        self.T = np.array([float(x) for x in lines[6].split()])
        self.D = np.array([float(x) for x in lines[7].split()]) if len(lines) > 7 else None

    @property
    def camera_center(self) -> np.ndarray:
        return -self.R.T @ self.T

    def project_to_ground(self, u: float, v: float, ground_z: float = 0.0) -> Optional[np.ndarray]:
        d_cam = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        d_world = self.R.T @ d_cam
        C = self.camera_center
        if abs(d_world[2]) < 1e-10:
            return None
        t = (ground_z - C[2]) / d_world[2]
        if t < 0:
            return None
        return C + t * d_world

    def bbox_foot_to_world(self, bbox: List[float], ground_z: float = 0.0) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        return self.project_to_ground((x1 + x2) / 2.0, max(y1, y2), ground_z)


def load_camera_model(camera_id: str) -> Optional[CameraModel]:
    if camera_id in INDOOR_CAMERAS:
        return None
    krtd_files = list(KRTD_DIR.glob(f"*.{camera_id}.krtd"))
    if not krtd_files:
        return None
    try:
        return CameraModel(krtd_files[0])
    except Exception:
        return None


def classify_proximity(distance_m: float) -> str:
    if distance_m <= 5.0:
        return "near"
    if distance_m <= 15.0:
        return "moderate"
    return "far"


_MEVID_NAME_RE = re.compile(r"^(\d{4})O(\d{3})C(\d+)T(\d{3})F(\d{5})\.jpg$")


def parse_mevid_person_cameras() -> Dict[int, Set[str]]:
    person_cameras: Dict[int, Set[str]] = defaultdict(set)
    for fname in ("train_name.txt", "test_name.txt"):
        fpath = MEVID_DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                m = _MEVID_NAME_RE.match(line.strip())
                if m:
                    person_cameras[int(m.group(1))].add(f"G{m.group(3)}")
    return dict(person_cameras)


def find_mevid_persons_for_slot(slot: str, slot_cameras: List[str]) -> Dict[int, Set[str]]:
    slot_set = set(slot_cameras)
    out = {}
    for pid, cams in parse_mevid_person_cameras().items():
        overlap = cams & slot_set
        if len(overlap) >= 2:
            out[pid] = overlap
    return out


def build_scene_graph(slot: str, events: List[Event]) -> SceneGraph:
    camera_ids = sorted(set(e.camera_id for e in events))
    cameras: Dict[str, CameraNode] = {}
    for cam_id in camera_ids:
        model = load_camera_model(cam_id)
        cameras[cam_id] = CameraNode(
            camera_id=cam_id, is_indoor=cam_id in INDOOR_CAMERAS,
            has_krtd=model is not None,
            position_enu=tuple(model.camera_center.tolist()) if model else None,
        )

    entity_actor_ids: Dict[str, Set[int]] = defaultdict(set)
    entity_types: Dict[str, Dict[int, str]] = defaultdict(dict)
    entity_events: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for evt in events:
        for actor in evt.actors:
            aid = actor["actor_id"]
            entity_actor_ids[evt.camera_id].add(aid)
            entity_types[evt.camera_id][aid] = actor.get("entity_type", "unknown")
            entity_events[evt.camera_id][aid].append(evt.event_id)

    clip_by_camera = {c["camera_id"]: c for c in find_clips_for_slot(slot)}
    entity_bboxes: Dict[str, Dict[int, Dict[int, List[int]]]] = {}
    entity_frame_ranges: Dict[str, Dict[int, tuple]] = {}
    for cam_id, actor_ids in entity_actor_ids.items():
        if cam_id not in clip_by_camera:
            continue
        geom_path = Path(clip_by_camera[cam_id]["activities_file"]).with_name(
            Path(clip_by_camera[cam_id]["activities_file"]).name.replace(".activities.yml", ".geom.yml"))
        if geom_path.exists():
            try:
                entity_bboxes[cam_id] = get_actor_keyframe_bboxes(geom_path, actor_ids, sample_every=30)
                entity_frame_ranges[cam_id] = get_actor_frame_range(geom_path)
            except Exception:
                pass

    entities: Dict[str, Entity] = {}
    fr = DEFAULT_FRAMERATE
    for cam_id, actor_ids in entity_actor_ids.items():
        cam_ranges = entity_frame_ranges.get(cam_id, {})
        cam_bboxes = entity_bboxes.get(cam_id, {})
        for aid in actor_ids:
            if aid in cam_ranges:
                first_frame, last_frame = cam_ranges[aid]
            else:
                ae = [e for e in events if e.camera_id == cam_id
                      and any(a["actor_id"] == aid for a in e.actors)]
                if ae:
                    first_frame = min(e.start_frame for e in ae)
                    last_frame = max(e.end_frame for e in ae)
                else:
                    first_frame, last_frame = 0, 0
            eid = f"{cam_id}_actor_{aid}"
            entities[eid] = Entity(
                entity_id=eid, camera_id=cam_id, actor_id=aid,
                entity_type=entity_types.get(cam_id, {}).get(aid, "unknown"),
                first_frame=first_frame, last_frame=last_frame,
                first_sec=round(first_frame / fr, 2), last_sec=round(last_frame / fr, 2),
                keyframe_bboxes=cam_bboxes.get(aid, {}),
                events=entity_events.get(cam_id, {}).get(aid, []),
            )

    events_by_camera: Dict[str, List[Event]] = defaultdict(list)
    for evt in events:
        events_by_camera[evt.camera_id].append(evt)
    return SceneGraph(slot=slot, cameras=cameras, entities=entities,
                      events=events, events_by_camera=dict(events_by_camera))


class _UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def clusters(self) -> Dict[str, Set[str]]:
        groups: Dict[str, Set[str]] = defaultdict(set)
        for item in self.parent:
            groups[self.find(item)].add(item)
        return dict(groups)


def _heuristic_links(sg: SceneGraph):
    MAX_GAP, MIN_GAP, MAX_PER = 10.0, 1.0, 2
    links = []
    link_count: Dict[str, int] = {}
    active = [{"entity_id": eid, "camera_id": e.camera_id, "first_sec": e.first_sec, "last_sec": e.last_sec}
              for eid, e in sg.entities.items() if e.entity_type == "person" and e.events]
    active.sort(key=lambda x: x["last_sec"])
    by_first = sorted(active, key=lambda x: x["first_sec"])
    for ea in active:
        if link_count.get(ea["entity_id"], 0) >= MAX_PER:
            continue
        for eb in by_first:
            if ea["camera_id"] == eb["camera_id"]:
                continue
            gap = eb["first_sec"] - ea["last_sec"]
            if gap < MIN_GAP:
                continue
            if gap > MAX_GAP:
                break
            if link_count.get(eb["entity_id"], 0) >= MAX_PER:
                continue
            conf = max(0.4, 1.0 - gap / MAX_GAP)
            links.append((ea["entity_id"], eb["entity_id"], round(conf, 2)))
            link_count[ea["entity_id"]] = link_count.get(ea["entity_id"], 0) + 1
            link_count[eb["entity_id"]] = link_count.get(eb["entity_id"], 0) + 1
    return links


def resolve_entities(sg: SceneGraph) -> ResolvedGraph:
    mevid_persons = find_mevid_persons_for_slot(sg.slot, list(sg.cameras.keys()))
    links = _heuristic_links(sg)

    uf = _UnionFind()
    for a, b, conf in links:
        if conf >= 0.7:
            uf.union(a, b)
    for eid in sg.entities:
        uf.find(eid)

    clusters = []
    idx = 0
    for _, members in uf.clusters().items():
        if len(members) < 2:
            continue
        cams = sorted(set(sg.entities[m].camera_id for m in members if m in sg.entities))
        if len(cams) < 2:
            continue
        clusters.append(EntityCluster(cluster_id=f"cluster_{idx}", entities=sorted(members), cameras=cams))
        idx += 1
    return ResolvedGraph(entity_clusters=clusters, mevid_persons_in_slot=len(mevid_persons))


def find_spatial_candidates(sg: SceneGraph) -> List[Dict]:
    camera_models: Dict[str, CameraModel] = {}
    for cam_id in sg.cameras:
        if cam_id in INDOOR_CAMERAS:
            continue
        model = load_camera_model(cam_id)
        if model is not None:
            camera_models[cam_id] = model

    clip_by_camera = {c["camera_id"]: c for c in find_clips_for_slot(sg.slot)}
    positions: Dict[str, Dict] = {}
    for eid, entity in sg.entities.items():
        if entity.camera_id not in camera_models or entity.entity_type != "person":
            continue
        model = camera_models[entity.camera_id]
        mid_frame = (entity.first_frame + entity.last_frame) // 2
        bbox = None
        if entity.keyframe_bboxes:
            closest = min(entity.keyframe_bboxes.keys(), key=lambda f: abs(int(f) - mid_frame))
            bbox = entity.keyframe_bboxes[closest]
        if bbox is None and entity.camera_id in clip_by_camera:
            geom_path = Path(clip_by_camera[entity.camera_id]["activities_file"]).with_name(
                Path(clip_by_camera[entity.camera_id]["activities_file"]).name.replace(".activities.yml", ".geom.yml"))
            if geom_path.exists():
                bbox = get_bbox_at_frame(geom_path, entity.actor_id, mid_frame, tolerance=15)
        if bbox is None:
            continue
        pos = model.bbox_foot_to_world(bbox)
        if pos is None:
            continue
        positions[eid] = {"position": pos, "camera_id": entity.camera_id, "entity": entity}

    candidates = []
    ids = sorted(positions.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pa, pb = positions[ids[i]], positions[ids[j]]
            distance = float(np.linalg.norm(pa["position"] - pb["position"]))
            if distance > 500:
                continue
            candidates.append({
                "entity_a": ids[i], "entity_b": ids[j],
                "camera_a": pa["camera_id"], "camera_b": pb["camera_id"],
                "distance_m": round(distance, 2), "proximity": classify_proximity(distance),
                "entity_a_obj": pa["entity"], "entity_b_obj": pb["entity"],
            })
    return candidates
