from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Camera:

    id: str
    name: Optional[str] = None
    is_indoor: Optional[bool] = None

    intrinsics: Optional[list[list[float]]] = None

    extrinsics: Optional[list[list[float]]] = None

    center: Optional[list[float]] = None
    attributes: dict = field(default_factory=dict)


@dataclass
class Entity:

    uid: str
    category: str


    description: Optional[str] = None

    cameras: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


@dataclass
class SpatialRelation:

    id: str
    frame: int
    subject: str
    reference: str
    predicate: str
    orientation_delta: Optional[float] = None
    distance: Optional[float] = None
    camera: Optional[str] = None


@dataclass
class Event:

    id: str
    entity: str
    activity: str
    t_start: int
    t_end: int
    cameras: list[str] = field(default_factory=list)

    best_camera: Optional[str] = None


    camera_proximity: dict = field(default_factory=dict)
    caption: Optional[str] = None

    participants: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


@dataclass
class STSG:

    scene_id: str
    dataset: str
    fps: Optional[float] = None
    num_frames: Optional[int] = None
    cameras: list[Camera] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    spatial_relations: list[SpatialRelation] = field(default_factory=list)

    metadata: dict = field(default_factory=dict)


    def entity(self, uid: str) -> Optional[Entity]:
        for e in self.entities:
            if e.uid == uid:
                return e
        return None

    def event(self, event_id: str) -> Optional[Event]:
        for ev in self.events:
            if ev.id == event_id:
                return ev
        return None

    def camera(self, camera_id: str) -> Optional[Camera]:
        for c in self.cameras:
            if c.id == camera_id:
                return c
        return None

    def events_for(self, uid: str) -> list[Event]:
        return [ev for ev in self.events if ev.entity == uid]
