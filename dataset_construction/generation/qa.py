from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class DistractorProvenance:

    option: str


    type: str
    source_event: Optional[str] = None
    source_entity: Optional[str] = None
    source_camera: Optional[str] = None
    note: Optional[str] = None


@dataclass
class STSGMetadata:


    grounding_events: list[str] = field(default_factory=list)
    target_event: Optional[str] = None
    relation: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    num_cameras: int = 0
    frame_span: Optional[list[int]] = None
    distractor_provenance: list[DistractorProvenance] = field(default_factory=list)

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QAEntry:

    question: str
    answer: str
    category: str
    dataset: str
    scene_id: str

    options: Optional[list[str]] = None
    stsg_metadata: Optional[STSGMetadata] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
