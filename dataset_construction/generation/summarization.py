from __future__ import annotations

import random

from stsg import STSG
from generation.qa import QAEntry, STSGMetadata
from generation import common
from generation.render import render_summary

CATEGORY = "summarization"

_QUESTION = (
    "Provide a comprehensive summary of everything that happens in this scene "
    "across all camera views, in a few sentences."
)

_PROMPT = None


def _load_prompt() -> str:
    global _PROMPT
    if _PROMPT is None:
        import os

        path = os.path.join(os.path.dirname(__file__), "prompts", "summarization.txt")
        with open(path) as f:
            _PROMPT = f.read()
    return _PROMPT


def _timeline(graph: STSG) -> str:
    lines = []
    for ev in sorted(graph.events, key=lambda e: e.t_start):
        cams = ", ".join(common.camera_display(graph, c) for c in ev.cameras)
        desc = ev.caption or common.describe_event(graph, ev)
        loc = f" [{cams}]" if cams else ""
        lines.append(f"- t={ev.t_start}-{ev.t_end}{loc}: {desc}")
    return "\n".join(lines)


def generate(
    graph: STSG,
    rng: random.Random,
    count: int = 1,
    use_gpt: bool = True,
) -> list[QAEntry]:
    if not graph.events:
        return []

    timeline = _timeline(graph)
    if use_gpt:
        rendered = render_summary(timeline, _load_prompt())
        reference = rendered["answer"]
        rendered_by = "gpt"
    else:
        reference = timeline
        rendered_by = "template"

    all_events = list(graph.events)
    meta = STSGMetadata(
        grounding_events=[e.id for e in all_events],
        target_event=None,
        relation="summary",
        entities=sorted({e.entity for e in all_events}),
        cameras=common.union_cameras(all_events),
        num_cameras=len(common.union_cameras(all_events)),
        frame_span=common.frame_span(all_events),
        distractor_provenance=[],
        extra={"timeline_event_ids": [e.id for e in all_events], "rendered_by": rendered_by},
    )
    return [
        QAEntry(
            question=_QUESTION,
            answer=reference,
            options=None,
            category=CATEGORY,
            dataset=graph.dataset,
            scene_id=graph.scene_id,
            stsg_metadata=meta,
        )
    ]
