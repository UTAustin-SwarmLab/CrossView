from __future__ import annotations

import random

from stsg import STSG
from generation.qa import QAEntry, STSGMetadata
from generation import common
from generation.distractors import sample_camera_distractors
from generation.render import render_multiple_choice

CATEGORY = "best_camera"

_PROMPT = None


def _load_prompt() -> str:
    global _PROMPT
    if _PROMPT is None:
        import os

        path = os.path.join(os.path.dirname(__file__), "prompts", "best_camera.txt")
        with open(path) as f:
            _PROMPT = f.read()
    return _PROMPT


def generate(
    graph: STSG,
    rng: random.Random,
    count: int = 3,
    use_gpt: bool = True,
) -> list[QAEntry]:
    candidates = [e for e in graph.events if e.best_camera]
    if not candidates:
        return []
    rng.shuffle(candidates)

    out: list[QAEntry] = []
    seen: set[str] = set()

    for ev in candidates:
        if len(out) >= count:
            break
        if ev.id in seen:
            continue
        seen.add(ev.id)

        correct_cam = ev.best_camera
        distractor_cams, prov = sample_camera_distractors(graph, [correct_cam], rng, n=3)
        if len(distractor_cams) < 2:
            continue

        correct_text = common.camera_display(graph, correct_cam)
        distractor_texts = [common.camera_display(graph, c) for c in distractor_cams]
        for p, c in zip(prov, distractor_cams):
            p.option = common.camera_display(graph, c)

        ev_desc = common.describe_event(graph, ev)
        facts = (
            f"Event: {ev_desc}\n"
            f"Best camera (correct answer): {correct_text}\n"
            f"Other available cameras: {', '.join(distractor_texts)}"
        )

        fallback_q = f"Which camera view most clearly captures: {ev_desc}?"
        rendered = render_multiple_choice(
            facts, correct_text, distractor_texts, _load_prompt(), rng,
            use_gpt=use_gpt, fallback_question=fallback_q,
        )

        meta = STSGMetadata(
            grounding_events=[ev.id],
            target_event=ev.id,
            relation="best_camera",
            entities=[ev.entity],
            cameras=sorted(set([correct_cam] + distractor_cams)),
            num_cameras=len(set([correct_cam] + distractor_cams)),
            frame_span=[ev.t_start, ev.t_end],
            distractor_provenance=prov,
            extra={
                "optimal_camera": correct_cam,
                "camera_proximity": ev.camera_proximity,
                "rendered_by": rendered["rendered_by"],
            },
        )
        out.append(
            QAEntry(
                question=rendered["question"],
                answer=rendered["answer"],
                options=rendered["options"],
                category=CATEGORY,
                dataset=graph.dataset,
                scene_id=graph.scene_id,
                stsg_metadata=meta,
            )
        )

    return out
