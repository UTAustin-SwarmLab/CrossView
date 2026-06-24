from __future__ import annotations

import random
from typing import Optional

from stsg import STSG, Event
from generation.qa import QAEntry, STSGMetadata
from generation import common
from generation.distractors import sample_event_distractors
from generation.render import render_multiple_choice

CATEGORY = "temporal"

_PROMPT = None


def _load_prompt() -> str:
    global _PROMPT
    if _PROMPT is None:
        import os

        path = os.path.join(os.path.dirname(__file__), "prompts", "temporal.txt")
        with open(path) as f:
            _PROMPT = f.read()
    return _PROMPT


def _relation(eg: Event, et: Event) -> Optional[str]:
    if et.t_end < eg.t_start:
        return "Before"
    if et.t_start > eg.t_end:
        return "After"
    overlap = min(eg.t_end, et.t_end) - max(eg.t_start, et.t_start)
    shorter = min(eg.t_end - eg.t_start, et.t_end - et.t_start) or 1
    if overlap > 0.5 * shorter:
        return "During"
    return None


def generate(
    graph: STSG,
    rng: random.Random,
    count: int = 3,
    use_gpt: bool = True,
    prefer_cross_camera: bool = True,
) -> list[QAEntry]:
    events = sorted(graph.events, key=lambda e: e.t_start)
    if len(events) < 2:
        return []

    out: list[QAEntry] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = 40 * max(count, 1)

    while len(out) < count and attempts < max_attempts:
        attempts += 1
        mode = rng.choice(["Before", "After", "During", "In-between"])

        if mode == "In-between":
            if len(events) < 3:
                continue
            trio = sorted(rng.sample(events, 3), key=lambda e: e.t_start)
            eg1, et, eg2 = trio
            if not (eg1.t_end < et.t_start and et.t_end < eg2.t_start):
                continue
            grounding = [eg1, eg2]
            relation = "In-between"
        else:
            eg, et = rng.sample(events, 2)
            relation = _relation(eg, et)
            if relation is None or relation != mode:
                continue
            grounding = [eg]

        if prefer_cross_camera:
            cams = common.union_cameras(grounding + [et])
            if len(cams) < 2 and attempts < max_attempts // 2:
                continue

        key = (",".join(sorted(g.id for g in grounding)), et.id)
        if key in seen:
            continue
        seen.add(key)

        distractors, prov = sample_event_distractors(
            graph, et, rng, n=3, exclude={g.id for g in grounding}
        )
        if len(distractors) < 2:
            continue

        correct_text = common.describe_event(graph, et)
        distractor_texts = [common.describe_event(graph, d) for d in distractors]

        for p, d in zip(prov, distractors):
            p.option = common.describe_event(graph, d)

        ground_desc = "; ".join(common.describe_event(graph, g) for g in grounding)
        facts = (
            f"Relation: {relation}\n"
            f"Grounding event(s): {ground_desc}\n"
            f"Target event (correct answer): {correct_text}"
        )
        if relation == "In-between":
            fallback_q = f"What happens in between {ground_desc}?"
        else:
            fallback_q = f"What happens {relation.lower()} {ground_desc}?"

        rendered = render_multiple_choice(
            facts, correct_text, distractor_texts, _load_prompt(), rng,
            use_gpt=use_gpt, fallback_question=fallback_q,
        )

        all_events = grounding + [et]
        meta = STSGMetadata(
            grounding_events=[g.id for g in grounding],
            target_event=et.id,
            relation=relation,
            entities=sorted({e.entity for e in all_events}),
            cameras=common.union_cameras(all_events),
            num_cameras=len(common.union_cameras(all_events)),
            frame_span=common.frame_span(all_events),
            distractor_provenance=prov,
            extra={"rendered_by": rendered["rendered_by"]},
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
