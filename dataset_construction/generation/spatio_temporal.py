from __future__ import annotations

import random

from stsg import STSG
from generation.qa import QAEntry, STSGMetadata
from generation import common
from generation.distractors import sample_event_distractors, sample_spatial_distractors
from generation.render import render_multiple_choice

CATEGORY = "temporal"

_DIR_PHRASE = {
    "in_front": "in front of",
    "behind": "behind",
    "left": "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
}

_PROMPT = None


def _load_prompt() -> str:
    global _PROMPT
    if _PROMPT is None:
        import os

        path = os.path.join(os.path.dirname(__file__), "prompts", "spatio_temporal.txt")
        with open(path) as f:
            _PROMPT = f.read()
    return _PROMPT


def _state_text(graph: STSG, rel) -> str:
    subj = common.describe_entity(graph, rel.subject)
    ref = common.describe_entity(graph, rel.reference)
    phrase = _DIR_PHRASE.get(rel.predicate, rel.predicate.replace("_", " "))
    return f"{subj} is {phrase} {ref}"


def _spatial_answer(predicate: str, ref: str) -> str:
    phrase = _DIR_PHRASE.get(predicate, predicate.replace("_", " "))
    return f"{phrase} {ref}"


def generate(
    graph: STSG,
    rng: random.Random,
    count: int = 3,
    use_gpt: bool = True,
) -> list[QAEntry]:
    rels = list(graph.spatial_relations)
    events = sorted(graph.events, key=lambda e: e.t_start)
    if not rels or not events:
        return []
    rng.shuffle(rels)

    out: list[QAEntry] = []
    seen: set[str] = set()

    for rel in rels:
        if len(out) >= count:
            break
        if rel.id in seen:
            continue
        seen.add(rel.id)


        later = [e for e in events if e.t_start >= rel.frame and e.entity != rel.subject]

        if later and rng.random() < 0.5:
            entry = _spatial_to_temporal(graph, rel, later, rng, use_gpt)
        else:
            entry = _temporal_to_spatial(graph, rel, events, rng, use_gpt)
        if entry is not None:
            out.append(entry)

    return out


def _spatial_to_temporal(graph, rel, later, rng, use_gpt):
    target = min(later, key=lambda e: e.t_start)
    distractors, prov = sample_event_distractors(graph, target, rng, n=3)
    if len(distractors) < 2:
        return None

    correct = common.describe_event(graph, target)
    dtexts = [common.describe_event(graph, d) for d in distractors]
    for p, d in zip(prov, distractors):
        p.option = common.describe_event(graph, d)

    state = _state_text(graph, rel)
    facts = f"Spatial state: {state}\nTarget event (correct answer): {correct}"
    fallback = f"What happens after {state}?"
    rendered = render_multiple_choice(
        facts, correct, dtexts, _load_prompt(), rng, use_gpt=use_gpt, fallback_question=fallback
    )

    meta = STSGMetadata(
        grounding_events=[],
        target_event=target.id,
        relation=f"spatial:{rel.predicate}->after",
        entities=sorted({rel.subject, rel.reference, target.entity}),
        cameras=common.union_cameras([target]),
        num_cameras=len(common.union_cameras([target])),
        frame_span=[rel.frame, target.t_end],
        distractor_provenance=prov,
        extra={"direction": "spatial_to_temporal", "spatial_relation_id": rel.id,
               "rendered_by": rendered["rendered_by"]},
    )
    return QAEntry(
        question=rendered["question"], answer=rendered["answer"], options=rendered["options"],
        category=CATEGORY, dataset=graph.dataset, scene_id=graph.scene_id, stsg_metadata=meta,
    )


def _temporal_to_spatial(graph, rel, events, rng, use_gpt):
    eg = rng.choice(events)
    ref = common.describe_entity(graph, rel.reference)
    subj = common.describe_entity(graph, rel.subject)

    correct = _spatial_answer(rel.predicate, ref)
    dpreds, prov = sample_spatial_distractors(rel.predicate)
    dtexts = [_spatial_answer(p, ref) for p in dpreds]
    for p, dt in zip(prov, dtexts):
        p.option = dt
    if len(dtexts) < 2:
        return None

    eg_desc = common.describe_event(graph, eg)
    facts = (
        f"Grounding event: {eg_desc}\n"
        f"At that time, {subj} is (correct answer): {correct}"
    )
    fallback = f"At the time when {eg_desc}, where is {subj} relative to {ref}?"
    rendered = render_multiple_choice(
        facts, correct, dtexts, _load_prompt(), rng, use_gpt=use_gpt, fallback_question=fallback
    )

    meta = STSGMetadata(
        grounding_events=[eg.id],
        target_event=None,
        relation=f"event->spatial:{rel.predicate}",
        entities=sorted({rel.subject, rel.reference, eg.entity}),
        cameras=common.union_cameras([eg]),
        num_cameras=len(common.union_cameras([eg])),
        frame_span=[eg.t_start, eg.t_end],
        distractor_provenance=prov,
        extra={"direction": "temporal_to_spatial", "spatial_relation_id": rel.id,
               "rendered_by": rendered["rendered_by"]},
    )
    return QAEntry(
        question=rendered["question"], answer=rendered["answer"], options=rendered["options"],
        category=CATEGORY, dataset=graph.dataset, scene_id=graph.scene_id, stsg_metadata=meta,
    )
