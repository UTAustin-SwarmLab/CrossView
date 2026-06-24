from __future__ import annotations

import random
from typing import Optional

from stsg import STSG
from generation.qa import QAEntry, STSGMetadata
from generation import common
from generation.distractors import sample_spatial_distractors
from generation.render import render_multiple_choice

CATEGORY = "spatial"

_PROMPT = None


def _load_prompt() -> str:
    global _PROMPT
    if _PROMPT is None:
        import os

        path = os.path.join(os.path.dirname(__file__), "prompts", "spatial.txt")
        with open(path) as f:
            _PROMPT = f.read()
    return _PROMPT


def _bucket(distance: float) -> str:
    if distance <= 5.0:
        return "near"
    if distance <= 15.0:
        return "moderate"
    return "far"


def _answer_text(predicate: str, ref_desc: str) -> str:
    proximity_phrases = {
        "near": f"near {ref_desc} (within a few meters)",
        "moderate": f"a moderate distance from {ref_desc} (5-15 meters)",
        "far": f"far from {ref_desc} (more than 15 meters)",
        "same_location": f"at the same location as {ref_desc}",
    }
    if predicate in proximity_phrases:
        return proximity_phrases[predicate]
    return f"{predicate.replace('_', ' ')} of {ref_desc}"


def generate(
    graph: STSG,
    rng: random.Random,
    count: int = 3,
    use_gpt: bool = True,
) -> list[QAEntry]:
    rels = list(graph.spatial_relations)
    if not rels:
        return []
    rng.shuffle(rels)

    out: list[QAEntry] = []
    seen: set[tuple[str, str]] = set()

    for rel in rels:
        if len(out) >= count:
            break

        predicate: Optional[str] = rel.predicate
        if (predicate is None or predicate == "") and rel.distance is not None:
            predicate = _bucket(rel.distance)
        if not predicate:
            continue

        key = (rel.subject, rel.reference)
        if key in seen:
            continue
        seen.add(key)

        subj_desc = common.describe_entity(graph, rel.subject)
        ref_desc = common.describe_entity(graph, rel.reference)
        correct_text = _answer_text(predicate, ref_desc)

        distractor_preds, prov = sample_spatial_distractors(predicate)
        distractor_texts = [_answer_text(p, ref_desc) for p in distractor_preds]
        for p, dp in zip(prov, distractor_texts):
            p.option = dp
        if len(distractor_texts) < 2:
            continue

        cams = [c for c in (rel.camera,) if c]
        facts = (
            f"Reference object: {ref_desc}\n"
            f"Subject object: {subj_desc}\n"
            f"Spatial relation (correct answer): {correct_text}"
            + (f"\nMetric distance: {rel.distance:.1f} m" if rel.distance is not None else "")
        )

        fallback_q = f"Where is {subj_desc} relative to {ref_desc}?"
        rendered = render_multiple_choice(
            facts, correct_text, distractor_texts, _load_prompt(), rng,
            use_gpt=use_gpt, fallback_question=fallback_q,
        )

        meta = STSGMetadata(
            grounding_events=[],
            target_event=None,
            relation=predicate,
            entities=sorted({rel.subject, rel.reference}),
            cameras=sorted(cams),
            num_cameras=len(set(cams)),
            frame_span=[rel.frame, rel.frame],
            distractor_provenance=prov,
            extra={
                "spatial_relation_id": rel.id,
                "distance_m": rel.distance,
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
