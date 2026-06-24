from __future__ import annotations

import random

from stsg import STSG, Event
from generation.qa import QAEntry, STSGMetadata, DistractorProvenance
from generation import common
from generation.render import render_multiple_choice

CATEGORY = "event_ordering"

_PROMPT = None


def _load_prompt() -> str:
    global _PROMPT
    if _PROMPT is None:
        import os

        path = os.path.join(os.path.dirname(__file__), "prompts", "ordering.txt")
        with open(path) as f:
            _PROMPT = f.read()
    return _PROMPT


def _permutations(n: int, rng: random.Random) -> list[list[int]]:
    identity = list(range(n))
    cands: list[list[int]] = []

    rev = identity[::-1]
    if rev != identity:
        cands.append(rev)

    for _ in range(8):
        swapped = identity[:]
        i = rng.randrange(n - 1)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        if swapped != identity and swapped not in cands:
            cands.append(swapped)

    for _ in range(8):
        shuf = identity[:]
        rng.shuffle(shuf)
        if shuf != identity and shuf not in cands:
            cands.append(shuf)

    return cands[:3]


def _order_text(events: list[Event], graph: STSG, order: list[int]) -> str:
    return " -> ".join(common.describe_event(graph, events[i]) for i in order)


def generate(
    graph: STSG,
    rng: random.Random,
    count: int = 3,
    use_gpt: bool = True,
) -> list[QAEntry]:
    events_all = sorted(graph.events, key=lambda e: e.t_start)
    if len(events_all) < 3:
        return []

    out: list[QAEntry] = []
    attempts = 0
    max_attempts = 40 * max(count, 1)
    seen: set[tuple[str, ...]] = set()

    while len(out) < count and attempts < max_attempts:
        attempts += 1
        k = rng.choice([3, 4, 5])
        if len(events_all) < k:
            k = len(events_all)
        chosen = sorted(rng.sample(events_all, k), key=lambda e: e.t_start)


        if len({e.activity for e in chosen}) < k:
            continue
        sig = tuple(e.id for e in chosen)
        if sig in seen:
            continue
        seen.add(sig)

        perms = _permutations(k, rng)
        if len(perms) < 3:
            continue

        correct_text = _order_text(chosen, graph, list(range(k)))
        distractor_texts = [_order_text(chosen, graph, p) for p in perms]
        prov = [
            DistractorProvenance(
                option=t, type="temporal", note="permutation of true order"
            )
            for t in distractor_texts
        ]

        shuffled = list(range(k))
        rng.shuffle(shuffled)
        listed = "\n".join(
            f"{n + 1}. {common.describe_event(graph, chosen[i])}"
            for n, i in enumerate(shuffled)
        )
        facts = (
            "Shuffled events:\n"
            f"{listed}\n"
            f"Correct chronological order: {correct_text}"
        )

        fallback_q = (
            "Order these events as they occur in the multi-camera video: "
            + "; ".join(common.describe_event(graph, chosen[i]) for i in shuffled)
        )
        rendered = render_multiple_choice(
            facts, correct_text, distractor_texts, _load_prompt(), rng,
            use_gpt=use_gpt, fallback_question=fallback_q,
        )

        meta = STSGMetadata(
            grounding_events=[e.id for e in chosen],
            target_event=None,
            relation="chronological_order",
            entities=sorted({e.entity for e in chosen}),
            cameras=common.union_cameras(chosen),
            num_cameras=len(common.union_cameras(chosen)),
            frame_span=common.frame_span(chosen),
            distractor_provenance=prov,
            extra={
                "true_order": [e.id for e in chosen],
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
