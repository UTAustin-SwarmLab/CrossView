from __future__ import annotations

import random
from typing import Optional

from stsg import STSG, Event
from generation.qa import DistractorProvenance


def sample_event_distractors(
    graph: STSG,
    correct: Event,
    rng: random.Random,
    n: int = 3,
    exclude: Optional[set[str]] = None,
) -> tuple[list[Event], list[DistractorProvenance]]:
    exclude = exclude or set()
    pool = [
        ev
        for ev in graph.events
        if ev.id != correct.id
        and ev.id not in exclude
        and ev.activity != correct.activity
    ]
    rng.shuffle(pool)
    chosen = pool[:n]
    prov = [
        DistractorProvenance(
            option=ev.activity,
            type="temporal",
            source_event=ev.id,
            source_entity=ev.entity,
            note="real event, wrong temporal position",
        )
        for ev in chosen
    ]
    return chosen, prov


def sample_activity_distractors(
    graph: STSG,
    correct_activity: str,
    activity_pool: list[str],
    rng: random.Random,
    n: int = 3,
) -> tuple[list[str], list[DistractorProvenance]]:
    in_scene = sorted({ev.activity for ev in graph.events} - {correct_activity})
    in_scene = [a for a in in_scene if a in activity_pool]
    out_scene = [a for a in activity_pool if a != correct_activity and a not in in_scene]
    rng.shuffle(in_scene)
    rng.shuffle(out_scene)

    chosen = (in_scene + out_scene)[:n]
    prov = []
    for a in chosen:
        in_this_scene = a in in_scene
        prov.append(
            DistractorProvenance(
                option=a,
                type="temporal" if in_this_scene else "existential",
                note="other activity in scene" if in_this_scene else "activity absent from scene",
            )
        )
    return chosen, prov


def sample_camera_distractors(
    graph: STSG,
    correct_cameras: list[str],
    rng: random.Random,
    n: int = 3,
) -> tuple[list[str], list[DistractorProvenance]]:
    pool = [c.id for c in graph.cameras if c.id not in correct_cameras]
    rng.shuffle(pool)
    chosen = pool[:n]
    prov = [
        DistractorProvenance(option=c, type="camera", source_camera=c)
        for c in chosen
    ]
    return chosen, prov


def sample_count_distractors(
    correct_count: int,
    rng: random.Random,
    n: int = 3,
    max_count: Optional[int] = None,
) -> tuple[list[int], list[DistractorProvenance]]:
    candidates: list[int] = []
    for offset in (1, 2, -1, -2):
        c = correct_count + offset
        if c >= 0 and c != correct_count:
            candidates.append(c)
    if 0 not in candidates and correct_count != 0:
        candidates.append(0)
    candidates = sorted(set(candidates))

    extra_base = (max_count if max_count is not None else correct_count) + 1
    while len(candidates) < n:
        cand = extra_base
        extra_base += 1
        if cand != correct_count and cand not in candidates:
            candidates.append(cand)

    rng.shuffle(candidates)
    chosen = candidates[:n]
    prov = [
        DistractorProvenance(option=str(c), type="count", note=f"offset {c - correct_count}")
        for c in chosen
    ]
    return chosen, prov


def sample_spatial_distractors(
    correct_predicate: str,
) -> tuple[list[str], list[DistractorProvenance]]:
    opposites = {
        "left": "right",
        "right": "left",
        "behind": "in_front",
        "in_front": "behind",
        "above": "below",
        "below": "above",
        "near": "far",
        "far": "near",
    }
    ordered = ["near", "moderate", "far"]
    if correct_predicate in ordered:


        chosen = [p for p in ordered if p != correct_predicate] + ["same_location"]
    else:
        chosen = []
        if correct_predicate in opposites:
            chosen.append(opposites[correct_predicate])
        for p in ("left", "right", "behind", "in_front"):
            if p != correct_predicate and p not in chosen:
                chosen.append(p)
        chosen = chosen[:3]
    prov = [
        DistractorProvenance(option=p, type="spatial", note="wrong direction/proximity")
        for p in chosen
    ]
    return chosen, prov
