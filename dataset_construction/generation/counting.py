from __future__ import annotations

import random
from collections import defaultdict

from stsg import STSG
from generation.qa import QAEntry, STSGMetadata
from generation import common
from generation.distractors import sample_count_distractors

CATEGORY = "counting"


def generate(
    graph: STSG,
    rng: random.Random,
    count: int = 3,
    use_gpt: bool = True,
    min_instances: int = 1,
) -> list[QAEntry]:

    by_class: dict[str, set[str]] = defaultdict(set)
    cameras_by_class: dict[str, set[str]] = defaultdict(set)
    for ent in graph.entities:
        by_class[ent.category].add(ent.uid)
        for c in ent.cameras:
            cameras_by_class[ent.category].add(c)

    classes = [
        (cls, len(uids)) for cls, uids in by_class.items() if len(uids) >= min_instances
    ]

    classes.sort(key=lambda x: (-x[1], x[0]))
    if not classes:
        return []

    max_count = max(c for _, c in classes)
    out: list[QAEntry] = []

    for cls, n in classes:
        if len(out) >= count:
            break
        cams = sorted(cameras_by_class[cls])
        distractor_counts, prov = sample_count_distractors(
            n, rng, n=3, max_count=max_count
        )
        cls_display = cls.replace("_", " ").replace(".", " ")
        question = (
            f"How many distinct {cls_display}s appear across all camera views "
            f"in this scene? Answer with a single number."
        )

        meta = STSGMetadata(
            grounding_events=[],
            target_event=None,
            relation="count",
            entities=sorted(by_class[cls]),
            cameras=cams,
            num_cameras=len(cams),
            frame_span=common.frame_span(graph.events) if graph.events else None,
            distractor_provenance=prov,
            extra={
                "counted_class": cls,
                "count": n,
                "counted_uids": sorted(by_class[cls]),
                "candidate_distractor_counts": [int(d) for d in distractor_counts],
            },
        )
        out.append(
            QAEntry(
                question=question,
                answer=str(n),
                options=None,
                category=CATEGORY,
                dataset=graph.dataset,
                scene_id=graph.scene_id,
                stsg_metadata=meta,
            )
        )

    return out
