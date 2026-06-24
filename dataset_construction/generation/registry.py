from __future__ import annotations

from generation import (
    temporal,
    event_ordering,
    spatial,
    counting,
    best_camera,
    summarization,
    spatio_temporal,
)

GENERATORS = {
    "temporal": temporal.generate,
    "event_ordering": event_ordering.generate,
    "spatial": spatial.generate,
    "counting": counting.generate,
    "best_camera": best_camera.generate,
    "summarization": summarization.generate,
}


CATEGORY_OVERRIDES = {
    "nuscenes": {"temporal": spatio_temporal.generate},
}


def generator_for(dataset: str, category: str):
    return CATEGORY_OVERRIDES.get(dataset, {}).get(category, GENERATORS[category])


DATASET_CATEGORIES = {
    "nuscenes": ["counting", "spatial", "temporal", "event_ordering", "summarization"],
    "meva": ["counting", "spatial", "temporal", "event_ordering", "best_camera", "summarization"],
    "ego-exo4d": ["temporal", "event_ordering", "best_camera", "summarization"],
    "agibot": ["temporal", "event_ordering", "summarization"],
}
