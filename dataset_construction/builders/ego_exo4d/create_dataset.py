from __future__ import annotations

import glob
import json
import os


def read_dataset(dataset_path: str, limit: int | None = None) -> dict:
    with open(os.path.join(dataset_path, "takes.json")) as f:
        takes = json.load(f)
    with open(os.path.join(dataset_path, "annotations", "splits.json")) as f:
        splits_data = json.load(f)
    with open(os.path.join(dataset_path, "metadata.json")) as f:
        metadata_data = json.load(f)

    with open(os.path.join(dataset_path, "annotations", "keystep_train.json")) as f:
        keystep_train = json.load(f)
    with open(os.path.join(dataset_path, "annotations", "keystep_val.json")) as f:
        keystep_val = json.load(f)
    keystep_annotations_data = keystep_train["annotations"] | keystep_val["annotations"]
    keystep_taxonomy_data = keystep_train["taxonomy"] | keystep_val["taxonomy"]

    with open(os.path.join(dataset_path, "annotations", "atomic_descriptions_train.json")) as f:
        atomic_descriptions_train = json.load(f)
    with open(os.path.join(dataset_path, "annotations", "atomic_descriptions_val.json")) as f:
        atomic_descriptions_val = json.load(f)
    atomic_descriptions_data = atomic_descriptions_train["annotations"] | atomic_descriptions_val["annotations"]

    def get_parent_hierarchy(node_id, taxonomy_for_scenario):
        node_info = taxonomy_for_scenario.get(str(node_id))
        if not node_info:
            return None
        current_node = {"node_id": node_info["id"], "node_name": node_info["name"]}
        if node_info["parent_id"] is not None:
            current_node["parent"] = get_parent_hierarchy(node_info["parent_id"], taxonomy_for_scenario)
        else:
            current_node["parent"] = None
        return current_node

    data = {}
    for take in takes:
        if limit is not None and len(data) >= limit:
            break
        if not (take["validated"] and take["is_narrated"]):
            continue

        keystep_take = keystep_annotations_data.get(take["take_uid"], {})
        keystep_array = []
        if keystep_take:
            for s in keystep_take["segments"]:
                if str(s["step_id"]) in keystep_taxonomy_data[keystep_take["scenario"]]:
                    keystep_array.append({
                        "start_time": s["start_time"],
                        "end_time": s["end_time"],
                        "category": keystep_take["scenario"],
                        "is_essential": s["is_essential"],
                        "description": s["step_description"],
                        "node": get_parent_hierarchy(s["step_id"], keystep_taxonomy_data[keystep_take["scenario"]]),
                    })

        atomic_descriptions_take = atomic_descriptions_data.get(take["take_uid"], [])
        atomic_descriptions_array = []
        for video_set in atomic_descriptions_take:
            to_add = []
            for description in video_set.get("descriptions", []):
                if not description["unsure"]:
                    to_add.append({
                        "timestamp": description["timestamp"],
                        "text": description["text"],
                        "subject": description["subject"],
                        "best_camera": (description.get("best_exo") or {}).get("cam_id"),
                    })
            atomic_descriptions_array.append(to_add)

        data[take["take_name"]] = {
            "take_name": take["take_name"],
            "take_uid": take["take_uid"],
            "task_id": metadata_data["tasks"][str(take["task_id"])],
            "best_camera": take["best_exo"],
            "video_files": [
                f for pattern in ["cam*.mp4", "gp*.mp4", "*_214-1.mp4"]
                for f in glob.glob(os.path.join(
                    dataset_path, "takes", take["take_name"], "frame_aligned_videos", "downscaled", "448", pattern))
            ],
            "benchmarks": splits_data["take_uid_to_benchmark"].get(take["take_uid"], []),
            "objects": [(obj["name"], obj["object_uid"]) for obj in take["objects"]],
            "annotations": atomic_descriptions_array,
            "keystep_annotations": keystep_array,
        }
    return data


def build(dataset_path: str, out_path: str, limit: int | None = None) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = read_dataset(dataset_path, limit=limit)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path
