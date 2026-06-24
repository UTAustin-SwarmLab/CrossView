from __future__ import annotations

import glob
import json
import os


def read_dataset(dataset_path: str, limit: int | None = None) -> list:
    task_info_dir = os.path.join(dataset_path, "task_info")
    json_files = glob.glob(os.path.join(task_info_dir, "*.json"))
    task_numbers = []
    for filepath in json_files:
        filename = os.path.basename(filepath)
        task_num = filename.replace("task_", "").replace(".json", "")
        task_numbers.append(int(task_num))

    data = []
    n_episodes = 0
    for task_num in sorted(task_numbers):
        if limit is not None and n_episodes >= limit:
            break
        filepath = os.path.join(task_info_dir, f"task_{task_num}.json")
        with open(filepath, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        episodes_array = []
        for episode in task_data:
            videos_dir = os.path.join(
                dataset_path, "observations", str(task_num), str(episode["episode_id"]), "videos"
            )
            video_paths = glob.glob(os.path.join(videos_dir, "*"))
            if video_paths:
                episodes_array.append({
                    "id": episode["episode_id"],
                    "annotations": episode["label_info"]["action_config"],
                    "paths": sorted(video_paths),
                })

        if episodes_array:
            data.append({
                "task": task_num,
                "task_name": task_data[0]["task_name"],
                "scene_descriptions": task_data[0]["init_scene_text"],
                "episodes": episodes_array,
            })
            n_episodes += len(episodes_array)
    return data


def build(dataset_path: str, out_path: str, limit: int | None = None) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = read_dataset(dataset_path, limit=limit)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path
