from openai import OpenAI
import numpy as np
import base64
import cv2
import json
import os
import datetime
import subprocess
import tqdm
import math
import argparse
from rouge_score import rouge_scorer

OPENAI_MODELS = {"gpt-5.2", "gpt-5", "gpt-4o", "o3"}


COUNTING_CATS = {"counting"}
SUMMARIZATION_CATS = {"summarization"}


_DEFAULT_CATEGORIES = {
    "nuscenes": ["counting", "spatial", "temporal", "event_ordering", "summarization"],
    "meva": ["counting", "spatial", "temporal", "event_ordering", "best_camera", "summarization"],
    "ego-exo4d": ["temporal", "event_ordering", "best_camera", "summarization"],
    "agibot": ["temporal", "event_ordering", "summarization"],
}


def _categories_for(dataset):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    try:
        import yaml
        with open(cfg_path) as f:
            dist = yaml.safe_load(f).get("distribution", {})
        if dataset in dist:
            return list(dist[dataset].keys())
    except Exception:
        pass
    return _DEFAULT_CATEGORIES[dataset]


class VLLMClient:
    def __init__(self, api_base, model, dataset, strategy, num_frames):
        if model in OPENAI_MODELS:
            self.client = OpenAI()
            self._tokens_key = "max_completion_tokens"
        else:
            self.client = OpenAI(api_key="EMPTY", base_url=api_base)
            self._tokens_key = "max_tokens"
        self.model = model
        print(f"Using model: {self.model}")

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_clean = model.replace("/", "_")
        self._log_path = os.path.join(
            log_dir, f"log_{model_clean}_{dataset}_{strategy}_{num_frames}f_{timestamp}.json"
        )
        self._log_entries = []

    def _create(self, messages, tokens, temperature):
        if self._tokens_key == "max_completion_tokens":
            return self.client.chat.completions.create(
                model=self.model, messages=messages, max_completion_tokens=tokens,
            )
        return self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=tokens, temperature=temperature,
        )

    def _log(self, user_content, max_tokens, temperature, chat_response):
        def strip_images(content):
            return [
                {"type": c["type"], "text": c.get("text", "<image>")} if c["type"] == "image_url" else c
                for c in content
            ]
        self._log_entries.append({
            "model": self.model, "max_tokens": max_tokens, "temperature": temperature,
            "user_content": strip_images(user_content), "response": chat_response.model_dump(),
        })
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(self._log_entries, f, indent=4, ensure_ascii=False)

    def _encode_frame(self, frame):
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def sample_frames(self, frames_data, strategy_type="uniform", camera_names=None) -> list:
        user_content = []
        if strategy_type == "stitched":
            user_content.append({"type": "text", "text": "The following is a sequence of multi-camera grid images"})
            for encoded in [self._encode_frame(f) for f in frames_data]:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        else:
            frames_by_cam = frames_data
            if len(frames_by_cam) == 1:
                user_content.append({"type": "text", "text": "The following is the sequence of images"})
                for encoded in [self._encode_frame(f) for f in list(frames_by_cam.values())[0]]:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
            else:
                user_content.append({"type": "text", "text": "The following is the sequence of images from multiple cameras"})
                for cam_name, frames in frames_by_cam.items():
                    display = (camera_names or {}).get(cam_name, cam_name)
                    user_content.append({"type": "text", "text": f"{display}:"})
                    for encoded in [self._encode_frame(f) for f in frames]:
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        return user_content

    def multiple_choice(self, frames_data, question, candidates, strategy_type="uniform", camera_names=None) -> str:
        user_content = self.sample_frames(frames_data, strategy_type, camera_names)
        parsing_rule = (
            "You must only return the letter of the answer choice, and nothing else. "
            "Do not include any other symbols, information, text, or justification in your answer. "
            "For example, if the correct answer is 'a) ...', you must only return 'a'."
        )
        prompt = f"{question}\n" + "".join(f"{c}\n" for c in candidates) + f"\n[PARSING RULE]: {parsing_rule}"
        user_content.append({"type": "text", "text": prompt})

        tokens = 5 if self.model in {"gpt-5.2", "gpt-5", "o3"} else 1
        chat_response = self._create([{"role": "user", "content": user_content}], tokens, 0.0)
        self._log(user_content, tokens, 0.0, chat_response)
        result = chat_response.choices[0].message.content.lower().strip()
        return result[0] if (tokens == 5 and result) else result

    def counting(self, frames_data, question, strategy_type="uniform", camera_names=None) -> str:
        user_content = self.sample_frames(frames_data, strategy_type, camera_names)
        parsing_rule = "You must only return a single number as your answer, and nothing else."
        user_content.append({"type": "text", "text": f"{question}\n\n[PARSING RULE]: {parsing_rule}"})
        chat_response = self._create([{"role": "user", "content": user_content}], 10, 0.0)
        self._log(user_content, 10, 0.0, chat_response)
        return chat_response.choices[0].message.content.strip()

    def summarize(self, frames_data, question, strategy_type="uniform", camera_names=None) -> str:
        user_content = self.sample_frames(frames_data, strategy_type, camera_names)
        user_content.append({"type": "text", "text": question})
        chat_response = self._create([{"role": "user", "content": user_content}], 1024, 0.7)
        self._log(user_content, 1024, 0.7, chat_response)
        return chat_response.choices[0].message.content.strip()


def get_video_frame_count(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path], capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def load_frames_ffmpeg(video_path, frame_indices):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", video_path], capture_output=True, text=True)
    try:
        w, h = map(int, result.stdout.strip().split(','))
    except ValueError:
        return []
    select_expr = '+'.join(f'eq(n\\,{int(idx)})' for idx in frame_indices)
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"select={select_expr}", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
    data = result.stdout
    frame_size = w * h * 3
    frames = []
    for i in range(len(frame_indices)):
        chunk = data[i * frame_size:(i + 1) * frame_size]
        if len(chunk) == frame_size:
            frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(h, w, 3).copy())
    return frames


def create_camera_grid(frames_dict, labels_dict):
    num_cameras = len(frames_dict)
    if num_cameras == 0:
        return None
    grid_cols = math.ceil(math.sqrt(num_cameras))
    grid_rows = math.ceil(num_cameras / grid_cols)
    camera_names = sorted(frames_dict.keys())
    frames = [frames_dict[cam] for cam in camera_names]
    if not frames:
        return None
    cell_height = max(f.shape[0] for f in frames)
    cell_width = max(f.shape[1] for f in frames)
    labeled_frames = []
    for cam_name, frame in zip(camera_names, frames):
        labeled_frame = frame.copy()
        label = labels_dict.get(cam_name, cam_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
        cv2.rectangle(labeled_frame, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
        cv2.putText(labeled_frame, label, (10, 10 + th), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        labeled_frames.append(labeled_frame)
    grid_canvas = np.zeros((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)
    for idx, frame in enumerate(labeled_frames):
        row, col = idx // grid_cols, idx % grid_cols
        h, w = frame.shape[:2]
        y0 = row * cell_height + (cell_height - h) // 2
        x0 = col * cell_width + (cell_width - w) // 2
        grid_canvas[y0:y0 + h, x0:x0 + w] = frame
    return grid_canvas


def uniform_sampling_strategy(video_paths, num_samples):
    frames_by_cam = {}
    for video_path in video_paths:
        if not os.path.exists(video_path):
            continue
        cam_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_count = get_video_frame_count(video_path)
        if frame_count < num_samples:
            frame_indices = np.arange(frame_count)
        else:
            frame_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)
        frames = load_frames_ffmpeg(video_path, frame_indices)
        if frames:
            frames_by_cam[cam_name] = frames
    return frames_by_cam


def stitched_frames_sampling_strategy(video_paths, num_samples, camera_names=None):
    if not video_paths:
        return []
    video_frame_counts, valid_paths = {}, []
    for video_path in video_paths:
        if not os.path.exists(video_path):
            continue
        fc = get_video_frame_count(video_path)
        if fc > 0:
            video_frame_counts[video_path] = fc
            valid_paths.append(video_path)
    if not valid_paths:
        return []
    min_frame_count = min(video_frame_counts.values())
    if min_frame_count < num_samples:
        frame_indices = np.arange(min_frame_count)
    else:
        frame_indices = np.linspace(0, min_frame_count - 1, num_samples, dtype=int)
    frames_by_video = {}
    for video_path in valid_paths:
        cam_name = os.path.splitext(os.path.basename(video_path))[0]
        frames = load_frames_ffmpeg(video_path, frame_indices)
        if frames:
            frames_by_video[cam_name] = frames
    if not frames_by_video:
        return []
    labels_dict = {c: (camera_names or {}).get(c, c) for c in frames_by_video}
    stitched_grids = []
    for time_idx in range(len(next(iter(frames_by_video.values())))):
        frames_at_time = {c: f[time_idx] for c, f in frames_by_video.items() if time_idx < len(f)}
        grid = create_camera_grid(frames_at_time, labels_dict)
        if grid is not None:
            stitched_grids.append(grid)
    return stitched_grids


import re


def counting_partial_credit(predicted, correct_answer):
    def num(s):
        m = re.search(r"-?\d+\.?\d*", str(s))
        return float(m.group()) if m else None
    pred, gt = num(predicted), num(correct_answer)
    if gt is None or pred is None:
        return 0.0
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1 - abs(pred - gt) / abs(gt))


def run_category_experiment(category_name, category_dataset, vllm_client, model_name, num_frames, dataset, strategy):
    results, correct, total, rouge_scores, counting_scores = {}, 0, 0, [], []
    print(f"\nProcessing '{category_name}' with {strategy} strategy...")
    for key in tqdm.tqdm(list(category_dataset.keys()), desc=category_name):
        entry = category_dataset[key]
        if not entry.get("video_paths"):
            continue
        camera_names = entry.get("camera_names", {})
        if strategy == "uniform":
            frames_data = uniform_sampling_strategy(entry["video_paths"], num_frames)
        else:
            frames_data = stitched_frames_sampling_strategy(entry["video_paths"], num_frames, camera_names=camera_names)
        if not frames_data:
            continue

        question, candidates, correct_answer = entry["question"], entry["candidates"], entry["correct_answer"]
        qtype = entry.get("question_type", category_name)
        try:
            if qtype in COUNTING_CATS:
                predicted = vllm_client.counting(frames_data, question, strategy, camera_names)
                pc = counting_partial_credit(predicted, correct_answer)
                exact = 1 if predicted.strip() == correct_answer.strip() else 0
                counting_scores.append(pc); total += 1
                results[key] = {"question": question, "predicted_answer": predicted,
                                "correct_answer": correct_answer, "partial_credit": pc,
                                "is_correct": exact, "question_type": qtype, "strategy": strategy}
            elif qtype in SUMMARIZATION_CATS:
                generated = vllm_client.summarize(frames_data, question, strategy, camera_names)
                raw = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(correct_answer, generated)
                scores = {k: raw[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}
                rouge_scores.append(scores); total += 1
                results[key] = {"question": question, "generated_answer": generated,
                                "reference_answer": correct_answer, "rouge_scores": scores,
                                "question_type": qtype, "strategy": strategy}
            else:
                predicted = vllm_client.multiple_choice(frames_data, question, candidates, strategy, camera_names)
                is_correct = 1 if predicted == correct_answer else 0
                correct += is_correct; total += 1
                results[key] = {"question": question, "predicted_answer": predicted,
                                "correct_answer": correct_answer, "is_correct": is_correct,
                                "question_type": qtype, "strategy": strategy}
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue

    model_name_clean = model_name.replace("/", "_")
    output_dir = os.path.join(os.path.dirname(__file__), model_name_clean, dataset, strategy)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name_clean}_{dataset}_{strategy}_{category_name}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    if rouge_scores:
        avg = {k: sum(s[k] for s in rouge_scores) / len(rouge_scores) for k in ("rouge1", "rouge2", "rougeL")}
        print(f"{category_name} avg ROUGE ({len(rouge_scores)}): R-1={avg['rouge1']:.4f} R-2={avg['rouge2']:.4f} R-L={avg['rougeL']:.4f}")
    elif counting_scores:
        pc = sum(counting_scores) / len(counting_scores)
        print(f"{category_name} partial-credit: {pc:.2%} ({len(counting_scores)})")
    else:
        acc = correct / total if total else 0
        print(f"{category_name} accuracy: {acc:.2%} ({correct}/{total})")
    return {"correct": correct, "total": total, "rouge_scores": rouge_scores,
            "counting_scores": counting_scores}


def run_experiment(datasets_by_category, vllm_client, model_name, num_frames, dataset, strategy):
    all_stats = {}
    for category_name, category_dataset in datasets_by_category.items():
        all_stats[category_name] = run_category_experiment(
            category_name, category_dataset, vllm_client, model_name, num_frames, dataset, strategy)

    model_name_clean = model_name.replace("/", "_")
    lines = ["=" * 50, f"Final Results ({strategy} strategy, {num_frames} frames):", "=" * 50]
    mc_correct = mc_total = 0
    for category, stats in all_stats.items():
        if stats["rouge_scores"]:
            rs = stats["rouge_scores"]
            avg = {k: sum(s[k] for s in rs) / len(rs) for k in ("rouge1", "rouge2", "rougeL")}
            lines.append(f"{category:20s}: R-1={avg['rouge1']:.4f} R-2={avg['rouge2']:.4f} R-L={avg['rougeL']:.4f} ({len(rs)})")
        elif stats.get("counting_scores"):
            cs = stats["counting_scores"]
            lines.append(f"{category:20s}: {sum(cs) / len(cs):.2%} partial-credit ({len(cs)})")
        else:
            acc = stats["correct"] / stats["total"] if stats["total"] else 0
            lines.append(f"{category:20s}: {acc:.2%} ({stats['correct']}/{stats['total']})")
            mc_correct += stats["correct"]; mc_total += stats["total"]
    lines.append("=" * 50)
    if mc_total:
        lines.append(f"{'MC Overall':20s}: {mc_correct / mc_total:.2%} ({mc_correct}/{mc_total})")
    lines.append("=" * 50)
    table = "\n".join(lines)
    print("\n" + table)

    results_dir = os.path.join(os.path.dirname(__file__), model_name_clean, dataset)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "results.txt")
    mode = "a" if os.path.exists(results_file) else "w"
    with open(results_file, mode) as f:
        if mode == "a":
            f.write("\n\n")
        f.write(table + "\n")


def load_datasets(dataset, data_dir):
    categories = _categories_for(dataset)
    datasets_by_category = {}
    for category in categories:
        file_path = os.path.join(data_dir, f"qa_{category}.json")
        if not os.path.exists(file_path):
            print(f"  (skip missing {file_path})")
            continue
        with open(file_path) as f:
            data = json.load(f)

        category_dict = {}
        for i, entry in enumerate(data):
            if not all(k in entry for k in ("question", "answer", "video_paths")):
                continue
            scene = entry.get("scene_id", "scene")
            qtype = entry.get("question_type", category)
            key = f"{scene}_{qtype}_{i}"
            if entry.get("options") is not None:
                candidates = entry["options"]
                correct_answer = entry["answer"].lower()
            else:
                candidates = None
                correct_answer = entry["answer"]
            category_dict[key] = {
                "question": entry["question"],
                "candidates": candidates,
                "correct_answer": correct_answer,
                "question_type": qtype,
                "video_paths": entry.get("video_paths", []),
                "camera_names": entry.get("camera_names", {}),
            }
        print(f"Using {len(category_dict)} {category} questions")
        datasets_by_category[category] = category_dict
    return datasets_by_category


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(_DEFAULT_CATEGORIES), default="meva")
    parser.add_argument("--data_dir", required=True, help="directory holding qa_<category>.json")
    parser.add_argument("--strategy", choices=["uniform", "stitched"], default="uniform")
    parser.add_argument("--gpu", type=int, default=0, help="sets api_base to localhost:800{gpu} for local vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()

    datasets_by_category = load_datasets(args.dataset, args.data_dir)
    total = sum(len(d) for d in datasets_by_category.values())
    print(f"\nLoaded {total} questions across {len(datasets_by_category)} categories ({args.dataset})")

    vllm_client = VLLMClient(
        api_base=f"http://localhost:800{args.gpu}/v1",
        model=args.model, dataset=args.dataset, strategy=args.strategy, num_frames=args.num_frames,
    )
    run_experiment(datasets_by_category, vllm_client, vllm_client.model, args.num_frames, args.dataset, args.strategy)


if __name__ == "__main__":
    main()
