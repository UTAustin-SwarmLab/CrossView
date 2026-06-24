<div align="center">

# CrossView: A Multi-Camera Video Question-Answering Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](#) [![Paper](https://img.shields.io/badge/Paper-pdf-green.svg)](#) [![Website](https://img.shields.io/badge/ProjectWebpage-crossview-orange.svg)](#) [![GitHub](https://img.shields.io/badge/Code-Source--Code-blue.svg)](#)

</div>

### Table of Contents
- [Overview](#overview)
- [Benchmark](#benchmark)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

<a name="overview"></a>
## :mega: Overview

<b>TL;DR: CrossView is a multi-camera video question-answering benchmark spanning autonomous driving, surveillance, egocentric/exocentric video, and robotics. Questions are generated programmatically from a Spatio-Temporal Scene Graph (STSG), so every answer and distractor is grounded in the source annotations, and the tasks require integrating evidence across many simultaneous viewpoints.</b>

> Video understanding benchmarks have long centered on single-camera settings, with modern multi-modal language models achieving strong performance across image and video tasks. Yet the real world is overwhelmingly centered on multi-camera networks: autonomous vehicles, security networks, and robotic systems all gather data across many simultaneous viewpoints. This is not simply "more" of the single-camera problem; it is fundamentally different. Multi-camera video reasoning requires handling context that scales with the number of views, resolving occlusions only visible from a subset of cameras, determining which views are most relevant, and integrating evidence across perspectives that may overlap or diverge. Current multi-modal language models struggle with exactly these challenges, yet no benchmark systematically targets them. We introduce CrossView, a multi-camera video question-answering benchmark spanning autonomous driving, security surveillance, egocentric/exocentric video, and robotics. Evaluations across both proprietary models, such as GPT-5.2, and open-source models like Qwen3-VL reveal consistently low accuracy, with open-source models trailing by a significant margin. Performance scales strongly with a model's ability to jointly process multiple viewpoints, positioning CrossView as a rigorous benchmark for multi-camera video-language understanding.

<a name="benchmark"></a>
## :bar_chart: Benchmark

CrossView draws from four source datasets across four domains and poses six categories of multi-camera reasoning questions (~6,000 total).

**Domains**
| Domain | Source | Cameras / question |
| --- | --- | --- |
| Autonomous driving | nuScenes | 6 (fixed rig) |
| Security surveillance | MEVA | up to 16 |
| Egocentric / exocentric | Ego-Exo4D | 4–7 |
| Robotics | AgiBot | 1–3 |

**Question categories**
- **Temporal** — order/relation of events across views (for nuScenes, this is spatio-temporal reasoning).
- **Event Ordering** — reconstruct the chronological sequence of events.
- **Spatial** — directional / proximity relations between objects in 3D.
- **Counting** — count unique object instances across all cameras (partial-credit scored).
- **Best Camera** — identify the most informative viewpoint for an event.
- **Summarization** — holistic free-form scene summary (ROUGE + LLM-as-judge).

Each question carries an `stsg_metadata` provenance block (grounding events, target, entities, cameras, frame span, and where each distractor was sampled) so the ground truth is mechanically verifiable without human relabeling.

<a name="installation"></a>
## :hammer: Installation

```bash
# system dependency: ffmpeg (used to extract frames from the videos)
sudo apt-get install -y ffmpeg

# python dependencies
pip install openai numpy opencv-python rouge-score pyyaml tqdm

# the GPT renderer (dataset construction) and the LLM-as-judge (evaluation) use the
# OpenAI API; open-source models are evaluated via a local vLLM server.
export OPENAI_API_KEY=sk-...
```

Source videos and annotations are read from the paths configured in `config.yaml` (see `inputs:`). Evaluating open-source models additionally requires a [vLLM](https://github.com/vllm-project/vllm) server serving the model at `localhost:800{gpu}/v1`.

<a name="usage"></a>
## :tv: Usage

All input/output paths, the generation/judge models, and the per-category target counts live in `config.yaml`.

### 1. Build the benchmark

```bash
cd dataset_construction

# generate the exact paper distribution (~6,000 questions) for all four datasets
python run.py --paper

# or a single dataset / category
python run.py --dataset meva --paper
python run.py --dataset nuscenes --paper --categories temporal

# quick dry run with deterministic phrasing (no GPT)
python run.py --dataset agibot --max-scenes 2 --per-category 2 --no-gpt
```

Output is written to `<output>/<dataset>/qa_<category>.json` (and nothing else), where `<output>` is set in `config.yaml`.

### 2. Evaluate a model

```bash
cd vqa

# proprietary model via OpenAI API
python vqa.py --dataset meva --data_dir /path/to/output/meva \
    --model gpt-5.2 --num_frames 4

# open-source model via local vLLM (sets api_base to localhost:800{gpu}/v1)
python vqa.py --dataset nuscenes --data_dir /path/to/output/nuscenes \
    --model Qwen/Qwen3-VL-8B-Instruct --gpu 0 --num_frames 8
```

Scoring: multiple-choice = exact-match accuracy, counting = partial credit `max(0, 1 - |pred-gt|/gt)`, summarization = ROUGE.

### 3. (Optional) LLM-as-judge for summarization

```bash
cd vqa
python summarization_evaluator.py \
    <model>/<dataset>/uniform/<model>_<dataset>_uniform_summarization.json
```

<a name="citation"></a>
## :clipboard: Citation

```bibtex

```
