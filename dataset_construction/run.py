from __future__ import annotations

import argparse
import json
import os
import random
import sys

import tqdm
import yaml

ALL_DATASETS = ["nuscenes", "meva", "ego-exo4d", "agibot"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stsg import STSG, save_stsg
from generation.registry import generator_for, DATASET_CATEGORIES


TWO_STEP = {"agibot", "ego-exo4d"}

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    base = os.path.dirname(os.path.abspath(path))
    cfg["intermediate"] = os.path.normpath(os.path.join(base, cfg.get("intermediate", "./intermediate")))
    return cfg


def _prepare_input(dataset: str, cfg: dict, refresh: bool, max_scenes: int | None) -> str:
    raw = cfg["inputs"][dataset]
    if dataset not in TWO_STEP:
        return raw


    suffix = f"_n{max_scenes}" if max_scenes else ""
    compiled = os.path.join(cfg["intermediate"], dataset, f"compiled{suffix}.json")
    if refresh or not os.path.exists(compiled):
        print(f"[{dataset}] create_dataset: reading raw {raw} -> {compiled}")
        if dataset == "agibot":
            from builders.agibot import create_dataset as cd
        else:
            from builders.ego_exo4d import create_dataset as cd
        cd.build(raw, compiled, limit=max_scenes)
    else:
        print(f"[{dataset}] using cached {compiled} (pass --refresh to rebuild)")
    return compiled


def _scene_iter(dataset: str, input_path):
    if dataset == "agibot":
        from builders.agibot import build as b

        yield from b.iter_stsgs(input_path)
    elif dataset == "ego-exo4d":
        from builders.ego_exo4d import build as b

        yield from b.iter_stsgs(input_path)
    elif dataset == "meva":
        from builders.meva import build as b

        yield from b.iter_stsgs(input_path)
    elif dataset == "nuscenes":
        from builders.nuscenes import build as b

        yield from b.iter_stsgs(input_path)
    else:
        raise ValueError(f"unknown dataset: {dataset}")


def validate(entry_meta, graph: STSG) -> list[str]:
    issues = []
    ids_events = {e.id for e in graph.events}
    ids_ent = {e.uid for e in graph.entities}
    ids_cam = {c.id for c in graph.cameras}
    for eid in entry_meta.grounding_events:
        if eid not in ids_events:
            issues.append(f"grounding event {eid} not in STSG")
    if entry_meta.target_event and entry_meta.target_event not in ids_events:
        issues.append(f"target event {entry_meta.target_event} not in STSG")
    for ent in entry_meta.entities:
        if ent and ent not in ids_ent:
            issues.append(f"entity {ent} not in STSG")
    for cam in entry_meta.cameras:
        if cam and cam not in ids_cam:
            issues.append(f"camera {cam} not in STSG")
    return issues


def _record(entry, graph: STSG) -> dict:
    rec = entry.to_dict()
    rec["question_type"] = entry.category
    rec["video_paths"] = graph.metadata.get("video_paths", [])


    rec["camera_names"] = graph.metadata.get("camera_names") or {
        c.id: (c.name or c.id) for c in graph.cameras
    }
    return rec


def _run_dataset(dataset: str, cfg: dict, args) -> None:

    targets = None
    if args.paper:
        targets = cfg.get("distribution", {}).get(dataset)
        if not targets:
            raise SystemExit(f"no distribution for '{dataset}' in config")
        categories = list(targets.keys())


        build_limit = max(targets.values()) * 2
    else:
        categories = DATASET_CATEGORIES[dataset]
        build_limit = args.max_scenes


    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",")}
        categories = [c for c in categories if c in wanted]
        if not categories:
            raise SystemExit(f"no matching categories for '{dataset}' in {sorted(wanted)}")

    input_path = _prepare_input(dataset, cfg, args.refresh, build_limit)


    out_dir = os.path.join(cfg["output"], dataset)
    os.makedirs(out_dir, exist_ok=True)
    by_category: dict[str, list] = {c: [] for c in categories}
    stsg_dir = out_dir.rstrip("/") + "_stsg"
    if args.save_stsg:
        os.makedirs(stsg_dir, exist_ok=True)

    def _full(cat: str) -> bool:
        return targets is not None and len(by_category[cat]) >= targets[cat]


    total = sum(targets[c] for c in categories) if targets else None
    bar = tqdm.tqdm(total=total, desc=dataset, unit="q")

    n_scenes = 0
    n_issues = 0
    for graph in _scene_iter(dataset, input_path):
        if args.max_scenes and n_scenes >= args.max_scenes:
            break
        if targets is not None and all(_full(c) for c in categories):
            break
        n_scenes += 1
        if args.save_stsg:
            save_stsg(graph, os.path.join(stsg_dir, f"{graph.scene_id}.json"))

        rng = random.Random(f"{args.seed}:{graph.scene_id}".__hash__() & 0xFFFFFFFF)
        for cat in categories:
            if _full(cat):
                continue
            gen = generator_for(dataset, cat)
            kwargs = {"count": args.per_category, "use_gpt": not args.no_gpt}
            if cat == "summarization":
                kwargs["count"] = 1
            try:
                entries = gen(graph, rng, **kwargs)
            except Exception as ex:
                print(f"[warn] {graph.scene_id}/{cat}: {ex}", file=sys.stderr)
                entries = []
            for e in entries:
                if _full(cat):
                    break
                if e.stsg_metadata is not None:
                    issues = validate(e.stsg_metadata, graph)
                    if issues:
                        n_issues += len(issues)
                        continue
                by_category[cat].append(_record(e, graph))
                bar.update(1)
        bar.set_postfix({c: len(by_category[c]) for c in categories}, refresh=False)
    bar.close()

    for cat, records in by_category.items():
        path = os.path.join(out_dir, f"qa_{cat}.json")
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        tgt = f" / target {targets[cat]}" if targets else ""
        short = " [SHORT]" if (targets and len(records) < targets[cat]) else ""
        print(f"wrote {len(records):5d}{tgt} -> {path}{short}")

    print(f"out={out_dir}  scenes={n_scenes}  dropped_for_provenance={n_issues}")
    if targets and any(len(by_category[c]) < targets[c] for c in categories):
        print("WARNING: some categories fell short of target (ran out of scenes "
              "or qualifying content). Counts above are the max achievable.")
    if args.save_stsg:
        print(f"stsg dumps -> {stsg_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    choices=["agibot", "ego-exo4d", "meva", "nuscenes"],
                    help="dataset to build; omit to run ALL four sequentially")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-category", type=int, default=3,
                    help="max questions per category per scene")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="cap number of scenes processed (for quick test runs)")
    ap.add_argument("--paper", action="store_true",
                    help="generate exactly the paper's per-category targets "
                         "(config.distribution); stops each category at its target")
    ap.add_argument("--categories", default=None,
                    help="comma-separated subset of categories to (re)generate; "
                         "other categories' qa_*.json files are left untouched")
    ap.add_argument("--no-gpt", action="store_true",
                    help="use deterministic template phrasing instead of GPT")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild compiled.json for two-step datasets (agibot/ego-exo4d)")
    ap.add_argument("--save-stsg", action="store_true",
                    help="also write intermediate STSGs (to <out>_stsg/, never inside <out>)")
    args = ap.parse_args()

    cfg = load_config(args.config)


    from generation import render
    render.configure(cfg.get("models", {}).get("renderer", "gpt-5.2"))

    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    for ds in datasets:
        print(f"\n================ {ds} ================")
        _run_dataset(ds, cfg, args)


if __name__ == "__main__":
    main()
