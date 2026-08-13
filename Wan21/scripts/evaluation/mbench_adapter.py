#!/usr/bin/env python3
"""Convert MBench-A cases to minWM inputs and package generated videos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


SUPPORTED_ACTIONS = {
    "left_then_right",
    "right_then_left",
    "forward_then_backward",
    "left_360",
    "right_360",
    "left_720",
    "right_720",
    "left_1080",
    "right_1080",
    "static",
}


def read_dataset_id(dataset_root: Path) -> str:
    config_path = dataset_root / "dataset.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"MBench dataset.yaml not found: {config_path}")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("dataset_id:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    raise ValueError(f"dataset_id is missing from {config_path}")


def action_from_condition(condition_id: str) -> str:
    for suffix in ("_10s", "_25s"):
        if condition_id.endswith(suffix):
            action = condition_id[: -len(suffix)]
            if action not in SUPPORTED_ACTIONS:
                raise ValueError(f"unsupported MBench-A action: {action}")
            return action
    raise ValueError(f"invalid MBench-A condition_id: {condition_id}")


def expected_latent_frames(condition_id: str) -> int:
    if condition_id.endswith("_10s"):
        return 40
    if condition_id.endswith("_25s"):
        return 100
    raise ValueError(f"invalid MBench-A condition_id: {condition_id}")


def trajectory_for_action(action: str, num_output_frames: int) -> str:
    steps = int(num_output_frames) - 1
    if steps <= 0:
        raise ValueError("num_output_frames must be at least 2")
    if action == "static":
        return f"n*{steps}"
    round_trips = {
        "left_then_right": ("j", "l"),
        "right_then_left": ("l", "j"),
        "forward_then_backward": ("w", "s"),
    }
    if action in round_trips:
        outward, returning = round_trips[action]
        half = steps // 2
        parts = [f"{outward}*{half}", f"{returning}*{half}"]
        if steps % 2:
            parts.append("n*1")
        return ",".join(parts)
    direction, degrees_text = action.split("_", 1)
    key = "j" if direction == "left" else "l"
    multiplier = float(degrees_text) / (3.0 * steps)
    return f"{key}@{multiplier:.10g}*{steps}"


def load_assignments(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"subset", "sample_id", "condition_id"} - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
        rows.append(row)
    return rows


def default_assignments(dataset_root: Path) -> Path:
    official = dataset_root / "models" / "hy_worldplay" / "samples.jsonl"
    if not official.is_file():
        raise FileNotFoundError(
            "official MBench-A assignments were not found at "
            f"{official}; pass --assignments explicitly"
        )
    return official


def sample_caption(dataset_root: Path, subset: str, sample_id: str) -> str:
    sample_path = dataset_root / "samples" / subset / sample_id / "sample.json"
    row = json.loads(sample_path.read_text(encoding="utf-8"))
    caption = row.get("caption") or (row.get("metadata") or {}).get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"caption is missing from {sample_path}")
    return caption.strip()


def prepare(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    if read_dataset_id(dataset_root) != "mbencha":
        raise ValueError("minWM camera cases require an MBench-A (mbencha) dataset")
    assignment_path = args.assignments.resolve() if args.assignments else default_assignments(dataset_root)
    subset_filter = (
        {value.strip() for value in args.subsets.split(",") if value.strip()}
        if args.subsets
        else set()
    )
    condition_filter = (
        {value.strip() for value in args.conditions.split(",") if value.strip()}
        if args.conditions
        else set()
    )
    rows = []
    for assignment in load_assignments(assignment_path):
        subset = assignment["subset"]
        condition = assignment["condition_id"]
        if subset_filter and subset not in subset_filter:
            continue
        if condition_filter and condition not in condition_filter:
            continue
        expected_frames = expected_latent_frames(condition)
        if int(args.num_output_frames) != expected_frames:
            raise ValueError(
                f"condition {condition} requires {expected_frames} checkpoint-aligned "
                f"latent frames, got {args.num_output_frames}"
            )
        action = action_from_condition(condition)
        rows.append(
            {
                "prompt_index": len(rows),
                "subset": subset,
                "sample_id": assignment["sample_id"],
                "condition_id": condition,
                "prompt": sample_caption(dataset_root, subset, assignment["sample_id"]),
                "trajectory": trajectory_for_action(action, args.num_output_frames),
                "num_output_frames": int(args.num_output_frames),
            }
        )
        if args.limit and len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError("no MBench-A cases matched the requested filters")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "prompts.txt").write_text(
        "".join(f"{row['prompt']}\n" for row in rows), encoding="utf-8"
    )
    (args.work_dir / "trajectories.txt").write_text(
        "".join(f"{row['trajectory']}\n" for row in rows), encoding="utf-8"
    )
    with (args.work_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"prepared {len(rows)} MBench-A cases in {args.work_dir}")


def link_video(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def load_jsonl_by_index(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["prompt_index"])] = row
    return rows


def package(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    if read_dataset_id(dataset_root) != "mbencha":
        raise ValueError("package target must be an MBench-A dataset")
    cases = load_jsonl_by_index(args.cases.resolve())
    generations = load_jsonl_by_index(args.generation_manifest.resolve())
    model_root = dataset_root / "models" / args.model_id
    output_rows = []
    for prompt_index, case in sorted(cases.items()):
        generation = generations.get(prompt_index)
        if generation is None:
            raise ValueError(f"generation is missing for prompt_index={prompt_index}")
        source = Path(generation["output_path"])
        if not source.is_file():
            raise FileNotFoundError(f"generated video not found: {source}")
        relative_video = (
            Path("outputs")
            / case["subset"]
            / case["sample_id"]
            / case["condition_id"]
            / "video.mp4"
        )
        link_video(source, model_root / relative_video, args.link_mode)
        output_rows.append(
            {
                "item_id": f"{case['subset']}:{case['sample_id']}:{case['condition_id']}",
                "dataset_id": "mbencha",
                "subset": case["subset"],
                "sample_id": case["sample_id"],
                "condition_id": case["condition_id"],
                "model_id": args.model_id,
                "media": {"videos": [{"path": str(relative_video), "role": "generated"}]},
                "artifacts": {},
                "annotations": {},
                "metadata": {
                    "generator": "minWM-dyKV",
                    "trajectory": case["trajectory"],
                    "num_output_frames": case["num_output_frames"],
                },
            }
        )
    model_root.mkdir(parents=True, exist_ok=True)
    with (model_root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"packaged {len(output_rows)} items as MBench model {args.model_id}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset-root", type=Path, required=True)
    prep.add_argument("--work-dir", type=Path, required=True)
    prep.add_argument("--assignments", type=Path)
    prep.add_argument("--subsets", default="")
    prep.add_argument("--conditions", default="")
    prep.add_argument("--num-output-frames", type=int, required=True)
    prep.add_argument("--limit", type=int)
    prep.set_defaults(func=prepare)

    pack = commands.add_parser("package")
    pack.add_argument("--dataset-root", type=Path, required=True)
    pack.add_argument("--cases", type=Path, required=True)
    pack.add_argument("--generation-manifest", type=Path, required=True)
    pack.add_argument("--model-id", required=True)
    pack.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    pack.set_defaults(func=package)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
