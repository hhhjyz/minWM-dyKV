#!/usr/bin/env python3
"""Paired, stage-wise analysis for the fixed-budget motion allocation ablation."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


CASES = (
    "retr16_compression_r033",
    "motion_alloc_cam_4chunk",
    "motion_alloc_cam_content_4chunk",
    "motion_alloc_cam_content_prerope_4chunk",
)
PRIMARY_METRICS = (
    "pose_psnr",
    "pose_ssim",
    "pose_lpips",
    "closure_psnr",
    "closure_ssim",
    "closure_lpips",
)
LOWER_IS_BETTER = {metric for metric in PRIMARY_METRICS if metric.endswith("lpips")}


def _load(case_dir: Path) -> dict[int, dict]:
    path = case_dir / "eval" / "loop_closure_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {int(row["prompt_index"]): row for row in data["videos"]}
    if len(rows) != int(data["num_evaluated"]):
        raise ValueError(f"duplicate prompt index in {path}")
    return rows


def _motion_group(trajectory: str) -> str:
    actions = {
        term.split("*", 1)[0].strip()
        for term in str(trajectory).split(",")
        if term.strip()
    }
    actions.discard("n")
    if actions and actions <= {"i", "j", "k", "l"}:
        return "rotation_only"
    if actions and actions <= {"a", "d", "w", "s"}:
        return "translation_only"
    return "mixed_motion"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _bootstrap_ci(values: list[float], *, seed: int, samples: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        _mean([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(samples)
    )
    low = means[max(0, int(math.floor(0.025 * samples)))]
    high = means[min(samples - 1, int(math.ceil(0.975 * samples)) - 1)]
    return low, high


def _paired_stats(
    before: dict[int, dict],
    after: dict[int, dict],
    *,
    metric: str,
    indices: list[int],
    seed: int,
    bootstrap_samples: int,
) -> dict:
    deltas = [float(after[index][metric]) - float(before[index][metric]) for index in indices]
    low, high = _bootstrap_ci(deltas, seed=seed, samples=bootstrap_samples)
    signs = [-value if metric in LOWER_IS_BETTER else value for value in deltas]
    return {
        "n": len(deltas),
        "before_mean": _mean([float(before[index][metric]) for index in indices]),
        "after_mean": _mean([float(after[index][metric]) for index in indices]),
        "paired_delta_mean": _mean(deltas),
        "paired_delta_median": _median(deltas),
        "bootstrap_95_ci": [low, high],
        "win_rate": sum(value > 0.0 for value in signs) / len(signs),
        "tie_rate": sum(value == 0.0 for value in signs) / len(signs),
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    rows = {case: _load(args.root / case) for case in CASES}
    shared = sorted(set.intersection(*(set(value) for value in rows.values())))
    if not shared:
        raise SystemExit("No prompt indices are shared by all four cases")
    if any(set(value) != set(shared) for value in rows.values()):
        raise SystemExit("Cases do not contain the same prompt-index set")
    groups = {
        "all": shared,
        **{
            name: [
                index
                for index in shared
                if _motion_group(rows[CASES[0]][index]["trajectory"]) == name
            ]
            for name in ("rotation_only", "translation_only", "mixed_motion")
        },
    }

    results = {}
    comparisons = list(zip(CASES, CASES[1:]))
    for before_case, after_case in comparisons:
        label = f"{after_case} vs {before_case}"
        results[label] = {}
        for group, indices in groups.items():
            if not indices:
                continue
            results[label][group] = {
                metric: _paired_stats(
                    rows[before_case],
                    rows[after_case],
                    metric=metric,
                    indices=indices,
                    seed=args.seed + sum(
                        (offset + 1) * ord(character)
                        for offset, character in enumerate(metric)
                    ),
                    bootstrap_samples=args.bootstrap_samples,
                )
                for metric in PRIMARY_METRICS
            }

    lines = [
        "# Motion Allocation 20s Paired Ablation",
        "",
        f"- Shared prompts: {len(shared)}; bootstrap samples: {args.bootstrap_samples}",
        "- Each row compares adjacent stages, so only one intended mechanism changes.",
        "- Delta is `experiment - control`; LPIPS improves when delta is negative.",
        "- Win rate is direction-normalized, so larger is always better.",
        "",
    ]
    for label, by_group in results.items():
        lines.extend([f"## {label}", ""])
        for group, metrics in by_group.items():
            lines.extend(
                [
                    f"### {group} (N={next(iter(metrics.values()))['n']})",
                    "",
                    "| Metric | Control | Experiment | Paired Δ | 95% CI | Win rate |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for metric, stat in metrics.items():
                low, high = stat["bootstrap_95_ci"]
                lines.append(
                    f"| {metric} | {_fmt(stat['before_mean'])} | "
                    f"{_fmt(stat['after_mean'])} | {_fmt(stat['paired_delta_mean'])} | "
                    f"[{_fmt(low)}, {_fmt(high)}] | {stat['win_rate']:.1%} |"
                )
            lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    json_path = args.json_output or args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {"cases": CASES, "shared_prompt_indices": shared, "groups": groups, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
