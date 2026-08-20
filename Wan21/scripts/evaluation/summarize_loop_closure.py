#!/usr/bin/env python3
"""Build two Markdown closure tables (MAG/LPIPS matching and pose matching) from per-case results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _fmt(value: Any, precision: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{precision}f}"


def _signed(value: Any, precision: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:+.{precision}f}"


def _mean(values: list[float]) -> float | None:
    valid = [v for v in values if v is not None and math.isfinite(v)]
    return sum(valid) / len(valid) if valid else None


def _load_case_summary(case_dir: Path) -> dict[str, Any] | None:
    json_path = case_dir / "eval" / "loop_closure_metrics.json"
    if not json_path.is_file():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Output root containing per-case subdirectories.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the Markdown report.")
    parser.add_argument("--baseline-case", default="baseline")
    args = parser.parse_args()

    case_dirs = sorted(
        d for d in args.root.iterdir()
        if d.is_dir() and (d / "eval" / "loop_closure_metrics.json").is_file()
    )
    if not case_dirs:
        raise SystemExit(f"No cases with eval/loop_closure_metrics.json found under {args.root}")

    summaries: dict[str, dict[str, Any]] = {}
    for case_dir in case_dirs:
        summaries[case_dir.name] = _load_case_summary(case_dir)

    baseline_metrics = summaries.get(args.baseline_case, {}).get("metrics", {})

    # Collect per-case mean metrics
    case_names = list(summaries.keys())

    def metric_row(case: str, field: str) -> float | None:
        return summaries[case].get("metrics", {}).get(field)

    # ---- Table 1: MAG / LPIPS matching ----
    headers1 = [
        "Case", "N",
        "PSNR ↑", "Δ",
        "SSIM ↑", "Δ",
        "LPIPS ↓", "Δ",
        "Unique ↑", "TempMAE ↓", "Reverse ↓",
    ]
    rows1 = []
    for case in case_names:
        m = summaries[case].get("metrics", {})
        n = summaries[case].get("num_evaluated", 0)
        psnr = m.get("lpips_psnr")
        ssim = m.get("lpips_ssim")
        lpips = m.get("lpips_lpips")
        unique = m.get("lpips_match_unique_ratio")
        tempmae = m.get("lpips_match_temporal_mae_normalized")
        reverse = m.get("lpips_match_reverse_violation_ratio")
        b_psnr = baseline_metrics.get("lpips_psnr")
        b_ssim = baseline_metrics.get("lpips_ssim")
        b_lpips = baseline_metrics.get("lpips_lpips")
        rows1.append([
            case, str(n),
            _fmt(psnr, 3), _signed((psnr - b_psnr) if psnr is not None and b_psnr is not None else None, 3),
            _fmt(ssim), _signed((ssim - b_ssim) if ssim is not None and b_ssim is not None else None),
            _fmt(lpips), _signed((lpips - b_lpips) if lpips is not None and b_lpips is not None else None),
            _fmt(unique), _fmt(tempmae), _fmt(reverse),
        ])

    # ---- Table 2: Pose matching ----
    headers2 = [
        "Case", "N",
        "PSNR ↑", "Δ",
        "SSIM ↑", "Δ",
        "LPIPS ↓", "Δ",
        "Unique ↑", "TempMAE ↓", "Reverse ↓",
    ]
    rows2 = []
    for case in case_names:
        m = summaries[case].get("metrics", {})
        n = summaries[case].get("num_evaluated", 0)
        psnr = m.get("pose_psnr")
        ssim = m.get("pose_ssim")
        lpips = m.get("pose_lpips")
        unique = m.get("pose_match_unique_ratio")
        tempmae = m.get("pose_match_temporal_mae_normalized")
        reverse = m.get("pose_match_reverse_violation_ratio")
        b_psnr = baseline_metrics.get("pose_psnr")
        b_ssim = baseline_metrics.get("pose_ssim")
        b_lpips = baseline_metrics.get("pose_lpips")
        rows2.append([
            case, str(n),
            _fmt(psnr, 3), _signed((psnr - b_psnr) if psnr is not None and b_psnr is not None else None, 3),
            _fmt(ssim), _signed((ssim - b_ssim) if ssim is not None and b_ssim is not None else None),
            _fmt(lpips), _signed((lpips - b_lpips) if lpips is not None and b_lpips is not None else None),
            _fmt(unique), _fmt(tempmae), _fmt(reverse),
        ])

    # ---- Table 3: Exact endpoint closure (shared by both modes) ----
    headers3 = ["Case", "Closure PSNR ↑", "Δ", "Closure SSIM ↑", "Δ", "Closure LPIPS ↓", "Δ"]
    rows3 = []
    for case in case_names:
        m = summaries[case].get("metrics", {})
        psnr = m.get("closure_psnr")
        ssim = m.get("closure_ssim")
        lpips = m.get("closure_lpips")
        b_psnr = baseline_metrics.get("closure_psnr")
        b_ssim = baseline_metrics.get("closure_ssim")
        b_lpips = baseline_metrics.get("closure_lpips")
        rows3.append([
            case,
            _fmt(psnr, 3), _signed((psnr - b_psnr) if psnr is not None and b_psnr is not None else None, 3),
            _fmt(ssim), _signed((ssim - b_ssim) if ssim is not None and b_ssim is not None else None),
            _fmt(lpips), _signed((lpips - b_lpips) if lpips is not None and b_lpips is not None else None),
        ])

    # ---- Table 4: LPIPS-vs-pose matching disagreement (diagnostic) ----
    headers4 = [
        "Case",
        "LPIPS TempMAE ↓", "Pose TempMAE ↓",
        "LPIPS Reverse ↓", "Pose Reverse ↓",
        "LPIPS Unique ↑", "Pose Unique ↑",
    ]
    rows4 = []
    for case in case_names:
        m = summaries[case].get("metrics", {})
        rows4.append([
            case,
            _fmt(m.get("lpips_match_temporal_mae_normalized")),
            _fmt(m.get("pose_match_temporal_mae_normalized")),
            _fmt(m.get("lpips_match_reverse_violation_ratio")),
            _fmt(m.get("pose_match_reverse_violation_ratio")),
            _fmt(m.get("lpips_match_unique_ratio")),
            _fmt(m.get("pose_match_unique_ratio")),
        ])

    lines = [
        "# Loop Closure Evaluation Report",
        "",
        f"- Root: `{args.root}`",
        f"- Cases: {len(case_names)}; baseline: `{args.baseline_case}`",
        "- Metrics: LPIPS-Alex, 256 px evaluation width, 30 videos per case.",
        "- `Δ` is the paired mean difference relative to baseline.",
        "- Direction: PSNR/SSIM/Unique ↑; LPIPS/TempMAE/Reverse ↓.",
        "",
        "## 1. MAG-style closure (LPIPS nearest-frame matching)",
        "",
        "Each revisit frame independently selects its LPIPS-nearest outbound frame;",
        "PSNR/SSIM/LPIPS are computed on those same matched pairs.",
        "",
        _md_table(headers1, rows1),
        "",
        "## 2. Pose-driven closure (camera SE(3) nearest-frame matching)",
        "",
        "Each revisit frame is matched to the outbound frame with the smallest camera pose",
        "distance (translation + rotation degrees). For exact string-loop trajectories this is",
        "the deterministic temporal mirror. PSNR/SSIM use those pose-matched pairs;",
        "LPIPS is measured on the same pose-matched pairs (not re-minimized).",
        "",
        _md_table(headers2, rows2),
        "",
        "## 3. Exact endpoint closure (frame 0 vs closure frame)",
        "",
        "Pixel/perceptual metrics between the first frame and the closure frame.",
        "Shared by both matching modes since no matching is needed.",
        "",
        _md_table(headers3, rows3),
        "",
        "## 4. Matching diagnostics: LPIPS vs pose",
        "",
        "Pose TempMAE should be near 0 for exact loop trajectories (geometric mirror is exact).",
        "LPIPS TempMAE > Pose TempMAE indicates the model's visual content drifts from the",
        "geometrically expected correspondence, or LPIPS matches to temporally wrong frames.",
        "",
        _md_table(headers4, rows4),
        "",
        "## Interpretation notes",
        "",
        "- Loop PSNR/SSIM/LPIPS use the outbound half of the same generated rollout as an internal",
        "  pseudo-reference; they measure visual closure/memory consistency, not fidelity to real video.",
        "- MAG (LPIPS) matching tolerates different traversal speeds but can match to temporally",
        "  wrong frames. Pose matching is geometrically exact for string-loop trajectories.",
        "- If pose PSNR >> LPIPS PSNR, the model generates consistent content at the same camera pose",
        "  but LPIPS matching is confused by intermediate frames. If both are similar, matching mode",
        "  does not matter. If LPIPS > pose, LPIPS finds better visual matches than the geometric mirror",
        "  (possible content drift at the exact closure pose).",
        "- Only seed 0 is available; small differences should be treated as preliminary.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
