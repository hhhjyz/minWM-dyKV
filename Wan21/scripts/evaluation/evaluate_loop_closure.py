#!/usr/bin/env python3
"""Evaluate visual loop closure inside generated minWM-dyKV videos.

Two matching modes are computed in a single pass and reported side by side:

1. ``lpips`` (MAG-style, as in minWM-back / MAG-Bench):
   Each revisit frame independently selects its LPIPS-nearest outbound frame.
   PSNR/SSIM are then computed on those same matched pairs.  This tolerates
   different traversal speeds but can match to temporally wrong frames.

2. ``pose`` (camera-pose-driven):
   Each revisit frame is matched to the outbound frame whose camera pose is
   closest in SE(3) distance (translation + rotation angle).  For the exact
   string-loop trajectories this reduces to the deterministic temporal
   mirror, but the implementation works for any trajectory.  This tests
   whether the model generates visually consistent content at the same
   camera pose, independent of LPIPS matching noise.

The outbound half of the generated rollout serves as an internal
pseudo-reference for the revisit half; there is no external ground truth.
The turnaround frame is excluded from both segments.

Inputs are the dyKV ``generation_manifest.jsonl`` (one JSON object per
video) and the shared ``demos_loop_closure/manifest.json`` that carries
``closure_pair_decoded`` per prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "lpips_psnr",
    "lpips_ssim",
    "lpips_lpips",
    "pose_psnr",
    "pose_ssim",
    "pose_lpips",
    "closure_psnr",
    "closure_ssim",
    "closure_lpips",
    "lpips_match_unique_ratio",
    "lpips_match_temporal_mae_normalized",
    "lpips_match_reverse_violation_ratio",
    "pose_match_unique_ratio",
    "pose_match_temporal_mae_normalized",
    "pose_match_reverse_violation_ratio",
]


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[Any]) -> float | None:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(valid) / len(valid)) if valid else None


def _load_generation_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("status") in {"generated", "skipped_exists"}:
            rows.append(obj)
    return rows


def _duration_label(
    manifest: dict[str, Any], num_output_frames: int, requested_label: str | None
) -> str:
    durations = manifest.get("durations", {})
    if requested_label:
        if requested_label not in durations:
            raise ValueError(
                f"Unknown duration label {requested_label!r}; expected one of {sorted(durations)}"
            )
        return requested_label
    matches = [
        label
        for label, spec in durations.items()
        if int(spec.get("latent_frames", -1)) == num_output_frames
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely map latent_frames={num_output_frames} to manifest duration: {matches}"
        )
    return matches[0]


def _manifest_sample(
    manifest: dict[str, Any], duration_label: str, prompt_index: int
) -> dict[str, Any]:
    samples = manifest["durations"][duration_label].get("samples", [])
    for sample in samples:
        if int(sample.get("prompt_index", -1)) == prompt_index:
            return sample
    raise ValueError(
        f"No manifest sample for duration={duration_label!r}, prompt_index={prompt_index}"
    )


def _uniform_indices(length: int, maximum: int) -> list[int]:
    if length <= 0:
        return []
    if maximum <= 0 or length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [length // 2]
    denominator = maximum - 1
    return [
        (index * (length - 1) + denominator // 2) // denominator
        for index in range(maximum)
    ]


def _read_selected_video_frames(
    path: Path, selected_indices: list[int], resize_width: int, cv2, np
) -> tuple[dict[int, Any], int]:
    wanted = set(selected_indices)
    if not wanted:
        raise ValueError("No video frames selected")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: dict[int, Any] = {}
    last_needed = max(wanted)
    index = 0
    while index <= last_needed:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_width > 0 and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                frame = cv2.resize(
                    frame,
                    (resize_width, max(1, int(round(frame.shape[0] * scale)))),
                    interpolation=interpolation,
                )
            frames[index] = frame.astype(np.float32) / 255.0
        index += 1
    capture.release()

    missing = sorted(wanted.difference(frames))
    if missing:
        raise ValueError(
            f"Video {path} ended before required frame(s) {missing[:5]}"
            + ("..." if len(missing) > 5 else "")
        )
    return frames, reported_count


def _stack(frames: dict[int, Any], indices: list[int], np):
    return np.stack([frames[index] for index in indices], axis=0)


def _pixel_metrics(reference, prediction, structural_similarity, np) -> tuple[float, float]:
    if reference.shape != prediction.shape:
        raise ValueError(f"Metric input shape mismatch: {reference.shape} != {prediction.shape}")
    mse = np.mean((reference - prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
    psnr_per_frame = np.full(mse.shape, 100.0, dtype=np.float64)
    nonzero = mse > 0.0
    psnr_per_frame[nonzero] = 10.0 * np.log10(1.0 / mse[nonzero])
    psnr_per_frame = np.minimum(psnr_per_frame, 100.0)
    ssim_per_frame = [
        structural_similarity(ref, pred, data_range=1.0, channel_axis=-1)
        for ref, pred in zip(reference, prediction)
    ]
    return float(np.mean(psnr_per_frame)), float(np.mean(ssim_per_frame))


def _lpips_tensor(video, torch):
    return torch.from_numpy(video.transpose(0, 3, 1, 2).copy()).float().mul_(2.0).sub_(1.0)


def _lpips_matrix(query, reference, model, device: str, batch_size: int, torch, np):
    query_tensor = _lpips_tensor(query, torch)
    reference_tensor = _lpips_tensor(reference, torch)
    num_query = len(query_tensor)
    num_reference = len(reference_tensor)
    total_pairs = num_query * num_reference
    matrix = np.empty((num_query, num_reference), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            flat_indices = np.arange(start, end, dtype=np.int64)
            query_indices = flat_indices // num_reference
            reference_indices = flat_indices % num_reference
            score = model(
                query_tensor[query_indices].to(device),
                reference_tensor[reference_indices].to(device),
            )
            matrix.reshape(-1)[start:end] = (
                score.reshape(end - start, -1).mean(dim=1).detach().cpu().numpy()
            )
    return matrix


def _lpips_pairs(left, right, model, device: str, batch_size: int, torch, np):
    if left.shape != right.shape:
        raise ValueError(f"LPIPS pair shape mismatch: {left.shape} != {right.shape}")
    left_tensor = _lpips_tensor(left, torch)
    right_tensor = _lpips_tensor(right, torch)
    values = []
    with torch.inference_mode():
        for start in range(0, len(left_tensor), batch_size):
            end = min(start + batch_size, len(left_tensor))
            score = model(
                left_tensor[start:end].to(device),
                right_tensor[start:end].to(device),
            )
            values.extend(score.reshape(end - start, -1).mean(dim=1).detach().cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def _init_lpips(device: str, net: str):
    import torch
    import lpips

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"LPIPS device {device!r} requested, but CUDA is unavailable")
    model = lpips.LPIPS(net=net, spatial=False).eval().to(device)
    return model, device, torch


# ---------------------------------------------------------------------------
# Camera-pose utilities
# ---------------------------------------------------------------------------

def _parse_trajectory(trajectory: str, wan_root: Path, np):
    """Parse trajectory string into (T, 4, 4) w2c viewmats via dyKV wan_utils."""
    import importlib

    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))
    mod = importlib.import_module("wan_utils.camera_trajectory")
    return mod.parse_trajectory(trajectory)


def _pose_distance(pose_a: "np.ndarray", pose_b: "np.ndarray", np) -> float:
    """SE(3) distance: translation norm + rotation angle (degrees)."""
    rel = np.linalg.inv(pose_a) @ pose_b
    translation = float(np.linalg.norm(rel[:3, 3]))
    trace = float(np.clip((np.trace(rel[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    rotation_deg = math.degrees(math.acos(trace))
    return translation + rotation_deg


def _decoded_to_latent(decoded_index: int) -> int:
    """Map decoded frame index to nearest latent frame index.

    decoded = 4 * latent for the closure pair, so latent = round(decoded / 4).
    """
    return int(round(decoded_index / 4))


def _pose_distance_matrix(
    revisit_indices: list[int],
    reference_indices: list[int],
    viewmats,
    np,
) -> "np.ndarray":
    """Pose distance between each revisit decoded frame and each reference decoded frame."""
    num_revisit = len(revisit_indices)
    num_reference = len(reference_indices)
    matrix = np.empty((num_revisit, num_reference), dtype=np.float64)
    num_latent = len(viewmats)
    for i, d_revisit in enumerate(revisit_indices):
        lat_revisit = min(_decoded_to_latent(d_revisit), num_latent - 1)
        for j, d_ref in enumerate(reference_indices):
            lat_ref = min(_decoded_to_latent(d_ref), num_latent - 1)
            matrix[i, j] = _pose_distance(viewmats[lat_revisit], viewmats[lat_ref], np)
    return matrix


def _match_diagnostics(
    matched_positions: "np.ndarray",
    sample_positions: list[int],
    reference_full_length: int,
    np,
) -> tuple[float, float, float]:
    """Return (unique_ratio, temporal_mae_normalized, reverse_violation_ratio)."""
    expected_positions = np.asarray(
        [reference_full_length - 1 - position for position in sample_positions],
        dtype=np.int64,
    )
    candidate_positions = np.asarray(sample_positions, dtype=np.int64)
    matched_full_positions = candidate_positions[matched_positions]
    normalizer = max(1, reference_full_length - 1)
    unique_ratio = float(len(np.unique(matched_positions)) / len(matched_positions))
    temporal_mae = float(np.mean(np.abs(matched_full_positions - expected_positions)) / normalizer)
    reverse_violations = (
        float(np.mean(np.diff(matched_full_positions) > 0))
        if len(matched_full_positions) > 1
        else 0.0
    )
    return unique_ratio, temporal_mae, reverse_violations


def _evaluate_video(
    *,
    video_path: Path,
    closure_pair: list[int],
    trajectory: str,
    wan_root: Path,
    resize_width: int,
    max_frames_per_segment: int,
    lpips_state,
    lpips_batch_size: int,
    cv2,
    np,
    structural_similarity,
) -> dict[str, Any]:
    if len(closure_pair) != 2:
        raise ValueError(f"Expected two closure frame indices, got {closure_pair!r}")
    closure_start, closure_end = map(int, closure_pair)
    path_length = closure_end - closure_start
    if path_length < 4 or path_length % 2:
        raise ValueError(
            f"Closure interval must have a positive even length, got {closure_pair!r}"
        )

    turnaround = closure_start + path_length // 2
    reference_full = list(range(closure_start, turnaround))
    revisit_full = list(range(turnaround + 1, closure_end + 1))
    if len(reference_full) != len(revisit_full):
        raise AssertionError("Internal loop segments do not have equal lengths")

    sample_positions = _uniform_indices(len(reference_full), max_frames_per_segment)
    reference_indices = [reference_full[position] for position in sample_positions]
    revisit_indices = [revisit_full[position] for position in sample_positions]
    selected = sorted(set(reference_indices + revisit_indices + [closure_start, closure_end]))
    frames, reported_frame_count = _read_selected_video_frames(
        video_path, selected, resize_width, cv2, np
    )

    reference = _stack(frames, reference_indices, np)
    revisit = _stack(frames, revisit_indices, np)
    closure_reference = _stack(frames, [closure_start], np)
    closure_revisit = _stack(frames, [closure_end], np)

    closure_psnr, closure_ssim = _pixel_metrics(
        closure_reference, closure_revisit, structural_similarity, np
    )

    result: dict[str, Any] = {
        "reported_video_frames": reported_frame_count,
        "closure_start_frame": closure_start,
        "turnaround_frame": turnaround,
        "closure_end_frame": closure_end,
        "full_segment_frames": len(reference_full),
        "metric_segment_frames": len(sample_positions),
        "closure_psnr": closure_psnr,
        "closure_ssim": closure_ssim,
        "closure_lpips": None,
        "lpips_psnr": None,
        "lpips_ssim": None,
        "lpips_lpips": None,
        "pose_psnr": None,
        "pose_ssim": None,
        "pose_lpips": None,
        "lpips_match_unique_ratio": None,
        "lpips_match_temporal_mae_normalized": None,
        "lpips_match_reverse_violation_ratio": None,
        "pose_match_unique_ratio": None,
        "pose_match_temporal_mae_normalized": None,
        "pose_match_reverse_violation_ratio": None,
        "lpips_match_reference_indices": [],
        "pose_match_reference_indices": [],
    }

    # --- Pose matching (always available, does not need LPIPS) ---
    viewmats = _parse_trajectory(trajectory, wan_root, np)
    pose_matrix = _pose_distance_matrix(revisit_indices, reference_indices, viewmats, np)
    pose_matched_positions = np.argmin(pose_matrix, axis=1)
    pose_matched_reference = reference[pose_matched_positions]
    pose_psnr, pose_ssim = _pixel_metrics(
        pose_matched_reference, revisit, structural_similarity, np
    )
    p_unique, p_temporal, p_reverse = _match_diagnostics(
        pose_matched_positions, sample_positions, len(reference_full), np
    )
    result.update(
        {
            "pose_psnr": pose_psnr,
            "pose_ssim": pose_ssim,
            "pose_match_unique_ratio": p_unique,
            "pose_match_temporal_mae_normalized": p_temporal,
            "pose_match_reverse_violation_ratio": p_reverse,
            "pose_match_reference_indices": [
                int(reference_indices[pos]) for pos in pose_matched_positions.tolist()
            ],
        }
    )

    # --- LPIPS (MAG) matching + closure LPIPS ---
    if lpips_state is None:
        return result

    model, device, torch = lpips_state
    closure_distances = _lpips_pairs(
        closure_revisit, closure_reference, model, device, lpips_batch_size, torch, np
    )
    result["closure_lpips"] = float(closure_distances.mean())

    distance_matrix = _lpips_matrix(
        revisit, reference, model, device, lpips_batch_size, torch, np
    )
    lpips_matched_positions = np.argmin(distance_matrix, axis=1)
    lpips_matched_reference = reference[lpips_matched_positions]
    lpips_psnr, lpips_ssim = _pixel_metrics(
        lpips_matched_reference, revisit, structural_similarity, np
    )
    l_unique, l_temporal, l_reverse = _match_diagnostics(
        lpips_matched_positions, sample_positions, len(reference_full), np
    )

    # Pose-matched LPIPS: use pose matching but measure LPIPS on those pairs.
    pose_lpips_values = distance_matrix[np.arange(len(pose_matched_positions)), pose_matched_positions]

    result.update(
        {
            "lpips_psnr": lpips_psnr,
            "lpips_ssim": lpips_ssim,
            "lpips_lpips": float(distance_matrix.min(axis=1).mean()),
            "pose_lpips": float(pose_lpips_values.mean()),
            "lpips_match_unique_ratio": l_unique,
            "lpips_match_temporal_mae_normalized": l_temporal,
            "lpips_match_reverse_violation_ratio": l_reverse,
            "lpips_match_reference_indices": [
                int(reference_indices[pos]) for pos in lpips_matched_positions.tolist()
            ],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MAG-style and pose-driven visual loop closure for minWM-dyKV videos."
    )
    parser.add_argument("--generation-manifest", type=Path, required=True,
                        help="generation_manifest.jsonl produced by the dyKV runner.")
    parser.add_argument("--closure-manifest", type=Path, required=True,
                        help="demos_loop_closure/manifest.json with closure_pair_decoded.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, required=True,
                        help="Wan21 root for importing wan_utils.camera_trajectory.")
    parser.add_argument("--duration-label", default=None,
                        help="Manifest duration key such as 10s; inferred from latent_frames by default.")
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--max-frames-per-segment", type=int, default=96,
                        help="Uniformly subsample each path segment to this many frames; 0 keeps every frame.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--lpips-batch-size", type=int, default=64)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--require-lpips", action="store_true")
    args = parser.parse_args()

    if args.skip_lpips and args.require_lpips:
        parser.error("--skip-lpips and --require-lpips are mutually exclusive")
    if args.lpips_batch_size <= 0:
        parser.error("--lpips-batch-size must be positive")
    if args.max_frames_per_segment < 0:
        parser.error("--max-frames-per-segment cannot be negative")

    try:
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise SystemExit(
            "Loop-closure evaluation requires numpy, opencv-python, and scikit-image. "
            f"Original import error: {exc}"
        ) from exc

    closure_manifest = json.loads(args.closure_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lpips_state = None
    lpips_error = None
    if not args.skip_lpips:
        try:
            lpips_state = _init_lpips(args.device, args.lpips_net)
        except Exception as exc:
            lpips_error = repr(exc)
            if args.require_lpips:
                raise

    rows = _load_generation_manifest(args.generation_manifest)
    metric_rows: list[dict[str, Any]] = []
    errors = []
    for row in rows:
        video_path = Path(row.get("output_path", ""))
        if not video_path.is_file():
            errors.append({"output_path": str(video_path), "error": "missing video"})
            continue
        try:
            prompt_index = int(row.get("prompt_index", ""))
            num_output_frames = int(row.get("num_output_frames", 0)) or int(
                closure_manifest["durations"].get(
                    args.duration_label or "", {}
                ).get("latent_frames", 40)
            )
            duration_label = _duration_label(
                closure_manifest, num_output_frames, args.duration_label
            )
            sample = _manifest_sample(closure_manifest, duration_label, prompt_index)
            metrics = _evaluate_video(
                video_path=video_path,
                closure_pair=sample["closure_pair_decoded"],
                trajectory=row.get("trajectory", sample.get("trajectory", "")),
                wan_root=args.wan_root,
                resize_width=max(0, int(args.resize_width)),
                max_frames_per_segment=int(args.max_frames_per_segment),
                lpips_state=lpips_state,
                lpips_batch_size=int(args.lpips_batch_size),
                cv2=cv2,
                np=np,
                structural_similarity=structural_similarity,
            )
        except Exception as exc:
            errors.append({"output_path": str(video_path), "error": repr(exc)})
            continue

        metric_rows.append(
            {
                "prompt_index": prompt_index,
                "duration_label": duration_label,
                "prompt": row.get("prompt", ""),
                "trajectory": row.get("trajectory", ""),
                "output_path": str(video_path),
                **metrics,
            }
        )

    fieldnames = [
        "prompt_index",
        "duration_label",
        "prompt",
        "trajectory",
        "output_path",
        "reported_video_frames",
        "closure_start_frame",
        "turnaround_frame",
        "closure_end_frame",
        "full_segment_frames",
        "metric_segment_frames",
        *METRIC_FIELDS,
        "lpips_match_reference_indices",
        "pose_match_reference_indices",
    ]
    csv_path = args.output_dir / "loop_closure_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            output_row = dict(row)
            for key in ("lpips_match_reference_indices", "pose_match_reference_indices"):
                output_row[key] = json.dumps(row.get(key, []), separators=(",", ":"))
            writer.writerow(output_row)

    summary_fields = [*METRIC_FIELDS]
    summary = {
        "generation_manifest": str(args.generation_manifest),
        "closure_manifest": str(args.closure_manifest),
        "num_evaluated": len(metric_rows),
        "num_errors": len(errors),
        "lpips_available": lpips_state is not None,
        "lpips_device": lpips_state[1] if lpips_state is not None else None,
        "lpips_error": lpips_error,
        "metrics": {
            field: _mean([row.get(field) for row in metric_rows]) for field in summary_fields
        },
        "protocol": {
            "split": (
                "The manifest closure interval [closure_start, closure_end] is split at its "
                "midpoint (turnaround). Outbound frames [start, turnaround) are the internal "
                "reference; revisit frames (turnaround, closure_end] are the evaluated segment."
            ),
            "lpips_match": (
                "MAG-style: each revisit frame independently selects its LPIPS-nearest outbound "
                "frame; PSNR/SSIM/LPIPS use those same pairs. Tolerates speed differences but can "
                "match to temporally wrong frames."
            ),
            "pose_match": (
                "Camera-pose-driven: each revisit frame is matched to the outbound frame with the "
                "smallest SE(3) distance (translation + rotation degrees). For exact string-loop "
                "trajectories this is the deterministic temporal mirror. Tests whether the model "
                "generates visually consistent content at the same camera pose."
            ),
            "diagnostics": (
                "Higher match_unique_ratio is better; lower match_temporal_mae_normalized and "
                "match_reverse_violation_ratio are better. pose_match_temporal_mae_normalized "
                "should be near 0 for exact loop trajectories."
            ),
            "subsampling": (
                f"At most {args.max_frames_per_segment} uniformly spaced frames per segment "
                "(0 means all frames)."
            ),
            "limitation": (
                "The outbound segment is a pseudo-reference generated by the same rollout, not an "
                "external ground-truth video. Metrics measure visual memory/closure consistency."
            ),
        },
        "errors": errors,
        "videos": metric_rows,
    }
    json_path = args.output_dir / "loop_closure_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    if lpips_state is None and not args.skip_lpips:
        print(
            "LPIPS was unavailable; pose matching and endpoint PSNR/SSIM were retained, "
            "but MAG-style LPIPS matching was skipped."
        )


if __name__ == "__main__":
    main()
