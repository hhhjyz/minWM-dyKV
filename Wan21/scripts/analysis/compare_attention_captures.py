#!/usr/bin/env python3
"""Compare paired dyKV attention captures and write reproducible plots/metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CASES = ("retrieval_no_compression", "motion_novelty_backfill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs=2, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--focus-layer", type=int, default=15)
    return parser.parse_args()


def load_case(root: Path, case: str) -> tuple[dict, dict]:
    capture_path = root / case / "attention_capture_00000.json"
    payload = json.loads(capture_path.read_text(encoding="utf-8"))
    records = {}
    for row in payload["records"]:
        attention = np.asarray(row["frame_attention"], dtype=np.float64)
        sink_frames, retrieval_frames, _ = row["region_sizes"]
        retrieval_end = sink_frames + retrieval_frames
        regions = np.stack(
            [
                attention[:, :, :sink_frames].sum(axis=2),
                attention[:, :, sink_frames:retrieval_end].sum(axis=2),
                attention[:, :, retrieval_end:].sum(axis=2),
            ],
            axis=2,
        )
        windows = attention[:, :, sink_frames:retrieval_end]
        window_mass = windows.mean(axis=(0, 1))
        distribution = window_mass / window_mass.sum()
        entropy = float(
            -(distribution * np.log(distribution + 1e-30)).sum()
            / math.log(len(distribution))
        )
        records[(int(row["current_frame"]), int(row["layer_idx"]))] = {
            "region_mass": regions.mean(axis=(0, 1)),
            "retrieval_windows": window_mass,
            "retrieval_entropy": entropy,
            "retrieval_cv": float(window_mass.std() / window_mass.mean()),
        }
    return payload, records


def validate_pairing(root: Path, cases: tuple[str, str]) -> dict:
    manifests = {}
    for case in cases:
        path = root / case / "generation_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            raise ValueError(f"generation manifest is empty: {path}")
        manifests[case] = rows[-1]
    fields = ("prompt_index", "prompt", "trajectory", "sample_seed", "initial_noise_fingerprint")
    for field in fields:
        values = [manifests[case].get(field) for case in cases]
        if values[0] != values[1]:
            raise ValueError(f"paired manifests differ in {field}: {values}")
    return {field: manifests[cases[0]].get(field) for field in fields}


def write_metrics(
    output_dir: Path,
    cases: tuple[str, str],
    records: dict[str, dict],
    frames: list[int],
    layers: list[int],
    pairing: dict,
) -> None:
    phases = {
        "outward": [frame for frame in frames if frame <= 48],
        "turnaround": [frame for frame in frames if frame == 52],
        "return": [frame for frame in frames if frame >= 56],
        "final": [frame for frame in frames if frame >= 96],
    }
    result = {
        "cases": list(cases),
        "pairing": pairing,
        "frames": frames,
        "layers": layers,
        "region_order": ["sink", "retrieval", "local"],
        "phase_region_mass": {},
        "layer_region_mass": {},
        "retrieval_window_statistics": {},
    }
    for phase, phase_frames in phases.items():
        result["phase_region_mass"][phase] = {}
        for case in cases:
            values = np.asarray(
                [
                    records[case][(frame, layer)]["region_mass"]
                    for frame in phase_frames
                    for layer in layers
                ]
            ).mean(axis=0)
            result["phase_region_mass"][phase][case] = values.tolist()

    for layer in layers:
        result["layer_region_mass"][str(layer)] = {}
        result["retrieval_window_statistics"][str(layer)] = {}
        for case in cases:
            masses = np.asarray(
                [records[case][(frame, layer)]["region_mass"] for frame in frames]
            ).mean(axis=0)
            entropy = np.mean(
                [records[case][(frame, layer)]["retrieval_entropy"] for frame in frames]
            )
            cv = np.mean(
                [records[case][(frame, layer)]["retrieval_cv"] for frame in frames]
            )
            result["layer_region_mass"][str(layer)][case] = masses.tolist()
            result["retrieval_window_statistics"][str(layer)][case] = {
                "normalized_entropy": float(entropy),
                "coefficient_of_variation": float(cv),
            }

    (output_dir / "attention_comparison_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def plot_region_curves(
    output_dir: Path,
    cases: tuple[str, str],
    records: dict[str, dict],
    frames: list[int],
    layers: list[int],
) -> None:
    focus_layers = [layer for layer in (10, 15, 20, 25) if layer in layers]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for axis, layer in zip(axes.flat, focus_layers):
        for case, color in zip(cases, ("#4477AA", "#CC6677")):
            values = [records[case][(frame, layer)]["region_mass"][1] for frame in frames]
            axis.plot(frames, values, marker="o", markersize=3, label=case, color=color)
        axis.axvline(52, color="black", linestyle="--", linewidth=1, alpha=0.5)
        axis.set_title(f"Layer {layer}")
        axis.set_ylabel("Retrieval attention mass")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Current end frame")
    axes[-1, 1].set_xlabel("Current end frame")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("MBench paired retrieval attention over time (turnaround at frame 52)")
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_attention_over_time.png", dpi=180)
    plt.close(fig)


def plot_region_delta(
    output_dir: Path,
    cases: tuple[str, str],
    records: dict[str, dict],
    frames: list[int],
    layers: list[int],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    names = ("sink", "retrieval", "local")
    matrices = []
    for region_index in range(3):
        matrix = np.asarray(
            [
                [
                    records[cases[1]][(frame, layer)]["region_mass"][region_index]
                    - records[cases[0]][(frame, layer)]["region_mass"][region_index]
                    for frame in frames
                ]
                for layer in layers
            ]
        )
        matrices.append(matrix)
    limit = max(float(np.abs(matrix).max()) for matrix in matrices)
    for axis, name, matrix in zip(axes, names, matrices):
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_yticks(range(len(layers)), [f"L{layer}" for layer in layers])
        axis.set_title(f"Motion backfill - no compression: {name}")
    axes[-1].set_xticks(range(len(frames)), frames, rotation=45)
    axes[-1].set_xlabel("Current end frame")
    fig.colorbar(image, ax=axes, label="Attention-mass difference", shrink=0.85)
    fig.savefig(output_dir / "region_attention_delta_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_windows(
    output_dir: Path,
    cases: tuple[str, str],
    records: dict[str, dict],
    frames: list[int],
    focus_layer: int,
) -> None:
    matrices = {
        case: np.asarray(
            [records[case][(frame, focus_layer)]["retrieval_windows"] for frame in frames]
        )
        for case in cases
    }
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for axis, case in zip(axes, cases):
        image = axis.imshow(matrices[case], aspect="auto", cmap="YlOrRd", vmin=0, vmax=maximum)
        axis.set_xticks(range(8), [f"W{i}" for i in range(8)])
        axis.set_yticks(range(len(frames)), frames)
        axis.set_xlabel("Contiguous retrieval window (F tokens)")
        axis.set_title(case)
    axes[0].set_ylabel("Current end frame")
    fig.suptitle(
        f"Layer {focus_layer} retrieval-window attention; motion windows are not source frames or virtual slots"
    )
    fig.colorbar(image, ax=axes, label="Attention mass", shrink=0.8)
    fig.savefig(output_dir / "retrieval_window_attention_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cases = tuple(args.cases)
    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    pairing = validate_pairing(args.root, cases)
    for case in cases:
        _, records[case] = load_case(args.root, case)
    key_sets = [set(records[case]) for case in cases]
    if key_sets[0] != key_sets[1]:
        raise ValueError("paired captures do not contain the same frame/layer keys")
    frames = sorted({frame for frame, _ in key_sets[0]})
    layers = sorted({layer for _, layer in key_sets[0]})
    if args.focus_layer not in layers:
        raise ValueError(f"focus layer {args.focus_layer} was not captured")

    write_metrics(output_dir, cases, records, frames, layers, pairing)
    plot_region_curves(output_dir, cases, records, frames, layers)
    plot_region_delta(output_dir, cases, records, frames, layers)
    plot_retrieval_windows(output_dir, cases, records, frames, args.focus_layer)
    print(f"Wrote paired attention analysis to {output_dir}")


if __name__ == "__main__":
    main()
