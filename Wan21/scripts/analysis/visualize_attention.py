"""Visualize attention capture data from two dyKV cases."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = "output/attn_capture"
CASES = ["retrieval_no_compression_honest", "motion_novelty_backfill_honest"]
CASE_LABELS = ["retrieval_no_compression", "motion_novelty_backfill"]
LAYER_INDICES = [0, 5, 10, 15, 20, 25, 29]
REGION_NAMES = ["sink", "retrieval", "local"]
REGION_COLORS = ["#4477AA", "#EE6677", "#228833"]


def load_case(case_name):
    path = os.path.join(ROOT, case_name, "attention_capture_00000.json")
    with open(path) as f:
        data = json.load(f)
    records = {}
    for r in data["records"]:
        key = (r["layer_idx"], r["current_frame"])
        records[key] = {
            "frame_attention": np.array(r["frame_attention"]),
            "region_sizes": r["region_sizes"],
            "call_count": r["call_count"],
        }
    return data, records


def region_mass(attn, region_sizes):
    """Sum attention mass per region. attn: [H, Qf, Kf] -> [H, Qf, 3]."""
    rs = region_sizes
    sink = attn[:, :, :rs[0]].sum(axis=2)
    retr = attn[:, :, rs[0]:rs[0]+rs[1]].sum(axis=2)
    local = attn[:, :, rs[0]+rs[1]:].sum(axis=2)
    return np.stack([sink, retr, local], axis=2)


def main():
    all_data = {}
    for case in CASES:
        data, records = load_case(case)
        all_data[case] = records

    current_frames = [24, 28, 32, 36, 40]
    focus_frame = 36
    focus_layer = 15

    # ---- Figure 1: Frame-level attention heatmaps for focus_frame ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        f"Frame-level attention (current_frame={focus_frame}, avg over heads & diffusion steps)\n"
        f"trajectory: k*19,i*19,k*1  |  KV layout: [sink(4) | retrieval(8) | local(8)]",
        fontsize=13,
    )

    for row, (case, label) in enumerate(zip(CASES, CASE_LABELS)):
        records = all_data[case]
        for col, layer in enumerate([0, 10, 15, 29]):
            ax = axes[row, col]
            key = (layer, focus_frame)
            if key not in records:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_title(f"layer {layer}")
                continue
            attn = records[key]["frame_attention"]
            avg = attn.mean(axis=0)
            im = ax.imshow(avg, aspect="auto", cmap="YlOrRd", vmin=0, vmax=avg.max())
            ax.set_xlabel("KV frame index")
            ax.set_ylabel("Query frame")
            rs = records[key]["region_sizes"]
            for x in [rs[0], rs[0]+rs[1]]:
                ax.axvline(x - 0.5, color="white", linewidth=1.5, linestyle="--")
            ax.set_title(f"{label}\nlayer {layer}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "attention_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved attention_heatmaps.png")

    # ---- Figure 2: Region-level attention bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Region-level attention mass (current_frame={focus_frame}, avg over heads & query frames)\n"
        f"trajectory: k*19,i*19,k*1",
        fontsize=13,
    )

    for ax_idx, (case, label) in enumerate(zip(CASES, CASE_LABELS)):
        ax = axes[ax_idx]
        records = all_data[case]
        x = np.arange(len(LAYER_INDICES))
        width = 0.25
        for ri, (rname, rcolor) in enumerate(zip(REGION_NAMES, REGION_COLORS)):
            vals = []
            for layer in LAYER_INDICES:
                key = (layer, focus_frame)
                if key in records:
                    rm = region_mass(records[key]["frame_attention"], records[key]["region_sizes"])
                    vals.append(rm.mean(axis=(0, 1))[ri])
                else:
                    vals.append(0)
            ax.bar(x + ri * width, vals, width, label=rname, color=rcolor)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"L{l}" for l in LAYER_INDICES])
        ax.set_ylabel("Attention mass")
        ax.set_title(label)
        ax.legend()
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "region_attention_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved region_attention_bars.png")

    # ---- Figure 3: Retrieval region detail (frame-level within retrieval) ----
    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    fig.suptitle(
        "Attention to individual retrieval frames (avg over heads & query frames)\n"
        "KV frames 4-11 are the retrieval region",
        fontsize=13,
    )

    for row, (case, label) in enumerate(zip(CASES, CASE_LABELS)):
        records = all_data[case]
        for col, cf in enumerate(current_frames):
            ax = axes[row, col]
            for layer in [0, 10, 15, 29]:
                key = (layer, cf)
                if key not in records:
                    continue
                attn = records[key]["frame_attention"]
                rs = records[key]["region_sizes"]
                retr_attn = attn[:, :, rs[0]:rs[0]+rs[1]].mean(axis=(0, 1))
                ax.plot(range(rs[1]), retr_attn, marker="o", label=f"layer {layer}", alpha=0.8)
            ax.set_xlabel("Retrieval frame index (0-7)")
            ax.set_ylabel("Attention mass")
            ax.set_title(f"{label}\ncurrent_frame={cf}")
            ax.legend(fontsize=8)
            ax.set_xticks(range(8))

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "retrieval_frame_detail.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved retrieval_frame_detail.png")

    # ---- Figure 4: Layer-wise comparison at focus_frame ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Layer-wise attention to each KV frame (current_frame={focus_frame}, avg over heads & query frames)",
        fontsize=13,
    )

    for ax_idx, (case, label) in enumerate(zip(CASES, CASE_LABELS)):
        ax = axes[ax_idx]
        records = all_data[case]
        for layer in LAYER_INDICES:
            key = (layer, focus_frame)
            if key not in records:
                continue
            attn = records[key]["frame_attention"]
            avg = attn.mean(axis=(0, 1))
            ax.plot(range(len(avg)), avg, marker="o", markersize=3, label=f"layer {layer}", alpha=0.8)
        ax.set_xlabel("KV frame index (0-19)")
        ax.set_ylabel("Attention mass")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xticks(range(20))
        rs = list(all_data[case][(focus_layer, focus_frame)]["region_sizes"])
        for x in [rs[0], rs[0]+rs[1]]:
            ax.axvline(x - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax.text(rs[0]/2 - 0.5, ax.get_ylim()[1]*0.9, "sink", ha="center", fontsize=10, color="blue")
        ax.text(rs[0]+rs[1]/2 - 0.5, ax.get_ylim()[1]*0.9, "retrieval", ha="center", fontsize=10, color="red")
        ax.text(rs[0]+rs[1]+rs[2]/2 - 0.5, ax.get_ylim()[1]*0.9, "local", ha="center", fontsize=10, color="green")

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "layerwise_attention.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved layerwise_attention.png")

    # ---- Print text summary ----
    print("\n" + "=" * 80)
    print("ATTENTION DISTRIBUTION SUMMARY")
    print("=" * 80)
    print(f"\nFocus: current_frame={focus_frame}, layer={focus_layer}")
    print(f"KV layout: [sink(4) | retrieval(8) | local(8)] = 20 frames\n")

    for case, label in zip(CASES, CASE_LABELS):
        records = all_data[case]
        key = (focus_layer, focus_frame)
        if key not in records:
            continue
        attn = records[key]["frame_attention"]
        rs = records[key]["region_sizes"]
        rm = region_mass(attn, rs)
        region_avg = rm.mean(axis=(0, 1))
        print(f"  {label}:")
        print(f"    sink:      {region_avg[0]:.4f} ({region_avg[0]*100:.1f}%)")
        print(f"    retrieval: {region_avg[1]:.4f} ({region_avg[1]*100:.1f}%)")
        print(f"    local:     {region_avg[2]:.4f} ({region_avg[2]*100:.1f}%)")
        print(f"    retrieval per-frame: {[f'{v:.4f}' for v in attn[:, :, rs[0]:rs[0]+rs[1]].mean(axis=(0, 1))]}")
        print()

    print("\nRegion mass across all layers (current_frame=36):")
    print(f"{'Layer':>6} | {'no_comp sink':>12} {'retr':>8} {'local':>8} | {'backfill sink':>14} {'retr':>8} {'local':>8} | {'Δ retr':>8}")
    print("-" * 90)
    for layer in LAYER_INDICES:
        vals = []
        for case in CASES:
            key = (layer, focus_frame)
            if key in all_data[case]:
                rm = region_mass(all_data[case][key]["frame_attention"], all_data[case][key]["region_sizes"])
                vals.append(rm.mean(axis=(0, 1)))
            else:
                vals.append([0, 0, 0])
        delta = vals[1][1] - vals[0][1]
        print(f"{layer:>6} | {vals[0][0]:>12.4f} {vals[0][1]:>8.4f} {vals[0][2]:>8.4f} | "
              f"{vals[1][0]:>14.4f} {vals[1][1]:>8.4f} {vals[1][2]:>8.4f} | {delta:>+8.4f}")


if __name__ == "__main__":
    main()
