"""
Demo 11 – Bring Your Own Video
================================
Feed any video file into V-JEPA 2 and get:
  1. Progressive prediction chart (25%, 50%, 75%, 100%)
  2. Confidence evolution line chart
  3. Animated GIF with live prediction bars

Usage:
    python demos/11_your_own_video.py path/to/video.mp4
    python demos/11_your_own_video.py path/to/video.mp4 --name my_demo
"""

import argparse
import gc
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs")
DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"

FPS_GIF = 2
N_STEPS = 16
N_HOLD = 6
TOP_K = 5

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
})


def load_video(path, max_frames=128):
    """Load video frames from a local file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {path}")
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    print(f"  Loaded {len(frames)} frames ({fps_src:.0f} fps source)")
    return frames


def load_model():
    from transformers import AutoVideoProcessor, AutoModelForVideoClassification

    print("  Loading V-JEPA 2...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
    model.eval().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params, {len(model.config.id2label)} action classes")
    return model, processor


def classify_at_fraction(model, processor, frames, fraction):
    """Classify using the first `fraction` of the video."""
    n_visible = max(1, int(len(frames) * fraction))
    visible = frames[:n_visible]
    n_input = 16

    if n_visible >= n_input:
        indices = np.linspace(0, n_visible - 1, n_input, dtype=int)
        sampled = [visible[i] for i in indices]
    else:
        indices = np.linspace(0, n_visible - 1, n_visible, dtype=int)
        sampled = [visible[i] for i in indices]
        sampled += [sampled[-1]] * (n_input - len(sampled))

    inputs = processor(sampled, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_idx = probs.argsort(descending=True)[:TOP_K]
    preds = [(model.config.id2label[i.item()], probs[i].item()) for i in top_idx]
    return preds, probs.cpu().numpy(), visible[-1]


# ---------- Progressive chart ----------

def plot_progressive(all_results, frames, fixed_labels, save_path):
    """Stacked: frame strip + bar chart per fraction."""
    fractions = sorted(all_results.keys())
    n_rows = len(fractions)

    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 3.5 * n_rows),
                              gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.35})
    fig.patch.set_facecolor("#FAFAFA")

    # Build label->index mapping
    label_to_idx = {name: idx for idx, name in enumerate(fixed_labels)}

    for row, frac in enumerate(fractions):
        preds, probs_full, last_frame = all_results[frac]
        pct = int(frac * 100)
        n_vis = max(1, int(len(frames) * frac))

        # Left: frame strip
        ax_f = axes[row, 0]
        visible = frames[:n_vis]
        n_thumbs = 4
        strip_idx = np.linspace(0, len(visible) - 1, n_thumbs, dtype=int)
        strip_h = visible[0].shape[0]
        strip_w = visible[0].shape[1]
        gap = 6
        canvas_w = n_thumbs * strip_w + (n_thumbs - 1) * gap
        canvas = np.ones((strip_h, canvas_w, 3), dtype=np.uint8) * 245
        for i, si in enumerate(strip_idx):
            x0 = i * (strip_w + gap)
            canvas[:, x0:x0 + strip_w] = visible[si]

        ax_f.imshow(canvas)
        ax_f.set_xticks([])
        ax_f.set_yticks([])
        color = plt.cm.RdYlGn(frac)
        for spine in ax_f.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)
        ax_f.set_title(f"{pct}% seen  ({n_vis}/{len(frames)} frames)",
                        fontsize=14, fontweight="bold", pad=8)

        # Right: bars with fixed labels
        ax_b = axes[row, 1]
        bar_probs = [probs_full[label_to_idx.get(l, 0)] if label_to_idx else 0
                     for l in fixed_labels]
        # Actually look up by scanning all class probs
        bar_probs = []
        for l in fixed_labels:
            found = False
            for pl, pp in preds:
                if pl == l:
                    bar_probs.append(pp)
                    found = True
                    break
            if not found:
                bar_probs.append(0.0)

        top_label = fixed_labels[np.argmax(bar_probs)]
        bar_colors = ["#2196F3" if l == top_label else "#B0BEC5" for l in fixed_labels]

        bars = ax_b.barh(range(len(fixed_labels)), bar_probs, color=bar_colors,
                         edgecolor="white", height=0.6)
        ax_b.set_yticks(range(len(fixed_labels)))
        ax_b.set_yticklabels(fixed_labels, fontsize=11)
        ax_b.invert_yaxis()
        ax_b.set_xlim(0, 1.08)
        ax_b.grid(axis="x", alpha=0.15)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)

        for bar, prob in zip(bars, bar_probs):
            if prob > 0.005:
                ax_b.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                          f"{prob:.1%}", va="center", fontsize=11, fontweight="bold")

    plt.suptitle("V-JEPA 2 — Progressive Action Recognition (your video)",
                 fontsize=18, fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.02, right=0.98, hspace=0.45)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


# ---------- Confidence line chart ----------

def plot_confidence(all_results, fixed_labels, save_path):
    fractions = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(fixed_labels)))

    for i, label in enumerate(fixed_labels):
        probs = []
        for frac in fractions:
            preds = all_results[frac][0]
            prob = 0
            for l, p in preds:
                if l == label:
                    prob = p
                    break
            probs.append(prob)

        ax.plot([f * 100 for f in fractions], probs,
                "o-", color=colors[i], linewidth=2, markersize=8,
                label=label[:45])

    ax.set_xlabel("Video Seen (%)", fontsize=13)
    ax.set_ylabel("Confidence", fontsize=13)
    ax.set_title("Confidence Evolution", fontsize=16)
    ax.set_xlim(15, 105)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------- Animated GIF ----------

def create_gif(model, processor, frames, fixed_labels, save_path):
    """Animated GIF: video on top, prediction bars below."""
    print("  Creating animated GIF...")
    fractions = np.linspace(0.1, 1.0, N_STEPS)

    # Build label->class index mapping
    label_to_cls = {}
    for cls_idx, name in model.config.id2label.items():
        label_to_cls[name] = cls_idx

    # Pre-compute
    all_preds = []
    all_probs = []
    all_frames_display = []
    for frac in fractions:
        preds, probs, frame = classify_at_fraction(model, processor, frames, frac)
        all_preds.append(preds)
        all_probs.append(probs)
        all_frames_display.append(frame)

    total_frames = N_STEPS + N_HOLD

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1A1A2E")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.4)
    fig.subplots_adjust(left=0.30, right=0.95, top=0.93, bottom=0.08)
    ax_frame = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[1, 0])

    progress_bg = plt.Rectangle(
        (0.1, 0.02), 0.8, 0.02,
        transform=fig.transFigure, facecolor="#333355", edgecolor="none", zorder=10)
    progress_fill = plt.Rectangle(
        (0.1, 0.02), 0.0, 0.02,
        transform=fig.transFigure, facecolor="#4FC3F7", edgecolor="none", zorder=11)
    fig.patches.extend([progress_bg, progress_fill])

    def update(i):
        idx = min(i, N_STEPS - 1)
        frac = fractions[idx]
        probs_full = all_probs[idx]
        frame = all_frames_display[idx]
        pct = int(frac * 100)

        ax_frame.clear()
        ax_frame.imshow(frame)
        ax_frame.set_title(f"{pct}% seen", fontsize=14,
                            fontweight="bold", color="white")
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        for spine in ax_frame.spines.values():
            spine.set_color("#4FC3F7")
            spine.set_linewidth(2)

        ax_bars.clear()
        display_labels = [l[:40] for l in fixed_labels]
        probs = [probs_full[label_to_cls[l]] for l in fixed_labels]

        top_label = fixed_labels[np.argmax(probs)]
        bar_colors = ["#4FC3F7" if l == top_label else "#555577" for l in fixed_labels]
        bars = ax_bars.barh(range(len(display_labels)), probs, color=bar_colors,
                            edgecolor="none", height=0.6)
        ax_bars.set_yticks(range(len(display_labels)))
        ax_bars.set_yticklabels(display_labels, fontsize=10, color="white")
        ax_bars.invert_yaxis()
        ax_bars.set_xlim(0, 1.05)
        ax_bars.set_facecolor("#1A1A2E")
        ax_bars.tick_params(colors="white")
        ax_bars.set_title("V-JEPA 2 predictions", fontsize=14,
                           fontweight="bold", color="white")

        for bar, prob in zip(bars, probs):
            ax_bars.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{prob:.0%}", va="center", fontsize=11, color="white",
                         fontweight="bold")

        ax_bars.grid(axis="x", alpha=0.1, color="white")
        ax_bars.spines["top"].set_visible(False)
        ax_bars.spines["right"].set_visible(False)
        ax_bars.spines["bottom"].set_color("#555577")
        ax_bars.spines["left"].set_color("#555577")

        progress_fill.set_width(0.8 * frac)
        return bars

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // FPS_GIF, blit=False)
    anim.save(str(save_path), writer="pillow", fps=FPS_GIF, dpi=100)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Run V-JEPA 2 on your own video")
    parser.add_argument("video", type=str, help="Path to video file (.mp4, .webm, ...)")
    parser.add_argument("--name", type=str, default=None,
                        help="Output name prefix (default: video filename)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: {video_path} not found")
        sys.exit(1)

    name = args.name or video_path.stem
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"  V-JEPA 2 — Your Video: {video_path.name}")
    print("=" * 60)

    print("\n[1/5] Loading video...")
    frames = load_video(video_path)

    print("\n[2/5] Loading model...")
    model, processor = load_model()

    print("\n[3/5] Progressive classification...")
    fractions = (0.25, 0.50, 0.75, 1.0)
    all_results = {}
    for frac in fractions:
        preds, probs, frame = classify_at_fraction(model, processor, frames, frac)
        all_results[frac] = (preds, probs, frame)
        pct = int(frac * 100)
        print(f"  {pct:3d}% → {preds[0][0][:60]} ({preds[0][1]:.1%})")

    # Fixed labels from final frame
    fixed_labels = [p[0] for p in all_results[1.0][0]]

    print("\n[4/5] Generating charts...")
    plot_progressive(all_results, frames, fixed_labels,
                     OUTPUT_DIR / f"11_progressive_{name}.png")
    plot_confidence(all_results, fixed_labels,
                    OUTPUT_DIR / f"11_confidence_{name}.png")

    print("\n[5/5] Creating animated GIF...")
    create_gif(model, processor, frames, fixed_labels,
               OUTPUT_DIR / f"11_prediction_{name}.gif")

    del model
    gc.collect()

    print(f"\n✓ Done! Check outputs/11_*_{name}.*")
    print(f"  Progressive chart: outputs/11_progressive_{name}.png")
    print(f"  Confidence chart:  outputs/11_confidence_{name}.png")
    print(f"  Animated GIF:      outputs/11_prediction_{name}.gif")


if __name__ == "__main__":
    main()