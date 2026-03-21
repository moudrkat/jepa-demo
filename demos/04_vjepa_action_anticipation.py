"""
Demo 04 — "What Happens Next?" Progressive Action Anticipation
===============================================================

Shows how V-JEPA 2's confidence evolves as it sees more of a video.
Early on it's uncertain; as more frames are revealed, it locks onto
the correct action.

This isn't true latent-space anticipation (that needs a different head),
but it tells the same compelling story: the model recognizes actions
before they're complete.

Run:
    python demos/04_vjepa_action_anticipation.py
"""

import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

SAMPLE_VIDEO_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"


# ---------------------------------------------------------------------------
# Reuse loading utilities from demo 03
# ---------------------------------------------------------------------------
def load_video_opencv(url, max_frames=64):
    import cv2
    import tempfile
    import urllib.request

    print(f"  Downloading video...")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    urllib.request.urlretrieve(url, tmp.name)

    cap = cv2.VideoCapture(tmp.name)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    print(f"  Loaded {len(frames)} frames")
    return np.stack(frames)


def load_model():
    from transformers import AutoVideoProcessor, AutoModelForVideoClassification
    print(f"  Loading {MODEL_ID} ...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
    model.eval()
    model.to(DEVICE)
    print(f"  Model loaded")
    return model, processor


# ---------------------------------------------------------------------------
# Progressive classification
# ---------------------------------------------------------------------------
def progressive_classify(model, processor, all_frames, fractions=(0.25, 0.50, 0.75, 1.0)):
    """
    For each fraction of the video, pad remaining with the last visible frame
    and classify. Returns dict of fraction -> (predictions, visible_frames).
    """
    total = len(all_frames)
    n_input = 16
    results = {}

    for frac in fractions:
        n_visible = max(1, int(total * frac))
        visible = all_frames[:n_visible]

        # Sample n_input frames from visible portion
        if n_visible >= n_input:
            indices = np.linspace(0, n_visible - 1, n_input, dtype=int)
            frames_input = visible[indices]
        else:
            # Pad with last visible frame
            indices = np.linspace(0, n_visible - 1, n_visible, dtype=int)
            sampled = visible[indices]
            pad = np.tile(sampled[-1:], (n_input - len(sampled), 1, 1, 1))
            frames_input = np.concatenate([sampled, pad], axis=0)

        # Classify
        inputs = processor(frames_input, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        top10_idx = probs.argsort(descending=True)[:10]
        preds = []
        for idx in top10_idx:
            label = model.config.id2label[idx.item()]
            preds.append((label, probs[idx].item()))

        results[frac] = {
            "predictions": preds,
            "visible_frames": frames_input,
            "n_visible_original": n_visible,
            "all_probs": probs.cpu().numpy(),
        }

        pct = int(frac * 100)
        print(f"    {pct:3d}% → {preds[0][0]} ({preds[0][1]:.1%})")

    return results


# ---------------------------------------------------------------------------
# Visualization 1: Progressive prediction grid
# ---------------------------------------------------------------------------
def plot_progressive_grid(results, all_frames, save_path):
    """Grid: rows = fractions, cols = frames + prediction bar."""
    fractions = sorted(results.keys())
    n_rows = len(fractions)
    n_frame_cols = 5

    fig = plt.figure(figsize=(18, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

    for row, frac in enumerate(fractions):
        data = results[frac]
        pct = int(frac * 100)

        # Left: frame strip
        ax_frames = fig.add_subplot(gs[row, 0])
        visible = data["visible_frames"]
        strip_indices = np.linspace(0, len(visible) - 1, n_frame_cols, dtype=int)

        # Create a horizontal strip
        strip_h = visible[0].shape[0]
        strip_w = visible[0].shape[1]
        gap = 4
        canvas_w = n_frame_cols * strip_w + (n_frame_cols - 1) * gap
        canvas = np.ones((strip_h, canvas_w, 3), dtype=np.uint8) * 240

        for i, si in enumerate(strip_indices):
            x0 = i * (strip_w + gap)
            canvas[:, x0:x0 + strip_w] = visible[si]

        ax_frames.imshow(canvas)
        ax_frames.axis("off")

        # Progress bar overlay
        n_vis = data["n_visible_original"]
        n_total = len(all_frames)
        ax_frames.set_title(f"{pct}% of video seen ({n_vis}/{n_total} frames)", fontsize=13)

        # Color border based on fraction
        color = plt.cm.RdYlGn(frac)
        for spine in ax_frames.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)

        # Right: top-5 predictions
        ax_bar = fig.add_subplot(gs[row, 1])
        preds = data["predictions"][:5]
        labels = [p[0] for p in preds]
        probs = [p[1] for p in preds]

        bar_colors = ["#2196F3"] + ["#B0BEC5"] * (len(labels) - 1)
        bars = ax_bar.barh(range(len(labels)), probs, color=bar_colors, edgecolor="white")
        ax_bar.set_yticks(range(len(labels)))
        ax_bar.set_yticklabels(labels, fontsize=10)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, 1)
        ax_bar.grid(axis="x", alpha=0.2)

        for bar, prob in zip(bars, probs):
            ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{prob:.1%}", va="center", fontsize=10)

    plt.suptitle(
        '"What Happens Next?" — V-JEPA 2 Progressive Action Recognition',
        fontsize=18, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 2: Confidence evolution line chart
# ---------------------------------------------------------------------------
def plot_confidence_evolution(results, save_path):
    """Line chart: top class probability vs video fraction."""
    fractions = sorted(results.keys())

    # Collect all unique top-3 labels across fractions
    all_top_labels = set()
    for frac in fractions:
        for label, _ in results[frac]["predictions"][:3]:
            all_top_labels.add(label)

    # Track probability of each label across fractions
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_top_labels)))

    for i, label in enumerate(sorted(all_top_labels)):
        probs = []
        for frac in fractions:
            # Find this label in predictions
            prob = 0
            for l, p in results[frac]["predictions"]:
                if l == label:
                    prob = p
                    break
            probs.append(prob)

        ax.plot(
            [f * 100 for f in fractions], probs,
            "o-", color=colors[i], linewidth=2, markersize=8,
            label=label[:40],  # truncate long labels
        )

    ax.set_xlabel("Video Seen (%)", fontsize=13)
    ax.set_ylabel("Confidence", fontsize=13)
    ax.set_title("How Confidence Evolves as More Video Is Revealed", fontsize=16)
    ax.set_xlim(15, 105)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 3: "Reveal" — question mark overlay
# ---------------------------------------------------------------------------
def plot_reveal(all_frames, results, save_path):
    """Show first half with '?' overlay, then the prediction, then the full video."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    half = len(all_frames) // 2

    # Panel 1: First half
    ax = axes[0]
    ax.imshow(all_frames[half // 2])  # middle of first half
    ax.set_title("What you see...", fontsize=14)
    ax.axis("off")

    # Panel 2: Question mark
    ax = axes[1]
    ax.imshow(all_frames[half + (len(all_frames) - half) // 2], alpha=0.15)
    ax.text(0.5, 0.5, "?", transform=ax.transAxes, fontsize=100,
            ha="center", va="center", color="#F44336", fontweight="bold", alpha=0.8)
    ax.set_title("What happens next?", fontsize=14)
    ax.axis("off")

    # Panel 3: Answer
    ax = axes[2]
    ax.imshow(all_frames[-1])
    top_pred = results[1.0]["predictions"][0]
    ax.set_title(f'V-JEPA says:\n"{top_pred[0]}"\n({top_pred[1]:.0%} confidence)',
                 fontsize=13, color="#2196F3")
    ax.axis("off")

    plt.suptitle("V-JEPA 2: Anticipating Actions", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print('  Demo 04 — "What Happens Next?" Anticipation')
    print("=" * 55)

    model, processor = load_model()
    all_frames = load_video_opencv(SAMPLE_VIDEO_URL)

    print("\n  Progressive classification:")
    results = progressive_classify(
        model, processor, all_frames,
        fractions=(0.25, 0.50, 0.75, 1.0),
    )

    print("\n  [1/3] Progressive prediction grid...")
    plot_progressive_grid(results, all_frames, OUTPUT_DIR / "04_progressive.png")

    print("  [2/3] Confidence evolution...")
    plot_confidence_evolution(results, OUTPUT_DIR / "04_confidence.png")

    print("  [3/3] Reveal visualization...")
    plot_reveal(all_frames, results, OUTPUT_DIR / "04_reveal.png")

    del model
    gc.collect()
    print(f"\n  Done! Check {OUTPUT_DIR}/04_*.png")


if __name__ == "__main__":
    main()
