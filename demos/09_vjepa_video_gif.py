"""
Demo 09 – Animated V-JEPA 2 video prediction GIF
=================================================
Shows V-JEPA 2 processing a video frame by frame with a live prediction
bar chart that updates as the model sees more of the video.

Perfect for embedding in a podcast or presentation — shows the model
"thinking" and narrowing down its prediction in real time.

Outputs:
  09_vjepa_live_prediction.gif  — animated GIF with frame + prediction bars
  09_vjepa_multi_video.gif      — side-by-side GIF of 3 different videos
"""

import gc
import tempfile
import urllib.request
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

# Three visually dramatic action videos from Something-Something V2
BASE_URL = (
    "https://huggingface.co/datasets/Nojah/limited_something_something_v2"
    "/resolve/main/videos"
)
VIDEOS = {
    "Transferring between bowls": f"{BASE_URL}/115408.webm",
    "Unscrewing a cap": f"{BASE_URL}/129954.webm",
    "Sorting crayons": f"{BASE_URL}/173061.webm",
}

FPS = 4  # slower for readability
N_STEPS = 24  # more steps for smoother progression
N_HOLD_FRAMES = 6  # hold final frame for emphasis
TOP_K = 5  # number of predictions to show


def load_video(url, max_frames=64):
    """Download and load video frames."""
    suffix = Path(url).suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    cap = cv2.VideoCapture(tmp.name)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def load_model():
    from transformers import AutoVideoProcessor, AutoModelForVideoClassification

    print("Loading V-JEPA 2...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
    model.eval().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.0f}M parameters")
    return model, processor


def classify_partial(model, processor, all_frames, fraction):
    """Classify using only the first `fraction` of the video."""
    total = len(all_frames)
    n_visible = max(1, int(total * fraction))
    visible = all_frames[:n_visible]
    n_input = 16

    if n_visible >= n_input:
        indices = np.linspace(0, n_visible - 1, n_input, dtype=int)
        frames = visible[indices]
    else:
        indices = np.linspace(0, n_visible - 1, n_visible, dtype=int)
        sampled = visible[indices]
        pad = np.tile(sampled[-1:], (n_input - len(sampled), 1, 1, 1))
        frames = np.concatenate([sampled, pad], axis=0)

    inputs = processor(frames, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_idx = probs.argsort(descending=True)[:TOP_K]
    preds = []
    for idx in top_idx:
        label = model.config.id2label[idx.item()]
        preds.append((label, probs[idx].item()))

    return preds, visible[-1]  # return last visible frame


def create_single_video_gif(model, processor, all_frames, video_name, save_path):
    """Create animated GIF for a single video with live prediction bars."""
    print(f"  Animating: {video_name}")
    fractions = np.linspace(0.1, 1.0, N_STEPS)

    # Pre-compute all predictions
    all_preds = []
    all_display_frames = []
    for frac in fractions:
        preds, frame = classify_partial(model, processor, all_frames, frac)
        all_preds.append(preds)
        all_display_frames.append(frame)

    # Create animation
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#1A1A2E")

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.3)
    ax_frame = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])

    # Progress bar area
    progress_bg = plt.Rectangle(
        (0.05, 0.04), 0.38, 0.03,
        transform=fig.transFigure, facecolor="#333355", edgecolor="none", zorder=10,
    )
    progress_fill = plt.Rectangle(
        (0.05, 0.04), 0.0, 0.03,
        transform=fig.transFigure, facecolor="#4FC3F7", edgecolor="none", zorder=11,
    )
    fig.patches.extend([progress_bg, progress_fill])

    # Add hold frames at end (repeat last frame)
    total_frames = N_STEPS + N_HOLD_FRAMES

    def update(i):
        idx = min(i, N_STEPS - 1)
        frac = fractions[idx]
        preds = all_preds[idx]
        frame = all_display_frames[idx]
        pct = int(frac * 100)

        # Video frame
        ax_frame.clear()
        ax_frame.imshow(frame)
        ax_frame.set_title(f"Video: {pct}% seen", fontsize=14, fontweight="bold", color="white")
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        for spine in ax_frame.spines.values():
            spine.set_color("#4FC3F7")
            spine.set_linewidth(2)

        # Prediction bars
        ax_bars.clear()
        labels = [p[0][:35] for p in preds]
        probs = [p[1] for p in preds]

        bar_colors = ["#4FC3F7"] + ["#555577"] * (len(labels) - 1)
        bars = ax_bars.barh(range(len(labels)), probs, color=bar_colors,
                            edgecolor="none", height=0.6)
        ax_bars.set_yticks(range(len(labels)))
        ax_bars.set_yticklabels(labels, fontsize=10, color="white")
        ax_bars.invert_yaxis()
        ax_bars.set_xlim(0, 1.05)
        ax_bars.set_facecolor("#1A1A2E")
        ax_bars.tick_params(colors="white")
        ax_bars.set_title("V-JEPA 2 predictions", fontsize=14, fontweight="bold", color="white")

        for bar, prob in zip(bars, probs):
            ax_bars.text(
                bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{prob:.0%}", va="center", fontsize=11, color="white", fontweight="bold",
            )

        ax_bars.grid(axis="x", alpha=0.1, color="white")
        ax_bars.spines["top"].set_visible(False)
        ax_bars.spines["right"].set_visible(False)
        ax_bars.spines["bottom"].set_color("#555577")
        ax_bars.spines["left"].set_color("#555577")

        # Progress bar
        progress_fill.set_width(0.38 * frac)

        return bars

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // FPS, blit=False,
    )
    anim.save(str(save_path), writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def create_multi_video_gif(model, processor, videos_data, save_path):
    """Side-by-side GIF: 3 videos classified simultaneously."""
    print("  Creating multi-video comparison GIF...")
    n_vids = len(videos_data)
    fractions = np.linspace(0.1, 1.0, N_STEPS)

    # Pre-compute predictions for all videos
    all_results = {}
    for name, frames in videos_data.items():
        preds_list = []
        frames_list = []
        for frac in fractions:
            preds, frame = classify_partial(model, processor, frames, frac)
            preds_list.append(preds)
            frames_list.append(frame)
        all_results[name] = (preds_list, frames_list)

    fig, axes = plt.subplots(2, n_vids, figsize=(6 * n_vids, 8),
                              gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.35})
    fig.patch.set_facecolor("#1A1A2E")

    # Progress bar
    progress_bg = plt.Rectangle(
        (0.1, 0.02), 0.8, 0.02,
        transform=fig.transFigure, facecolor="#333355", edgecolor="none", zorder=10,
    )
    progress_fill = plt.Rectangle(
        (0.1, 0.02), 0.0, 0.02,
        transform=fig.transFigure, facecolor="#4FC3F7", edgecolor="none", zorder=11,
    )
    fig.patches.extend([progress_bg, progress_fill])

    names = list(videos_data.keys())
    total_frames = N_STEPS + N_HOLD_FRAMES

    def update(i):
        idx = min(i, N_STEPS - 1)
        frac = fractions[idx]
        pct = int(frac * 100)

        for col, name in enumerate(names):
            preds, frame = all_results[name][0][idx], all_results[name][1][idx]

            # Top row: video frame
            ax = axes[0, col]
            ax.clear()
            ax.imshow(frame)
            ax.set_title(f"{pct}% seen", fontsize=12, fontweight="bold", color="white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#4FC3F7")
                spine.set_linewidth(1.5)

            # Bottom row: top-3 predictions
            ax = axes[1, col]
            ax.clear()
            labels = [p[0][:30] for p in preds[:3]]
            probs = [p[1] for p in preds[:3]]

            bar_colors = ["#4FC3F7"] + ["#555577"] * (len(labels) - 1)
            bars = ax.barh(range(len(labels)), probs, color=bar_colors,
                           edgecolor="none", height=0.5)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9, color="white")
            ax.invert_yaxis()
            ax.set_xlim(0, 1.15)
            ax.set_facecolor("#1A1A2E")
            ax.tick_params(colors="white")

            for bar, prob in zip(bars, probs):
                ax.text(
                    bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.0%}", va="center", fontsize=10, color="white",
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#555577")
            ax.spines["left"].set_color("#555577")

        fig.suptitle(
            f"V-JEPA 2 — Real-time Action Recognition",
            fontsize=16, fontweight="bold", color="white", y=0.98,
        )
        progress_fill.set_width(0.8 * frac)

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // FPS, blit=False,
    )
    anim.save(str(save_path), writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(42)

    print("=" * 60)
    print("Demo 09: Animated V-JEPA 2 Video Prediction")
    print("=" * 60)

    print("\n[1/4] Loading model...")
    model, processor = load_model()

    print("\n[2/4] Downloading videos...")
    videos_data = {}
    for name, url in VIDEOS.items():
        print(f"  {name}...")
        frames = load_video(url)
        videos_data[name] = frames
        print(f"    {len(frames)} frames loaded")

    print("\n[3/4] Creating single-video GIF (first video)...")
    first_name = list(VIDEOS.keys())[0]
    create_single_video_gif(
        model, processor, videos_data[first_name], first_name,
        OUTPUT_DIR / "09_vjepa_live_prediction.gif",
    )

    print("\n[4/4] Creating multi-video comparison GIF...")
    create_multi_video_gif(
        model, processor, videos_data,
        OUTPUT_DIR / "09_vjepa_multi_video.gif",
    )

    del model
    gc.collect()
    print("\n✓ Done! Check outputs/09_*.gif")


if __name__ == "__main__":
    main()