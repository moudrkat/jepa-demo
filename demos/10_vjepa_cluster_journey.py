"""
Demo 10 – Animated cluster journey: video + t-SNE side by side
==============================================================
Shows a video playing on the left while a dot traces through the
embedding space on the right, revealing how V-JEPA 2 moves between
semantic clusters as the action changes.

Uses the pretrained (non-fine-tuned) model to show what pure
self-supervised learning captures.

Outputs:
  10_cluster_journey.gif — animated side-by-side GIF
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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs")
DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

WINDOW_SIZE = 16
STRIDE = 4
FPS = 3
N_HOLD_FRAMES = 8

BASE_URL = (
    "https://huggingface.co/datasets/Nojah/"
    "limited_something_something_v2/resolve/main/videos"
)

CLIPS = {
    "pouring":      f"{BASE_URL}/9704.webm",
    "folding":      f"{BASE_URL}/4648.webm",
    "transferring": f"{BASE_URL}/115408.webm",
    "unscrewing":   f"{BASE_URL}/129954.webm",
    "placing":      f"{BASE_URL}/76271.webm",
    "lifting":      f"{BASE_URL}/90845.webm",
    "opening":      f"{BASE_URL}/134611.webm",
    "sorting":      f"{BASE_URL}/173061.webm",
}

ACTION_COLORS = {
    "pouring":      "#e6194b",
    "folding":      "#3cb44b",
    "transferring": "#4363d8",
    "unscrewing":   "#f58231",
    "placing":      "#911eb4",
    "lifting":      "#42d4f4",
    "opening":      "#f032e6",
    "sorting":      "#bfef45",
}


def download_video(url):
    suffix = Path(url).suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


def load_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def build_concat_video():
    all_frames = []
    segment_labels = []
    for action, url in CLIPS.items():
        print(f"  Downloading {action}...")
        path = download_video(url)
        frames = load_frames(path)
        print(f"    {len(frames)} frames")
        all_frames.extend(frames)
        segment_labels.extend([action] * len(frames))

    heights = [f.shape[0] for f in all_frames]
    widths = [f.shape[1] for f in all_frames]
    h, w = min(heights), min(widths)
    all_frames = [cv2.resize(f, (w, h)) for f in all_frames]
    print(f"  Total: {len(all_frames)} frames, resized to {w}x{h}")
    return np.stack(all_frames), segment_labels


def load_model():
    from transformers import AutoVideoProcessor, AutoModel

    print(f"  Loading {MODEL_ID} (pretrained)...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params")
    return model, processor


def extract_windowed_embeddings(model, processor, video):
    T = len(video)
    embeddings = []
    window_centers = []
    n_windows = max(1, (T - WINDOW_SIZE) // STRIDE + 1)
    print(f"  Extracting {n_windows} embeddings...")

    for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
        frames_16 = video[start : start + WINDOW_SIZE]
        inputs = processor(frames_16, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings.append(emb)
        window_centers.append(start + WINDOW_SIZE // 2)

        done = len(embeddings)
        if done % 10 == 0 or done == n_windows:
            print(f"    [{done}/{n_windows}]")

    return np.array(embeddings), window_centers


def create_journey_gif(
    video, segment_labels, embeddings, window_centers, tsne_2d, cluster_labels,
):
    """Animated GIF: video frame (left) + moving dot in t-SNE space (right)."""
    print("  Creating cluster journey GIF...")
    n_emb = len(embeddings)

    # Normalize t-SNE to [-1, 1]
    emb_min = tsne_2d.min(axis=0)
    emb_max = tsne_2d.max(axis=0)
    tsne_norm = 2 * (tsne_2d - emb_min) / (emb_max - emb_min) - 1

    # Map each embedding to its ground-truth action
    emb_actions = [segment_labels[c] for c in window_centers]
    emb_colors = [ACTION_COLORS[a] for a in emb_actions]

    # Step through every 2nd embedding for manageable GIF length
    step = 2
    frame_indices = list(range(0, n_emb, step))
    total_frames = len(frame_indices) + N_HOLD_FRAMES

    fig, (ax_vid, ax_tsne) = plt.subplots(1, 2, figsize=(16, 7),
                                           gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor("#1A1A2E")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08, wspace=0.15)

    # -- Set up t-SNE axes --
    ax_tsne.set_facecolor("#1A1A2E")
    ax_tsne.set_xlim(-1.3, 1.3)
    ax_tsne.set_ylim(-1.3, 1.3)
    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])
    for spine in ax_tsne.spines.values():
        spine.set_visible(False)
    ax_tsne.set_title("Embedding space (t-SNE)", fontsize=14,
                       fontweight="bold", color="white")

    # Plot all points as faint background
    ax_tsne.scatter(tsne_norm[:, 0], tsne_norm[:, 1],
                    c=emb_colors, s=25, alpha=0.15, edgecolors="none")

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=ACTION_COLORS[a], markersize=8,
                    label=a, linestyle="None")
        for a in CLIPS.keys()
    ]
    ax_tsne.legend(handles=legend_handles, loc="upper left", fontsize=9,
                    facecolor="#2A2A4A", edgecolor="#555", labelcolor="white",
                    ncol=2, framealpha=0.9)

    # Trail line and current dot (will be updated each frame)
    trail_line, = ax_tsne.plot([], [], color="white", linewidth=1.5, alpha=0.6)
    current_dot = ax_tsne.scatter([], [], s=200, c="white", edgecolors="white",
                                   linewidths=2, zorder=10)
    # Visited dots (accumulate)
    visited_scatter = ax_tsne.scatter([], [], s=40, c=[], edgecolors="white",
                                      linewidths=0.5, alpha=0.8, zorder=5)

    # Video axes
    ax_vid.set_facecolor("#1A1A2E")
    ax_vid.set_xticks([])
    ax_vid.set_yticks([])
    for spine in ax_vid.spines.values():
        spine.set_color("#4FC3F7")
        spine.set_linewidth(2)

    # Progress bar
    progress_bg = plt.Rectangle(
        (0.02, 0.03), 0.45, 0.02,
        transform=fig.transFigure, facecolor="#333355", edgecolor="none", zorder=10,
    )
    progress_fill = plt.Rectangle(
        (0.02, 0.03), 0.0, 0.02,
        transform=fig.transFigure, facecolor="#4FC3F7", edgecolor="none", zorder=11,
    )
    fig.patches.extend([progress_bg, progress_fill])

    # Suptitle
    fig.suptitle("V-JEPA 2 — Cluster Journey (pretrained, no labels)",
                 fontsize=16, fontweight="bold", color="white", y=0.96)

    def update(i):
        idx = min(i, len(frame_indices) - 1)
        emb_idx = frame_indices[idx]
        center_frame = window_centers[emb_idx]
        action = emb_actions[emb_idx]
        frac = (idx + 1) / len(frame_indices)

        # Video frame
        ax_vid.clear()
        ax_vid.imshow(video[center_frame])
        ax_vid.set_title(f"Action: {action}", fontsize=14, fontweight="bold",
                         color=ACTION_COLORS[action])
        ax_vid.set_xticks([])
        ax_vid.set_yticks([])
        for spine in ax_vid.spines.values():
            spine.set_color(ACTION_COLORS[action])
            spine.set_linewidth(3)

        # Trail: all visited points up to now
        visited_idx = frame_indices[:idx + 1]
        trail_x = tsne_norm[visited_idx, 0]
        trail_y = tsne_norm[visited_idx, 1]
        trail_line.set_data(trail_x, trail_y)

        # Visited dots with action colors
        visited_xy = np.column_stack([trail_x, trail_y])
        visited_colors = [emb_colors[fi] for fi in visited_idx]
        visited_scatter.set_offsets(visited_xy)
        visited_scatter.set_facecolors(visited_colors)

        # Current position — large dot
        cx, cy = tsne_norm[emb_idx]
        current_dot.set_offsets([[cx, cy]])
        current_dot.set_facecolors([ACTION_COLORS[action]])

        # Progress bar
        progress_fill.set_width(0.45 * frac)

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // FPS, blit=False,
    )

    save_path = OUTPUT_DIR / "10_cluster_journey.gif"
    anim.save(str(save_path), writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Demo 10: V-JEPA 2 Cluster Journey (pretrained)")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    model, processor = load_model()

    print("\n[2/5] Building concatenated video...")
    video, segment_labels = build_concat_video()

    print("\n[3/5] Extracting embeddings...")
    embeddings, window_centers = extract_windowed_embeddings(model, processor, video)
    print(f"  Shape: {embeddings.shape}")

    del model
    gc.collect()

    print("\n[4/5] Computing t-SNE + k-means...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_2d = tsne.fit_transform(embeddings)

    km = KMeans(n_clusters=len(CLIPS), random_state=42, n_init=10)
    cluster_labels = km.fit_predict(embeddings)

    # Save embeddings
    np.savez(
        OUTPUT_DIR / "10_embeddings.npz",
        embeddings=embeddings, tsne_2d=tsne_2d,
        window_centers=window_centers, cluster_labels=cluster_labels,
    )

    print("\n[5/5] Creating cluster journey GIF...")
    create_journey_gif(
        video, segment_labels, embeddings, window_centers, tsne_2d, cluster_labels,
    )

    print("\n✓ Done! Check outputs/10_cluster_journey.gif")


if __name__ == "__main__":
    main()