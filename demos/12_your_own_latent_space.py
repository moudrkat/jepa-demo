"""
Demo 12 – Your Own Latent Space
================================
Feed a long video (minutes, not seconds) and V-JEPA 2 will:
  1. Slide a window across the video extracting embeddings
  2. Cluster them with k-means (discovers action segments)
  3. Build a t-SNE map with frame thumbnails
  4. Create an animated cluster journey GIF
  5. Show a timeline of discovered action segments

Usage:
    python demos/12_your_own_latent_space.py path/to/long_video.mp4
    python demos/12_your_own_latent_space.py video.mp4 --name playground --clusters 6
    python demos/12_your_own_latent_space.py video.mp4 --stride 8  # faster, fewer embeddings
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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs")
DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"  # pretrained, no fine-tuning

WINDOW_SIZE = 16
FPS_GIF = 3
N_HOLD = 8

# Distinct colors for up to 12 clusters
CLUSTER_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324",
]


def load_video(path, max_frames=None):
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {path}")
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps_src if fps_src > 0 else 0

    frames = []
    while cap.isOpened():
        if max_frames and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    print(f"  {len(frames)} frames, {fps_src:.0f} fps, ~{duration:.1f}s")
    return frames, fps_src


def load_model():
    from transformers import AutoVideoProcessor, AutoModel

    print("  Loading V-JEPA 2 (pretrained, no labels)...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params")
    return model, processor


def extract_embeddings(model, processor, frames, stride):
    """Slide a window and extract one embedding per position."""
    T = len(frames)
    n_windows = max(1, (T - WINDOW_SIZE) // stride + 1)
    print(f"  {n_windows} windows (size={WINDOW_SIZE}, stride={stride})")

    embeddings = []
    centers = []
    center_frames = []

    for start in range(0, T - WINDOW_SIZE + 1, stride):
        window = frames[start : start + WINDOW_SIZE]
        inputs = processor(window, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings.append(emb)

        center = start + WINDOW_SIZE // 2
        centers.append(center)
        center_frames.append(frames[center])

        done = len(embeddings)
        if done % 10 == 0 or done == n_windows:
            print(f"    [{done}/{n_windows}]")

    return np.array(embeddings), centers, center_frames


# ---------- Timeline ----------

def plot_timeline(cluster_labels, centers, fps, n_clusters, save_path):
    """Horizontal timeline showing discovered action segments."""
    fig, ax = plt.subplots(figsize=(18, 3))
    fig.patch.set_facecolor("#FAFAFA")

    n = len(cluster_labels)
    for i in range(n):
        t_start = centers[i] / fps
        t_end = centers[min(i + 1, n - 1)] / fps if i < n - 1 else centers[i] / fps + 0.5
        color = CLUSTER_COLORS[cluster_labels[i] % len(CLUSTER_COLORS)]
        ax.barh(0, t_end - t_start, left=t_start, height=0.8,
                color=color, edgecolor="white", linewidth=0.5)

    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontsize=13)
    ax.set_title("Discovered Action Segments (no labels — pure V-JEPA 2 embeddings + k-means)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, centers[-1] / fps + 0.5)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
               for i in range(n_clusters)]
    ax.legend(handles, [f"Cluster {i}" for i in range(n_clusters)],
              loc="upper right", fontsize=9, ncol=min(n_clusters, 6))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------- t-SNE with thumbnails ----------

def plot_tsne_thumbnails(tsne_2d, cluster_labels, center_frames, n_clusters, save_path):
    """t-SNE scatter with frame thumbnails."""
    emb_min = tsne_2d.min(axis=0)
    emb_max = tsne_2d.max(axis=0)
    norm = 2 * (tsne_2d - emb_min) / (emb_max - emb_min + 1e-8) - 1

    fig, ax = plt.subplots(figsize=(16, 13))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    # Show every Nth frame as thumbnail
    n = len(center_frames)
    step = max(1, n // 60)
    for i in range(0, n, step):
        img = center_frames[i]
        thumb = Image.fromarray(img).resize((32, 32), Image.LANCZOS)
        color = CLUSTER_COLORS[cluster_labels[i] % len(CLUSTER_COLORS)]

        im = OffsetImage(np.array(thumb), zoom=1.0)
        ab = AnnotationBbox(
            im, (norm[i, 0], norm[i, 1]),
            frameon=True,
            bboxprops=dict(edgecolor=color, linewidth=2,
                           facecolor="#1A1A2E", alpha=0.95),
            pad=0.1,
        )
        ax.add_artist(ab)

    # Cluster labels at centroids
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        cx, cy = norm[mask].mean(axis=0)
        ax.annotate(
            f"Cluster {c}",
            xy=(cx, cy), xytext=(cx, cy + 0.15),
            fontsize=11, fontweight="bold", color="white",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                      alpha=0.85, edgecolor="white", linewidth=0.5),
            arrowprops=dict(arrowstyle="-",
                            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                            lw=1.5, alpha=0.6),
        )

    ax.set_title("Your video's latent space — V-JEPA 2 embeddings (t-SNE)",
                 fontsize=16, fontweight="bold", color="white", pad=15)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------- Cluster journey GIF ----------

def create_journey_gif(tsne_2d, cluster_labels, center_frames, n_clusters, save_path):
    """Animated GIF: video frame + moving dot in t-SNE space."""
    print("  Creating cluster journey GIF...")
    n = len(tsne_2d)

    emb_min = tsne_2d.min(axis=0)
    emb_max = tsne_2d.max(axis=0)
    norm = 2 * (tsne_2d - emb_min) / (emb_max - emb_min + 1e-8) - 1

    colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in cluster_labels]

    step = max(1, n // 80)
    frame_indices = list(range(0, n, step))
    total_frames = len(frame_indices) + N_HOLD

    fig, (ax_vid, ax_tsne) = plt.subplots(1, 2, figsize=(16, 7),
                                           gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor("#1A1A2E")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08, wspace=0.15)

    ax_tsne.set_facecolor("#1A1A2E")
    ax_tsne.set_xlim(-1.3, 1.3)
    ax_tsne.set_ylim(-1.3, 1.3)
    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])
    for spine in ax_tsne.spines.values():
        spine.set_visible(False)
    ax_tsne.set_title("Your latent space (t-SNE)", fontsize=14,
                       fontweight="bold", color="white")

    # Background points
    ax_tsne.scatter(norm[:, 0], norm[:, 1], c=colors, s=20, alpha=0.12, edgecolors="none")

    # Legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                           markersize=8, label=f"Cluster {i}", linestyle="None")
               for i in range(n_clusters)]
    ax_tsne.legend(handles=handles, loc="upper left", fontsize=9,
                    facecolor="#2A2A4A", edgecolor="#555", labelcolor="white",
                    ncol=2, framealpha=0.9)

    trail_line, = ax_tsne.plot([], [], color="white", linewidth=1.5, alpha=0.5)
    current_dot = ax_tsne.scatter([], [], s=200, c="white", edgecolors="white",
                                   linewidths=2, zorder=10)
    visited_scatter = ax_tsne.scatter([], [], s=35, c=[], edgecolors="white",
                                      linewidths=0.5, alpha=0.8, zorder=5)

    ax_vid.set_facecolor("#1A1A2E")
    ax_vid.set_xticks([])
    ax_vid.set_yticks([])

    progress_bg = plt.Rectangle((0.02, 0.03), 0.45, 0.02,
                                 transform=fig.transFigure, facecolor="#333355",
                                 edgecolor="none", zorder=10)
    progress_fill = plt.Rectangle((0.02, 0.03), 0.0, 0.02,
                                   transform=fig.transFigure, facecolor="#4FC3F7",
                                   edgecolor="none", zorder=11)
    fig.patches.extend([progress_bg, progress_fill])

    fig.suptitle("Your Video — Cluster Journey",
                 fontsize=16, fontweight="bold", color="white", y=0.96)

    def update(i):
        idx = min(i, len(frame_indices) - 1)
        emb_idx = frame_indices[idx]
        cluster = cluster_labels[emb_idx]
        color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
        frac = (idx + 1) / len(frame_indices)

        ax_vid.clear()
        ax_vid.imshow(center_frames[emb_idx])
        ax_vid.set_title(f"Cluster {cluster}", fontsize=14,
                         fontweight="bold", color=color)
        ax_vid.set_xticks([])
        ax_vid.set_yticks([])
        for spine in ax_vid.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)

        visited_idx = frame_indices[:idx + 1]
        trail_x = norm[visited_idx, 0]
        trail_y = norm[visited_idx, 1]
        trail_line.set_data(trail_x, trail_y)

        visited_xy = np.column_stack([trail_x, trail_y])
        visited_colors = [colors[fi] for fi in visited_idx]
        visited_scatter.set_offsets(visited_xy)
        visited_scatter.set_facecolors(visited_colors)

        cx, cy = norm[emb_idx]
        current_dot.set_offsets([[cx, cy]])
        current_dot.set_facecolors([color])

        progress_fill.set_width(0.45 * frac)

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // FPS_GIF, blit=False)
    anim.save(str(save_path), writer="pillow", fps=FPS_GIF, dpi=100)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Build your own latent space from a video")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--name", type=str, default=None,
                        help="Output name prefix (default: video filename)")
    parser.add_argument("--clusters", type=int, default=6,
                        help="Number of k-means clusters (default: 6)")
    parser.add_argument("--stride", type=int, default=4,
                        help="Window stride in frames (default: 4, higher = faster)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to load (default: all)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: {video_path} not found")
        sys.exit(1)

    name = args.name or video_path.stem
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"  Your Own Latent Space: {video_path.name}")
    print("=" * 60)

    print("\n[1/6] Loading video...")
    frames, fps = load_video(video_path, max_frames=args.max_frames)

    print("\n[2/6] Loading model...")
    model, processor = load_model()

    print("\n[3/6] Extracting embeddings...")
    embeddings, centers, center_frames = extract_embeddings(
        model, processor, frames, stride=args.stride)
    print(f"  Shape: {embeddings.shape}")

    del model
    gc.collect()

    print("\n[4/6] Computing t-SNE + k-means...")
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
    tsne_2d = tsne.fit_transform(embeddings)

    km = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(embeddings)

    # Save embeddings
    np.savez(OUTPUT_DIR / f"12_embeddings_{name}.npz",
             embeddings=embeddings, tsne_2d=tsne_2d,
             centers=centers, cluster_labels=cluster_labels)

    print("\n[5/6] Creating visualizations...")
    plot_timeline(cluster_labels, centers, fps, args.clusters,
                  OUTPUT_DIR / f"12_timeline_{name}.png")
    plot_tsne_thumbnails(tsne_2d, cluster_labels, center_frames, args.clusters,
                         OUTPUT_DIR / f"12_tsne_{name}.png")

    print("\n[6/6] Creating cluster journey GIF...")
    create_journey_gif(tsne_2d, cluster_labels, center_frames, args.clusters,
                       OUTPUT_DIR / f"12_journey_{name}.gif")

    print(f"\n✓ Done! Your latent space is ready:")
    print(f"  Timeline:    outputs/12_timeline_{name}.png")
    print(f"  t-SNE map:   outputs/12_tsne_{name}.png")
    print(f"  Journey GIF: outputs/12_journey_{name}.gif")
    print(f"  Embeddings:  outputs/12_embeddings_{name}.npz")


if __name__ == "__main__":
    main()