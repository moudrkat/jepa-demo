"""
Demo 06 — V-JEPA 2 Temporal Cluster Analysis (pretrained, NOT fine-tuned)
=========================================================================

Same cluster analysis as Demo 05, but using the **base pretrained** V-JEPA 2
model (no SSv2 fine-tuning).  This shows what the self-supervised representation
learns purely from video prediction — no action labels involved.

Run:
    python demos/06_vjepa_cluster_pretrained.py
"""

import gc
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"
# Base pretrained model — NO fine-tuning on SSv2 labels
# Note: base uses fpc64 (64 frames-per-clip); the SSv2-finetuned variant uses fpc16
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

WINDOW_SIZE = 16        # frames per embedding (model's temporal receptive field)
STRIDE = 4             # slide the window by this many frames

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Video clips — distinct SSv2 hand-object actions
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Video loading & concatenation
# ---------------------------------------------------------------------------
def download_video(url):
    """Download a video to a temp file and return the path."""
    import urllib.request
    suffix = Path(url).suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


def load_frames(path):
    """Load all frames from a video file. Returns list of (H, W, 3) uint8."""
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
    """Download clips and concatenate into one video.

    Returns:
        frames: np.ndarray (T, H, W, 3) — the full concatenated video
        segment_labels: list[str] — ground-truth action label per frame
    """
    all_frames = []
    segment_labels = []

    for action, url in CLIPS.items():
        print(f"  Downloading {action}...")
        path = download_video(url)
        frames = load_frames(path)
        print(f"    {len(frames)} frames")
        all_frames.extend(frames)
        segment_labels.extend([action] * len(frames))

    # Resize all frames to a common size (smallest common dimensions)
    heights = [f.shape[0] for f in all_frames]
    widths = [f.shape[1] for f in all_frames]
    h, w = min(heights), min(widths)
    all_frames = [cv2.resize(f, (w, h)) for f in all_frames]

    print(f"  Total: {len(all_frames)} frames, resized to {w}x{h}")
    return np.stack(all_frames), segment_labels


# ---------------------------------------------------------------------------
# Model & embedding extraction
# ---------------------------------------------------------------------------
def load_model():
    from transformers import AutoVideoProcessor, AutoModel
    print(f"  Loading {MODEL_ID} (pretrained, no fine-tuning) ...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded ({n_params:.0f}M params)")
    return model, processor


def extract_embedding(model, processor, frames_16):
    """Extract a single embedding from 16 frames.

    Uses the last hidden state of the ViT backbone, mean-pooled over all
    patch tokens.  No classification head is involved.
    """
    inputs = processor(frames_16, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    # Base model returns last_hidden_state directly: (1, n_patches, hidden_dim)
    hidden = outputs.last_hidden_state
    # Mean-pool over patch tokens → (hidden_dim,)
    embedding = hidden.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding


def extract_windowed_embeddings(model, processor, video, stride=STRIDE):
    """Slide a window of WINDOW_SIZE frames across the video and extract
    embeddings.

    Returns:
        embeddings: np.ndarray (N, hidden_dim)
        window_centers: list[int] — center frame index for each window
    """
    T = len(video)
    embeddings = []
    window_centers = []

    n_windows = max(1, (T - WINDOW_SIZE) // stride + 1)
    print(f"  Extracting {n_windows} embeddings (window={WINDOW_SIZE}, stride={stride})...")

    for start in range(0, T - WINDOW_SIZE + 1, stride):
        frames_16 = video[start : start + WINDOW_SIZE]
        emb = extract_embedding(model, processor, frames_16)
        embeddings.append(emb)
        window_centers.append(start + WINDOW_SIZE // 2)

        done = len(embeddings)
        if done % 10 == 0 or done == n_windows:
            print(f"    [{done}/{n_windows}]")

    return np.array(embeddings), window_centers


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster_embeddings(embeddings, n_clusters=None):
    """K-means clustering. If n_clusters is None, use the number of actions."""
    if n_clusters is None:
        n_clusters = len(CLIPS)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_tsne_clusters(embeddings, cluster_labels, gt_labels, save_path):
    """t-SNE scatter plot coloured by (a) k-means cluster and (b) ground-truth."""
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Left: ground-truth action labels ---
    unique_actions = list(dict.fromkeys(gt_labels))  # preserve order
    for action in unique_actions:
        mask = np.array([l == action for l in gt_labels])
        ax1.scatter(coords[mask, 0], coords[mask, 1], label=action, alpha=0.7, s=50,
                    color=ACTION_COLORS[action], edgecolors="white", linewidths=0.5)
    ax1.set_title("Ground-Truth Action Labels")
    ax1.legend(fontsize=9, loc="best")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # --- Right: k-means clusters ---
    n_clusters = len(set(cluster_labels))
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    for k in range(n_clusters):
        mask = cluster_labels == k
        ax2.scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {k}",
                    alpha=0.7, s=50, color=cmap(k), edgecolors="white", linewidths=0.5)
    ax2.set_title("K-Means Clusters (unsupervised)")
    ax2.legend(fontsize=9, loc="best")
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.suptitle("V-JEPA 2 Pretrained (no fine-tuning) — Embedding Clusters",
                 fontsize=17, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_timeline(segment_labels, cluster_labels, window_centers, total_frames, save_path):
    """Timeline showing ground-truth actions vs discovered clusters."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 4), sharex=True)

    # Ground-truth timeline (per-frame)
    unique_actions = list(dict.fromkeys(segment_labels))
    gt_colors = [ACTION_COLORS[l] for l in segment_labels]
    for i in range(len(segment_labels) - 1):
        ax1.axvspan(i, i + 1, color=gt_colors[i], alpha=0.8)
    ax1.set_ylabel("Ground\nTruth", fontsize=11, rotation=0, labelpad=60, va="center")
    ax1.set_yticks([])
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ACTION_COLORS[a],
               markersize=10, label=a)
        for a in unique_actions
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=4)

    # Cluster timeline (per-window, drawn at window center)
    n_clusters = len(set(cluster_labels))
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    half_w = WINDOW_SIZE // 2
    for center, cl in zip(window_centers, cluster_labels):
        ax2.axvspan(center - half_w, center + half_w, color=cmap(cl), alpha=0.6)
    ax2.set_ylabel("K-Means\nCluster", fontsize=11, rotation=0, labelpad=60, va="center")
    ax2.set_yticks([])
    ax2.set_xlabel("Frame index", fontsize=11)
    ax2.set_xlim(0, total_frames)
    legend_elements2 = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(k),
               markersize=10, label=f"Cluster {k}")
        for k in range(n_clusters)
    ]
    ax2.legend(handles=legend_elements2, loc="upper right", fontsize=8, ncol=4)

    plt.suptitle("Pretrained V-JEPA 2 — Ground Truth vs Discovered Clusters",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_cluster_samples(video, cluster_labels, window_centers, save_path):
    """Show representative frames from each cluster."""
    n_clusters = len(set(cluster_labels))
    samples_per_cluster = 5
    cmap = plt.cm.get_cmap("tab10", n_clusters)

    fig, axes = plt.subplots(n_clusters, samples_per_cluster,
                             figsize=(3 * samples_per_cluster, 3 * n_clusters))
    if n_clusters == 1:
        axes = axes[np.newaxis, :]

    for k in range(n_clusters):
        indices = np.where(cluster_labels == k)[0]
        # Pick evenly spaced samples from this cluster
        sample_idx = indices[np.linspace(0, len(indices) - 1, samples_per_cluster, dtype=int)]

        for j, idx in enumerate(sample_idx):
            frame_idx = window_centers[idx]
            axes[k, j].imshow(video[frame_idx])
            axes[k, j].axis("off")
            if j == 0:
                axes[k, j].set_ylabel(f"Cluster {k}", fontsize=12, rotation=0,
                                       labelpad=55, va="center",
                                       color=cmap(k), fontweight="bold")

    plt.suptitle("Pretrained V-JEPA 2 — Representative Frames per Cluster",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  Demo 06 — V-JEPA 2 Cluster Analysis (pretrained, NO fine-tuning)")
    print("=" * 65)

    # 1. Build concatenated video
    print("\n[1/4] Downloading and concatenating clips...")
    video, segment_labels = build_concat_video()

    # 2. Load model & extract embeddings
    print("\n[2/4] Loading model and extracting embeddings...")
    model, processor = load_model()
    embeddings, window_centers = extract_windowed_embeddings(model, processor, video)

    # Ground-truth label for each window (majority vote from frames in window)
    gt_labels = []
    for center in window_centers:
        start = max(0, center - WINDOW_SIZE // 2)
        end = min(len(segment_labels), center + WINDOW_SIZE // 2)
        window_labels = segment_labels[start:end]
        # Majority label
        gt_labels.append(Counter(window_labels).most_common(1)[0][0])

    # Free model memory
    del model
    gc.collect()

    # 3. Cluster
    print("\n[3/4] Clustering embeddings...")
    cluster_labels, km = cluster_embeddings(embeddings)
    print(f"  {len(set(cluster_labels))} clusters found")

    # Save embeddings + metadata for later real-time use
    np.savez(
        OUTPUT_DIR / "06_embeddings.npz",
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        cluster_centers=km.cluster_centers_,
        window_centers=np.array(window_centers),
        gt_labels=np.array(gt_labels),
    )
    print(f"  Saved embeddings → {OUTPUT_DIR / '06_embeddings.npz'}")

    # 4. Visualise
    print("\n[4/4] Creating visualisations...")
    plot_tsne_clusters(
        embeddings, cluster_labels, gt_labels,
        OUTPUT_DIR / "06_tsne_clusters.png",
    )
    plot_timeline(
        segment_labels, cluster_labels, window_centers, len(video),
        OUTPUT_DIR / "06_timeline.png",
    )
    plot_cluster_samples(
        video, cluster_labels, window_centers,
        OUTPUT_DIR / "06_cluster_samples.png",
    )

    print(f"\n  Done! Check {OUTPUT_DIR}/06_*.png")
    print("  Embeddings saved to 06_embeddings.npz for comparison with Demo 05.")


if __name__ == "__main__":
    main()