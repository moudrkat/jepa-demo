"""
Demo 08 – Animated t-SNE: watch clusters form
==============================================
Extracts I-JEPA features from STL-10 images (10 everyday categories),
then animates t-SNE going from random noise to organised semantic clusters.

Outputs:
  08_tsne_animation.gif  — animated GIF (podcast-ready)
  08_tsne_final.png      — high-res final frame with class labels
  08_tsne_thumbnails.png — final t-SNE with actual image thumbnails
"""

import gc
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
DEVICE = "cpu"
MODEL_ID = "facebook/ijepa_vith14_1k"

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# Visually distinct colors for 10 classes
CLASS_COLORS = [
    "#E53935", "#8E24AA", "#1E88E5", "#FF8F00", "#43A047",
    "#F4511E", "#6D4C41", "#00ACC1", "#3949AB", "#7CB342",
]

N_PER_CLASS = 30  # 300 total
BATCH_SIZE = 8
N_FRAMES = 40  # animation frames
N_HOLD_FRAMES = 10  # hold final state ~3s
FPS = 3  # ~333ms per frame, comfortable pace


def load_model():
    print("Loading I-JEPA model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.0f}M parameters")
    return model, processor


def load_stl10_samples():
    """Load N_PER_CLASS images per class from STL-10 test set."""
    dataset = datasets.STL10(str(DATA_DIR), split="test", download=True)

    class_counts = {i: 0 for i in range(10)}
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if class_counts[label] < N_PER_CLASS:
            indices.append(i)
            class_counts[label] += 1
        if all(v >= N_PER_CLASS for v in class_counts.values()):
            break

    return dataset, indices


def extract_features(model, processor, dataset, indices):
    """Extract I-JEPA features via global average pooling."""
    features = []
    labels = []
    raw_images = []

    for start in tqdm(range(0, len(indices), BATCH_SIZE), desc="Extracting features"):
        batch_idx = indices[start : start + BATCH_SIZE]
        imgs = []
        for idx in batch_idx:
            img, label = dataset[idx]
            imgs.append(img)
            labels.append(label)
            raw_images.append(img)

        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)  # (B, D)
        features.append(pooled.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels, raw_images


def compute_tsne(features):
    """Compute final t-SNE embedding."""
    print("Computing t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        init="random",
    )
    embedding = tsne.fit_transform(features)
    return embedding


def ease_in_out(t):
    """Smooth ease-in-out interpolation."""
    return t * t * (3 - 2 * t)


def create_animation(embedding, labels, raw_images):
    """Create animated GIF: random → organized clusters."""
    print("Creating animation...")
    n = len(embedding)

    # Normalise final embedding to [-1, 1] range
    emb_min = embedding.min(axis=0)
    emb_max = embedding.max(axis=0)
    emb_range = emb_max - emb_min
    final = 2 * (embedding - emb_min) / emb_range - 1

    # Random starting positions (normal distribution, similar scale)
    rng = np.random.RandomState(42)
    start = rng.randn(n, 2) * 0.8

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    # Pre-compute colors
    colors = [CLASS_COLORS[l] for l in labels]

    scatter = ax.scatter(
        start[:, 0], start[:, 1],
        c=colors, s=30, alpha=0.7, edgecolors="white", linewidths=0.3,
    )

    title = ax.set_title("", fontsize=18, fontweight="bold", color="white", pad=15)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[i],
                    markersize=8, label=STL10_CLASSES[i], linestyle="None")
        for i in range(10)
    ]
    legend = ax.legend(
        handles=legend_handles, loc="upper left", fontsize=10,
        facecolor="#2A2A4A", edgecolor="#555", labelcolor="white",
        ncol=2, framealpha=0.9,
    )

    # Progress bar at bottom
    progress_bg = plt.Rectangle((0.1, 0.02), 0.8, 0.015,
                                 transform=fig.transFigure, facecolor="#333355",
                                 edgecolor="none", zorder=10)
    progress_bar = plt.Rectangle((0.1, 0.02), 0.0, 0.015,
                                  transform=fig.transFigure, facecolor="#4FC3F7",
                                  edgecolor="none", zorder=11)
    fig.patches.extend([progress_bg, progress_bar])

    total_frames = N_FRAMES + N_HOLD_FRAMES

    def update(frame):
        # Clamp to last real frame during hold period
        f = min(frame, N_FRAMES - 1)
        t = f / (N_FRAMES - 1)
        t_smooth = ease_in_out(t)
        pos = (1 - t_smooth) * start + t_smooth * final
        scatter.set_offsets(pos)

        # Increase alpha and size as clusters form
        alpha = 0.4 + 0.5 * t_smooth
        scatter.set_alpha(alpha)
        scatter.set_sizes([20 + 30 * t_smooth] * n)

        title.set_text("")

        # Progress bar
        progress_bar.set_width(0.8 * t)

        return scatter, title

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // FPS, blit=False,
    )

    save_path = OUTPUT_DIR / "08_tsne_animation.gif"
    anim.save(str(save_path), writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_final_frame(embedding, labels):
    """High-res static final frame."""
    emb_min = embedding.min(axis=0)
    emb_max = embedding.max(axis=0)
    final = 2 * (embedding - emb_min) / (emb_max - emb_min) - 1

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    colors = [CLASS_COLORS[l] for l in labels]
    ax.scatter(final[:, 0], final[:, 1], c=colors, s=50, alpha=0.85,
               edgecolors="white", linewidths=0.4)

    # Add class labels offset above each cluster centroid
    for cls_idx in range(10):
        mask = labels == cls_idx
        cx, cy = final[mask].mean(axis=0)
        # Place label above cluster, connected by a short line
        ax.annotate(
            STL10_CLASSES[cls_idx].upper(),
            xy=(cx, cy), xytext=(cx, cy + 0.15),
            fontsize=11, fontweight="bold", color="white",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CLASS_COLORS[cls_idx],
                      alpha=0.85, edgecolor="white", linewidth=0.5),
            arrowprops=dict(arrowstyle="-", color=CLASS_COLORS[cls_idx],
                            lw=1.5, alpha=0.6),
        )

    ax.set_title(
        "I-JEPA learned representations on STL-10\n"
        "(self-supervised — no labels used during training)",
        fontsize=16, fontweight="bold", color="white", pad=15,
    )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.5)  # extra room for labels above
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[i],
                    markersize=10, label=STL10_CLASSES[i], linestyle="None")
        for i in range(10)
    ]
    ax.legend(
        handles=legend_handles, loc="upper left", fontsize=11,
        facecolor="#2A2A4A", edgecolor="#555", labelcolor="white",
        ncol=2, framealpha=0.9,
    )

    save_path = OUTPUT_DIR / "08_tsne_final.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_thumbnail_tsne(embedding, labels, raw_images):
    """t-SNE with actual image thumbnails — the wow visual."""
    emb_min = embedding.min(axis=0)
    emb_max = embedding.max(axis=0)
    final = 2 * (embedding - emb_min) / (emb_max - emb_min) - 1

    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    # Plot a subset as thumbnails (too many = clutter)
    step = 3  # every 3rd image
    for i in range(0, len(raw_images), step):
        img = raw_images[i]
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        thumb = Image.fromarray(img).resize((28, 28), Image.LANCZOS)

        im = OffsetImage(thumb, zoom=1.0)
        ab = AnnotationBbox(
            im, (final[i, 0], final[i, 1]),
            frameon=True,
            bboxprops=dict(edgecolor=CLASS_COLORS[labels[i]], linewidth=1.5,
                           facecolor="#1A1A2E", alpha=0.95),
            pad=0.1,
        )
        ax.add_artist(ab)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                    markerfacecolor=CLASS_COLORS[i], markersize=10,
                    label=STL10_CLASSES[i], linestyle="None")
        for i in range(10)
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=11,
              facecolor="#2A2A4A", edgecolor="#555", labelcolor="white",
              ncol=2, framealpha=0.9)

    ax.set_title(
        "I-JEPA clusters STL-10 images by semantic meaning\n"
        "(trained on ImageNet, never saw these categories)",
        fontsize=18, fontweight="bold", color="white", pad=15,
    )
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.6, 1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_path = OUTPUT_DIR / "08_tsne_thumbnails.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Demo 08: Animated t-SNE — Watch Clusters Form")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    model, processor = load_model()

    print("\n[2/5] Loading STL-10 samples...")
    dataset, indices = load_stl10_samples()
    print(f"  {len(indices)} images across {len(STL10_CLASSES)} classes")

    print("\n[3/5] Extracting features...")
    features, labels, raw_images = extract_features(model, processor, dataset, indices)
    print(f"  Feature shape: {features.shape}")

    # Free model memory
    del model
    gc.collect()

    print("\n[4/5] Computing t-SNE...")
    embedding = compute_tsne(features)

    print("\n[5/5] Creating visualizations...")
    create_animation(embedding, labels, raw_images)
    plot_final_frame(embedding, labels)
    plot_thumbnail_tsne(embedding, labels, raw_images)

    # Save embeddings for reuse
    np.savez(
        OUTPUT_DIR / "08_embeddings.npz",
        features=features, embedding=embedding,
        labels=labels, classes=STL10_CLASSES,
    )
    print(f"  Saved: {OUTPUT_DIR / '08_embeddings.npz'}")

    print("\n✓ Done! Check outputs/08_*.gif and outputs/08_*.png")


if __name__ == "__main__":
    main()