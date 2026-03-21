"""
Demo 01 — I-JEPA Representations: Clustering & Similarity
==========================================================

Loads the pretrained I-JEPA ViT-H/14 from HuggingFace, extracts features
from CIFAR-10 images, and visualizes how the model organizes visual concepts
in its representation space.

Run:
    python demos/01_ijepa_representations.py
"""

import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.manifold import TSNE

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"
MODEL_ID = "facebook/ijepa_vith14_1k"
N_SAMPLES = 200  # images for t-SNE (keep moderate for CPU speed)
BATCH_SIZE = 8

# 10 visually distinct flower classes (indices into Flowers102)
SELECTED_CLASSES = {
    41: "daffodil",
    53: "sunflower",
    74: "rose",
    73: "water lily",
    83: "hibiscus",
    87: "magnolia",
    76: "morning glory",
    43: "poinsettia",
    25: "corn poppy",
    47: "buttercup",
}
CLASS_NAMES = [SELECTED_CLASSES[k] for k in sorted(SELECTED_CLASSES.keys())]
N_CLASSES = len(CLASS_NAMES)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def load_model():
    from transformers import AutoModel, AutoImageProcessor
    print(f"  Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    model.to(DEVICE)
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    return model, processor


def extract_features(model, processor, dataset, indices):
    """Extract CLS-pooled features for given dataset indices."""
    features = []
    labels = []
    raw_images = []

    for i in tqdm(range(0, len(indices), BATCH_SIZE), desc="  Extracting", ncols=70):
        batch_idx = indices[i : i + BATCH_SIZE]
        imgs = [dataset[j][0] for j in batch_idx]
        lbls = [dataset[j][1] for j in batch_idx]
        raw_images.extend(imgs)
        labels.extend(lbls)

        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # Global average pool over patch tokens
            feat = outputs.last_hidden_state.mean(dim=1)  # (B, D)
            features.append(feat.cpu())

    features = torch.cat(features).numpy()
    labels = np.array(labels)
    return features, labels, raw_images


# ---------------------------------------------------------------------------
# Viz 1: t-SNE clustering
# ---------------------------------------------------------------------------
def plot_tsne(features, labels, save_path):
    print("  Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    coords = tsne.fit_transform(features)

    colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 10))
    for c in range(N_CLASSES):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[c]],
                   label=CLASS_NAMES[c], s=30, alpha=0.7, edgecolors="white", linewidths=0.3)
    ax.legend(fontsize=11, markerscale=2, loc="best", frameon=True)
    ax.set_title("I-JEPA Representation Space (t-SNE)", fontsize=18)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Viz 2: Image similarity retrieval
# ---------------------------------------------------------------------------
def plot_similarity_retrieval(features, labels, images, save_path, n_queries=5, top_k=6):
    """For each query image, show top-K most similar images by cosine similarity."""
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norms
    sim_matrix = normed @ normed.T

    # Pick one query per class
    query_indices = []
    seen_classes = set()
    for i, l in enumerate(labels):
        if l not in seen_classes and len(query_indices) < n_queries:
            query_indices.append(i)
            seen_classes.add(l)

    fig, axes = plt.subplots(n_queries, top_k + 1, figsize=(3 * (top_k + 1), 3 * n_queries))
    for row, qi in enumerate(query_indices):
        # Query image
        ax = axes[row, 0]
        img = images[qi]
        if hasattr(img, "numpy"):
            img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Query\n{CLASS_NAMES[labels[qi]]}", fontsize=11, fontweight="bold",
                     color="#1565C0")
        ax.axis("off")
        # Blue border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#1565C0")
            spine.set_linewidth(3)

        # Top-K similar (exclude self)
        sims = sim_matrix[qi].copy()
        sims[qi] = -1  # exclude self
        top_indices = np.argsort(sims)[::-1][:top_k]

        for col, ti in enumerate(top_indices):
            ax = axes[row, col + 1]
            timg = images[ti]
            if hasattr(timg, "numpy"):
                timg = timg.permute(1, 2, 0).numpy()
            ax.imshow(timg)
            match = labels[ti] == labels[qi]
            color = "#4CAF50" if match else "#F44336"
            ax.set_title(f"{CLASS_NAMES[labels[ti]]}\n{sims[ti]:.3f}", fontsize=10, color=color)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)

    plt.suptitle("Image Similarity Retrieval with I-JEPA", fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Viz 3: Cosine similarity heatmap
# ---------------------------------------------------------------------------
def plot_similarity_heatmap(features, labels, save_path, n_per_class=5):
    """Pairwise cosine similarity heatmap, sorted by class."""
    # Pick n images per class, in class order
    indices = []
    for c in range(N_CLASSES):
        class_idx = np.where(labels == c)[0][:n_per_class]
        indices.extend(class_idx)
    indices = np.array(indices)

    sub_features = features[indices]
    sub_labels = labels[indices]

    norms = np.linalg.norm(sub_features, axis=1, keepdims=True) + 1e-8
    normed = sub_features / norms
    sim = normed @ normed.T

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)

    # Class boundary lines
    for i in range(1, N_CLASSES):
        pos = i * n_per_class - 0.5
        ax.axhline(pos, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(pos, color="black", linewidth=0.5, alpha=0.5)

    # Class labels
    tick_pos = [i * n_per_class + n_per_class / 2 - 0.5 for i in range(N_CLASSES)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)

    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    ax.set_title("Pairwise Cosine Similarity of I-JEPA Features", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  Demo 01 — I-JEPA Representations")
    print("=" * 55)

    model, processor = load_model()

    # Load Flowers102 — high-res images, pick 10 distinct classes
    full_dataset = datasets.Flowers102(root="./data", split="test", download=True)
    class_ids = sorted(SELECTED_CLASSES.keys())
    class_id_to_idx = {cid: i for i, cid in enumerate(class_ids)}

    # Collect indices for selected classes, up to N_SAMPLES total
    rng = np.random.RandomState(42)
    per_class = N_SAMPLES // len(class_ids)
    indices = []
    for cid in class_ids:
        class_indices = [i for i in range(len(full_dataset)) if full_dataset[i][1] == cid]
        chosen = rng.choice(class_indices, size=min(per_class, len(class_indices)), replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    rng.shuffle(indices)

    print(f"\n  Extracting features for {len(indices)} images across {len(class_ids)} flower types...")
    features, labels_raw, images = extract_features(model, processor, full_dataset, indices)
    # Remap labels to 0..9
    labels = np.array([class_id_to_idx[l] for l in labels_raw])
    print(f"  Feature shape: {features.shape}")

    # Free model memory
    del model
    gc.collect()

    print("\n  [1/3] t-SNE visualization...")
    plot_tsne(features, labels, OUTPUT_DIR / "01_tsne.png")

    print("  [2/3] Similarity retrieval...")
    plot_similarity_retrieval(features, labels, images, OUTPUT_DIR / "01_similarity.png")

    print("  [3/3] Similarity heatmap...")
    plot_similarity_heatmap(features, labels, OUTPUT_DIR / "01_heatmap.png")

    print(f"\n  Done! Check {OUTPUT_DIR}/01_*.png")


if __name__ == "__main__":
    main()
