"""
Demo 07 – Side-by-side MAE vs I-JEPA comparison
=================================================
Shows the fundamental difference between MAE (pixel reconstruction)
and JEPA (abstract representation prediction) using the same images.

MAE:  mask random patches → reconstruct exact pixels
JEPA: mask contiguous blocks → predict abstract representations

Uses STL-10 images (96×96 upscaled) for recognisable everyday categories.
"""

import gc
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from transformers import (
    AutoModel,
    AutoImageProcessor,
    ViTMAEForPreTraining,
)

matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
DEVICE = "cpu"

MAE_MODEL_ID = "facebook/vit-mae-large"
JEPA_MODEL_ID = "facebook/ijepa_vith14_1k"

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# Pick visually distinct classes for the comparison
SELECTED_CLASSES = [3, 5, 0, 8]  # cat, dog, airplane, ship
N_IMAGES = 4


def load_stl10_samples():
    """Load one clear image per selected class from STL-10 test set."""
    dataset = datasets.STL10(str(DATA_DIR), split="test", download=True)
    images = []
    for cls_idx in SELECTED_CLASSES:
        for i in range(len(dataset)):
            img, label = dataset[i]
            if label == cls_idx:
                images.append((img, STL10_CLASSES[label]))
                break
    return images


def load_mae_model():
    print("Loading MAE model...")
    processor = AutoImageProcessor.from_pretrained(MAE_MODEL_ID)
    model = ViTMAEForPreTraining.from_pretrained(MAE_MODEL_ID)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MAE: {n_params / 1e6:.0f}M parameters")
    return model, processor


def load_jepa_model():
    print("Loading I-JEPA model...")
    processor = AutoImageProcessor.from_pretrained(JEPA_MODEL_ID)
    model = AutoModel.from_pretrained(JEPA_MODEL_ID)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  I-JEPA: {n_params / 1e6:.0f}M parameters")
    return model, processor


# ---------- MAE reconstruction helpers ----------

def unpatchify(model, patchified, patch_size=16):
    """Convert patch-level predictions back to an image."""
    h = w = int(patchified.shape[1] ** 0.5)
    c = 3
    p = patch_size
    patchified = patchified.reshape(1, h, w, p, p, c)
    patchified = patchified.permute(0, 5, 1, 3, 2, 4)  # (1, C, H, p, W, p)
    img = patchified.reshape(1, c, h * p, w * p)
    return img


def mae_reconstruct(model, processor, pil_image):
    """Run MAE on an image and return (masked_image, reconstruction, mask)."""
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits: (1, num_patches, patch_size^2 * 3) = predicted patches
    # outputs.mask: (1, num_patches) → 1 = masked, 0 = visible
    logits = outputs.logits
    mask = outputs.mask  # (1, num_patches)

    # Original image as tensor
    pixel_values = inputs["pixel_values"]  # (1, 3, H, W)

    # Patchify original
    patch_size = model.config.patch_size
    img_size = pixel_values.shape[-1]
    n_patches_side = img_size // patch_size
    n_patches = n_patches_side ** 2

    # Reconstruct from MAE predictions
    recon = unpatchify(model, logits, patch_size)

    # Create masked image (grey out masked patches)
    masked_img = pixel_values.clone()
    mask_2d = mask.reshape(n_patches_side, n_patches_side)  # (H_p, W_p)
    for i in range(n_patches_side):
        for j in range(n_patches_side):
            if mask_2d[i, j] == 1:
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                masked_img[0, :, y0:y1, x0:x1] = 0.5  # grey

    # Composite: visible patches from original + reconstructed masked patches
    composite = pixel_values.clone()
    for i in range(n_patches_side):
        for j in range(n_patches_side):
            if mask_2d[i, j] == 1:
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                composite[0, :, y0:y1, x0:x1] = recon[0, :, y0:y1, x0:x1]

    def to_numpy(t):
        t = t.squeeze(0).permute(1, 2, 0).numpy()
        t = np.clip(t * processor.image_std + processor.image_mean, 0, 1)
        return t

    def to_numpy_raw(t):
        """For reconstruction that's already in pixel space."""
        t = t.squeeze(0).permute(1, 2, 0).numpy()
        return np.clip(t, 0, 1)

    original_np = to_numpy(pixel_values)
    masked_np = to_numpy(masked_img)
    composite_np = to_numpy(composite)
    mask_np = mask_2d.numpy()

    return original_np, masked_np, composite_np, mask_np


# ---------- JEPA masking & feature helpers ----------

def sample_block(grid_h, grid_w, min_scale=0.10, max_scale=0.18):
    """Sample a random rectangular block in a patch grid."""
    n_total = grid_h * grid_w
    target_size = int(n_total * random.uniform(min_scale, max_scale))
    aspect = random.uniform(0.75, 1.5)
    h = max(1, int(round((target_size * aspect) ** 0.5)))
    w = max(1, int(round(target_size / h)))
    h = min(h, grid_h)
    w = min(w, grid_w)
    top = random.randint(0, grid_h - h)
    left = random.randint(0, grid_w - w)
    indices = set()
    for r in range(top, top + h):
        for c in range(left, left + w):
            indices.add(r * grid_w + c)
    return indices, (top, left, h, w)


def generate_jepa_masks(grid_h=16, grid_w=16, n_targets=4):
    """Generate I-JEPA-style masks: multiple target blocks + one context block."""
    n_total = grid_h * grid_w
    target_patches = set()
    for _ in range(n_targets):
        for _ in range(50):
            block, _ = sample_block(grid_h, grid_w)
            overlap = len(block & target_patches) / max(len(block), 1)
            if overlap < 0.3:
                target_patches |= block
                break

    remaining = set(range(n_total)) - target_patches
    context, _ = sample_block(grid_h, grid_w, min_scale=0.15, max_scale=0.30)
    context = context & remaining
    return context, target_patches


def jepa_visualise(pil_image, processor, model):
    """Create JEPA-style masked image and attention/feature heatmap."""
    img_np = np.array(pil_image.resize((224, 224), Image.LANCZOS)) / 255.0
    grid_h = grid_w = 16
    patch_size = 14  # 224 / 16

    context, targets = generate_jepa_masks(grid_h, grid_w)

    # --- Masked image ---
    masked_img = img_np.copy()
    overlay = np.zeros((*img_np.shape[:2], 4))
    for idx in range(grid_h * grid_w):
        r, c = divmod(idx, grid_w)
        y0, y1 = r * patch_size, (r + 1) * patch_size
        x0, x1 = c * patch_size, (c + 1) * patch_size
        if idx in context:
            overlay[y0:y1, x0:x1] = [0.13, 0.59, 0.95, 0.25]  # blue
        elif idx in targets:
            overlay[y0:y1, x0:x1] = [1.0, 0.34, 0.13, 0.35]  # red
        else:
            masked_img[y0:y1, x0:x1] = 0.6  # grey

    masked_vis = np.clip(masked_img + overlay[:, :, :3] * overlay[:, :, 3:], 0, 1)

    # --- Feature-norm heatmap from I-JEPA ---
    # Shows which patches carry the most semantic information
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Per-patch feature norms (I-JEPA has no CLS token)
    hidden = outputs.last_hidden_state[0]  # (num_patches, D)
    patch_norms = hidden.norm(dim=-1)  # (num_patches,)
    n_patches = patch_norms.shape[0]
    side = int(n_patches ** 0.5)
    attn_map = patch_norms.reshape(side, side).numpy()
    # Resize to image size
    attn_map = np.array(
        Image.fromarray(attn_map).resize((224, 224), Image.LANCZOS)
    )
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return img_np, masked_vis, attn_map


# ---------- Main comparison figure ----------

def plot_comparison(mae_model, mae_proc, jepa_model, jepa_proc, samples):
    """Create the main side-by-side comparison figure.

    4 columns: Original | MAE: 75% hidden | MAE: reconstruction | JEPA: semantic focus
    Shows what was masked so you can see which pixels MAE reconstructed.
    """
    n = len(samples)

    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n + 1))
    fig.patch.set_facecolor("#FAFAFA")

    col_titles = [
        "Original",
        "MAE: 75% hidden",
        "MAE: pixel reconstruction",
        "JEPA: semantic focus",
    ]
    col_colors = ["#333333", "#D32F2F", "#D32F2F", "#1565C0"]

    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col].set_title(title, fontsize=14, fontweight="bold",
                               pad=12, color=color)

    for i, (pil_img, class_name) in enumerate(samples):
        print(f"  Processing {class_name}...")
        pil_224 = pil_img.resize((224, 224), Image.LANCZOS)

        # MAE reconstruction
        orig_np, masked_np, composite_np, mask_np = mae_reconstruct(
            mae_model, mae_proc, pil_224
        )
        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_ylabel(class_name.upper(), fontsize=14, fontweight="bold",
                               rotation=0, labelpad=60, va="center")
        axes[i, 1].imshow(masked_np)
        axes[i, 2].imshow(composite_np)

        # JEPA feature activation
        img_np, masked_vis, attn_map = jepa_visualise(pil_224, jepa_proc, jepa_model)
        axes[i, 3].imshow(img_np)
        axes[i, 3].imshow(attn_map, cmap="inferno", alpha=0.6)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("What does each model output?", fontsize=18, fontweight="bold", y=1.0)

    fig.text(0.5, 0.01,
             "MAE must reconstruct every hidden pixel (blurry)  |  "
             "JEPA learns which parts matter (semantic heatmap)",
             ha="center", fontsize=12, color="#555")

    plt.tight_layout(rect=[0.06, 0.03, 1.0, 0.96])
    save_path = OUTPUT_DIR / "07_mae_vs_jepa.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------- Zoomed-in patch comparison ----------

def plot_patch_zoom(mae_model, mae_proc, jepa_model, jepa_proc, samples):
    """Show a zoomed patch-level comparison for one image.

    4 columns: Original | MAE: masked | MAE: reconstruction | JEPA: semantic focus
    """
    pil_img, class_name = samples[0]
    pil_224 = pil_img.resize((224, 224), Image.LANCZOS)

    orig_np, masked_np, composite_np, mask_np = mae_reconstruct(
        mae_model, mae_proc, pil_224
    )
    img_np, masked_vis, attn_map = jepa_visualise(pil_224, jepa_proc, jepa_model)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#FAFAFA")

    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original ({class_name})", fontsize=14, fontweight="bold")

    axes[1].imshow(masked_np)
    axes[1].set_title("MAE: 75% hidden", fontsize=14,
                      fontweight="bold", color="#D32F2F")

    axes[2].imshow(composite_np)
    axes[2].set_title("MAE: pixel reconstruction", fontsize=14,
                      fontweight="bold", color="#D32F2F")

    axes[3].imshow(img_np)
    axes[3].imshow(attn_map, cmap="inferno", alpha=0.6)
    axes[3].set_title("JEPA: semantic focus", fontsize=14,
                      fontweight="bold", color="#1565C0")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_path = OUTPUT_DIR / "07_patch_zoom.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Demo 07: MAE vs I-JEPA — Side-by-Side Comparison")
    print("=" * 60)

    print("\n[1/4] Loading STL-10 samples...")
    samples = load_stl10_samples()
    print(f"  Loaded {len(samples)} images: {[s[1] for s in samples]}")

    print("\n[2/4] Loading models...")
    mae_model, mae_proc = load_mae_model()
    jepa_model, jepa_proc = load_jepa_model()

    print("\n[3/4] Creating main comparison figure...")
    plot_comparison(mae_model, mae_proc, jepa_model, jepa_proc, samples)

    print("\n[4/4] Creating zoomed patch comparison...")
    plot_patch_zoom(mae_model, mae_proc, jepa_model, jepa_proc, samples)

    gc.collect()
    print("\n✓ Done! Check outputs/07_*.png")


if __name__ == "__main__":
    main()