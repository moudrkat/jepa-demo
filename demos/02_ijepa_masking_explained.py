"""
Demo 02 — How I-JEPA Learns: Multi-Block Masking Explained
==========================================================

No model needed — purely visual.  Shows *how* I-JEPA trains by
illustrating the multi-block masking strategy on sample images.

Run:
    python demos/02_ijepa_masking_explained.py
"""

import random
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

# Colors
C_CONTEXT = "#2196F3"   # blue
C_TARGET = "#FF5722"    # red-orange
C_UNUSED = "#BDBDBD"    # grey
C_BG = "#F5F5F5"


# ---------------------------------------------------------------------------
# Multi-block masking (following the I-JEPA paper)
# ---------------------------------------------------------------------------
def sample_block(grid_h, grid_w, min_scale, max_scale, min_aspect, max_aspect):
    """Sample a rectangular block within a patch grid."""
    scale = random.uniform(min_scale, max_scale)
    aspect = random.uniform(min_aspect, max_aspect)
    num_patches = grid_h * grid_w
    block_area = int(scale * num_patches)
    block_h = max(1, int(round(np.sqrt(block_area * aspect))))
    block_w = max(1, int(round(np.sqrt(block_area / aspect))))
    block_h = min(block_h, grid_h)
    block_w = min(block_w, grid_w)
    top = random.randint(0, grid_h - block_h)
    left = random.randint(0, grid_w - block_w)
    indices = set()
    for r in range(top, top + block_h):
        for c in range(left, left + block_w):
            indices.add(r * grid_w + c)
    return indices, (top, left, block_h, block_w)


def generate_ijepa_masks(grid_h=16, grid_w=16, n_targets=4):
    """
    Generate I-JEPA-style masks:
      - 4 target blocks (large, ~15% each)
      - 1 context block (from remaining patches)
    """
    all_patches = set(range(grid_h * grid_w))
    target_patches = set()
    target_blocks = []

    # Sample target blocks
    for _ in range(n_targets):
        for _attempt in range(50):
            indices, bbox = sample_block(
                grid_h, grid_w,
                min_scale=0.10, max_scale=0.18,
                min_aspect=0.75, max_aspect=1.5,
            )
            if len(indices & target_patches) < len(indices) * 0.3:
                target_patches |= indices
                target_blocks.append(bbox)
                break

    # Context: a single large block from the remaining patches
    remaining = all_patches - target_patches
    context_patches = set()
    for _attempt in range(50):
        indices, bbox = sample_block(
            grid_h, grid_w,
            min_scale=0.15, max_scale=0.30,
            min_aspect=0.75, max_aspect=1.5,
        )
        overlap = indices & target_patches
        if len(overlap) < len(indices) * 0.2:
            context_patches = indices - target_patches
            break

    if not context_patches:
        # Fallback: random subset of remaining
        remaining_list = list(remaining)
        random.shuffle(remaining_list)
        context_patches = set(remaining_list[: len(remaining_list) // 3])

    return context_patches, target_patches


# ---------------------------------------------------------------------------
# Visualization 1: Masking strategy on images
# ---------------------------------------------------------------------------
def load_sample_images():
    """Load sample images from assets/ directory, resized to 224x224."""
    asset_dir = Path("assets")
    image_files = sorted(asset_dir.glob("*.jpg")) + sorted(asset_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError("No images found in assets/. Add some .jpg or .png files.")
    images = []
    for p in image_files[:4]:
        img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
        images.append(np.array(img) / 255.0)
    # Pad to 4 if fewer
    while len(images) < 4:
        images.append(images[-1])
    return images


def plot_masking_on_images():
    """Show masking overlays on sample images from assets/."""
    images = load_sample_images()

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    grid_size = 16  # 224 / 14 = 16 patches per side
    patch_px = 224 // grid_size

    for col in range(4):
        img = images[col]

        # Row 0: original
        axes[0, col].imshow(img)
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=14, fontweight="bold")

        # Generate mask
        ctx, tgt = generate_ijepa_masks(grid_size, grid_size)

        # Row 1: mask overlay
        axes[1, col].imshow(img)
        overlay = np.zeros((224, 224, 4))
        for p in range(grid_size * grid_size):
            r, c = divmod(p, grid_size)
            y0, x0 = r * patch_px, c * patch_px
            if p in ctx:
                overlay[y0:y0 + patch_px, x0:x0 + patch_px] = [0.13, 0.59, 0.95, 0.5]
            elif p in tgt:
                overlay[y0:y0 + patch_px, x0:x0 + patch_px] = [1.0, 0.34, 0.13, 0.6]
            else:
                overlay[y0:y0 + patch_px, x0:x0 + patch_px] = [0.74, 0.74, 0.74, 0.3]
        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel("Masked", fontsize=14, fontweight="bold")

        # Row 2: what the encoder sees (only context)
        masked_img = np.ones_like(img) * 0.85
        for p in ctx:
            r, c = divmod(p, grid_size)
            y0, x0 = r * patch_px, c * patch_px
            masked_img[y0:y0 + patch_px, x0:x0 + patch_px] = img[y0:y0 + patch_px, x0:x0 + patch_px]
        axes[2, col].imshow(masked_img)
        axes[2, col].axis("off")
        if col == 0:
            axes[2, col].set_ylabel("Encoder sees", fontsize=14, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_CONTEXT, alpha=0.6, label="Context (visible to encoder)"),
        mpatches.Patch(facecolor=C_TARGET, alpha=0.7, label="Target (model must predict)"),
        mpatches.Patch(facecolor=C_UNUSED, alpha=0.4, label="Unused"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=13,
               frameon=True, fancybox=True)

    plt.suptitle("I-JEPA Multi-Block Masking Strategy", fontsize=20, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = OUTPUT_DIR / "02_masking_on_images.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Visualization 2: Multiple masks for same image
# ---------------------------------------------------------------------------
def plot_mask_variations():
    """Show 6 different random masks on the same image."""
    images = load_sample_images()
    img = images[0]

    grid_size = 16
    patch_px = 224 // grid_size

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
    for i, ax in enumerate(axes.flat):
        ctx, tgt = generate_ijepa_masks(grid_size, grid_size)
        ax.imshow(img)
        overlay = np.zeros((224, 224, 4))
        for p in range(grid_size * grid_size):
            r, c = divmod(p, grid_size)
            y0, x0 = r * patch_px, c * patch_px
            if p in ctx:
                overlay[y0:y0 + patch_px, x0:x0 + patch_px] = [0.13, 0.59, 0.95, 0.5]
            elif p in tgt:
                overlay[y0:y0 + patch_px, x0:x0 + patch_px] = [1.0, 0.34, 0.13, 0.6]
            else:
                overlay[y0:y0 + patch_px, x0:x0 + patch_px] = [0.74, 0.74, 0.74, 0.3]
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"Sample {i + 1}", fontsize=13)

    plt.suptitle(
        "Same Image, Different Masks — Each Training Step Sees a New View",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUTPUT_DIR / "02_mask_variations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Visualization 3: Architecture diagram
# ---------------------------------------------------------------------------
def plot_architecture_diagram():
    """Clean box-and-arrow diagram of the I-JEPA architecture."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def add_box(x, y, w, h, text, color, fontsize=12):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white" if color != "#FFF9C4" else "black")

    def add_arrow(x1, y1, x2, y2, text="", color="black"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
        )
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, text, fontsize=10, color=color, fontstyle="italic")

    # Image
    add_box(0.5, 3, 2, 1.8, "Input\nImage", "#78909C")

    # Masking
    add_box(3.5, 4.3, 2.2, 1.2, "Context\nPatches", C_CONTEXT, fontsize=11)
    add_box(3.5, 2.3, 2.2, 1.2, "Target\nPatches", C_TARGET, fontsize=11)

    # Encoders
    add_box(7, 4.3, 2.2, 1.2, "Context\nEncoder (ViT)", "#1565C0")
    add_box(7, 2.3, 2.2, 1.2, "Target\nEncoder (EMA)", "#BF360C")

    # Predictor
    add_box(10.5, 4.3, 2.2, 1.2, "Predictor\n(small ViT)", "#6A1B9A")

    # Loss
    add_box(10.5, 2.3, 2.2, 1.2, "L2 Loss", "#FFF9C4", fontsize=13)

    # Arrows
    add_arrow(2.5, 4.0, 3.5, 4.9)    # image -> context
    add_arrow(2.5, 3.8, 3.5, 2.9)    # image -> target
    add_arrow(5.7, 4.9, 7.0, 4.9)    # context -> encoder
    add_arrow(5.7, 2.9, 7.0, 2.9)    # target -> encoder
    add_arrow(9.2, 4.9, 10.5, 4.9)   # ctx encoder -> predictor
    add_arrow(11.6, 4.3, 11.6, 3.5)  # predictor -> loss
    add_arrow(9.2, 2.9, 10.5, 2.9)   # tgt encoder -> loss

    # Annotations
    ax.text(8.1, 1.5, "stop gradient", fontsize=10, fontstyle="italic", color="#BF360C",
            ha="center")
    ax.annotate(
        "", xy=(8.1, 2.3), xytext=(8.1, 1.8),
        arrowprops=dict(arrowstyle="-", color="#BF360C", lw=1.5, linestyle="--"),
    )

    # EMA arrow: Context Encoder -> Target Encoder (right side, connecting the two)
    ax.annotate(
        "", xy=(9.0, 3.5), xytext=(9.0, 4.3),
        arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=2, linestyle="--"),
    )
    ax.text(9.6, 3.9, "EMA\nupdate", fontsize=10, fontstyle="italic", color="#1565C0",
            ha="left", va="center")

    # Title
    ax.text(7, 7.4, "I-JEPA Architecture", fontsize=20, fontweight="bold",
            ha="center", va="center")
    ax.text(7, 6.8, "Predict abstract representations, not pixels",
            fontsize=13, ha="center", va="center", color="#616161", fontstyle="italic")

    path = OUTPUT_DIR / "02_architecture.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Visualization 4: JEPA vs MAE vs Contrastive
# ---------------------------------------------------------------------------
def plot_method_comparison():
    """Side-by-side comparison of self-supervised learning approaches."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    methods = [
        {
            "title": "Contrastive\n(SimCLR, DINO)",
            "color": "#4CAF50",
            "pros": "Good features",
            "cons": "Needs augmentations\n& negative pairs",
            "how": "Pull views of same\nimage together,\npush others apart",
        },
        {
            "title": "Generative\n(MAE)",
            "color": "#FF9800",
            "pros": "Simple objective",
            "cons": "Wastes capacity on\nunpredictable pixels",
            "how": "Mask patches,\nreconstruct\nraw pixels",
        },
        {
            "title": "JEPA\n(This work)",
            "color": "#2196F3",
            "pros": "Best of both worlds",
            "cons": "",
            "how": "Mask patches,\npredict abstract\nrepresentations",
        },
    ]

    for i, m in enumerate(methods):
        ax = axes[i]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Title box
        box = FancyBboxPatch(
            (0.5, 7.5), 9, 2,
            boxstyle="round,pad=0.2",
            facecolor=m["color"], edgecolor="black", linewidth=1.5, alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(5, 8.5, m["title"], ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

        # How it works
        ax.text(5, 6.5, "How:", ha="center", fontsize=11, fontweight="bold")
        ax.text(5, 5.3, m["how"], ha="center", fontsize=11, color="#424242",
                linespacing=1.4)

        # Pros
        if m["pros"]:
            ax.text(5, 3.2, f"+ {m['pros']}", ha="center", fontsize=11,
                    color="#2E7D32", fontweight="bold")

        # Cons
        if m["cons"]:
            ax.text(5, 2.0, f"- {m['cons']}", ha="center", fontsize=10,
                    color="#C62828", linespacing=1.3)

        # Highlight JEPA
        if i == 2:
            highlight = FancyBboxPatch(
                (0.2, 0.5), 9.6, 9.2,
                boxstyle="round,pad=0.1",
                facecolor="none", edgecolor=m["color"], linewidth=3, linestyle="--",
            )
            ax.add_patch(highlight)

    plt.suptitle("Self-Supervised Learning: Three Paradigms", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / "02_method_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Visualization 5: Step-by-step prediction flow
# ---------------------------------------------------------------------------
def plot_prediction_flow():
    """Show how a single prediction step works, left to right."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    steps = [
        ("1. Input Image", "#78909C",
         "224×224 image\ndivided into\n16×16 = 256 patches"),
        ("2. Mask & Split", "#607D8B",
         "Context: ~25%\n(contiguous block)\n\nTarget: ~60%\n(4 large blocks)\n\nUnused: ~15%"),
        ("3. Encode Context", "#1565C0",
         "ViT processes\nonly visible patches\n→ rich embeddings\nfor each patch"),
        ("4. Predict Targets", "#6A1B9A",
         "Small ViT takes\ncontext embeddings\n+ target positions\n→ predicted\ntarget embeddings"),
        ("5. Compare", "#F57F17",
         "L2 distance between\npredicted and actual\ntarget embeddings\n\nNo pixels involved!"),
    ]

    for i, (title, color, desc) in enumerate(steps):
        ax = axes[i]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Step box
        box = FancyBboxPatch(
            (0.3, 6.5), 9.4, 2.5,
            boxstyle="round,pad=0.2",
            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(5, 7.8, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")

        # Description
        ax.text(5, 3.5, desc, ha="center", va="center",
                fontsize=10, color="#333333", linespacing=1.5)

        # Arrow to next
        if i < 4:
            ax.annotate("", xy=(10.2, 7.8), xytext=(9.8, 7.8),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=2),
                        annotation_clip=False)

    plt.suptitle("I-JEPA: One Training Step", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = OUTPUT_DIR / "02_prediction_flow.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Visualization 6: Why JEPA features transfer
# ---------------------------------------------------------------------------
def plot_transfer_diagram():
    """Diagram explaining why JEPA representations generalize to unseen domains."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def add_box(x, y, w, h, text, color, fontsize=11, text_color="white"):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                linespacing=1.4)

    # Training side
    ax.text(3.5, 6.5, "TRAINING", fontsize=14, fontweight="bold",
            ha="center", color="#1565C0")

    add_box(0.5, 4.8, 2.5, 1.2, "ImageNet\n1.3M images", "#78909C")
    add_box(3.5, 4.8, 3, 1.2, "I-JEPA learns:\nshape, texture,\ncolor, structure", "#1565C0")

    ax.annotate("", xy=(3.5, 5.4), xytext=(3.0, 5.4),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    # What it learns (middle)
    concepts = [
        (1.0, 3.0, "edges &\ncontours", "#42A5F5"),
        (3.5, 3.0, "shapes &\nparts", "#1E88E5"),
        (6.0, 3.0, "spatial\nrelations", "#1565C0"),
        (8.5, 3.0, "semantic\nconcepts", "#0D47A1"),
    ]
    for x, y, text, color in concepts:
        add_box(x, y, 2, 1.0, text, color, fontsize=10)

    # Arrows between concept levels
    for i in range(3):
        x1 = concepts[i][0] + 2
        x2 = concepts[i + 1][0]
        ax.annotate("", xy=(x2, 3.5), xytext=(x1, 3.5),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))

    ax.text(5.5, 2.3, "low-level  ──────────────────────────▶  high-level",
            fontsize=10, ha="center", color="#666", fontstyle="italic")

    # Transfer side
    ax.text(11, 6.5, "TRANSFER", fontsize=14, fontweight="bold",
            ha="center", color="#2E7D32")

    add_box(9.5, 4.8, 3, 1.2, "New domain:\nFlowers, cars,\nmedical scans...", "#2E7D32")

    ax.annotate("", xy=(9.5, 5.4), xytext=(6.5, 5.4),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    # Why it works box
    add_box(0.5, 0.3, 13, 1.5,
            "Key: JEPA predicts MEANING, not pixels  →  learns universal visual concepts  →  transfers to any visual domain",
            "#FFF9C4", fontsize=12, text_color="#333")

    ax.text(7, 6.8, "Why I-JEPA Features Transfer to Unseen Domains",
            fontsize=16, fontweight="bold", ha="center")

    path = OUTPUT_DIR / "02_transfer.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Visualization 7: MAE vs JEPA side by side (what each predicts)
# ---------------------------------------------------------------------------
def plot_mae_vs_jepa():
    """Visual comparison of what MAE and JEPA actually predict."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    images = load_sample_images()
    img = images[0]
    grid_size = 16
    patch_px = 224 // grid_size
    ctx, tgt = generate_ijepa_masks(grid_size, grid_size)

    # Row 0: MAE approach
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image", fontsize=13)
    axes[0, 0].axis("off")

    # MAE: random scattered mask
    rng = random.Random(42)
    mae_mask = set(rng.sample(range(grid_size * grid_size), int(grid_size * grid_size * 0.75)))
    masked_mae = img.copy()
    for p in mae_mask:
        r, c = divmod(p, grid_size)
        y0, x0 = r * patch_px, c * patch_px
        masked_mae[y0:y0 + patch_px, x0:x0 + patch_px] = 0.7
    axes[0, 1].imshow(masked_mae)
    axes[0, 1].set_title("MAE: Random 75% Masked", fontsize=13)
    axes[0, 1].axis("off")

    # MAE output: reconstructed pixels (simulated with blur)
    from PIL import ImageFilter
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(3))) / 255.0
    recon = img.copy()
    for p in mae_mask:
        r, c = divmod(p, grid_size)
        y0, x0 = r * patch_px, c * patch_px
        recon[y0:y0 + patch_px, x0:x0 + patch_px] = blurred[y0:y0 + patch_px, x0:x0 + patch_px]
    axes[0, 2].imshow(recon)
    axes[0, 2].set_title("MAE Predicts: Pixels\n(must reconstruct textures, colors)", fontsize=12,
                         color="#E65100")
    axes[0, 2].axis("off")

    # Row 1: JEPA approach
    axes[1, 0].imshow(img)
    axes[1, 0].set_title("Original Image", fontsize=13)
    axes[1, 0].axis("off")

    # JEPA: block mask
    masked_jepa = np.ones_like(img) * 0.85
    overlay = np.zeros((224, 224, 4))
    for p in range(grid_size * grid_size):
        r, c = divmod(p, grid_size)
        y0, x0 = r * patch_px, c * patch_px
        if p in ctx:
            masked_jepa[y0:y0 + patch_px, x0:x0 + patch_px] = img[y0:y0 + patch_px, x0:x0 + patch_px]
            overlay[y0:y0 + patch_px, x0:x0 + patch_px] = [0.13, 0.59, 0.95, 0.4]
        elif p in tgt:
            overlay[y0:y0 + patch_px, x0:x0 + patch_px] = [1.0, 0.34, 0.13, 0.5]

    axes[1, 1].imshow(img)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("JEPA: Block Masking\n(blue=context, red=target)", fontsize=12)
    axes[1, 1].axis("off")

    # JEPA output: show that predictions are abstract vectors, not pixels
    # Draw a clean schematic: target regions with vector notation
    abstract_bg = np.ones((224, 224, 3)) * 0.95
    # Show context patches faintly
    for p in ctx:
        r, c = divmod(p, grid_size)
        y0, x0 = r * patch_px, c * patch_px
        abstract_bg[y0:y0 + patch_px, x0:x0 + patch_px] = [0.85, 0.91, 0.97]
    # Show target patches as colored blocks with vector labels
    target_colors = [
        [0.99, 0.88, 0.85],  # light red
        [0.85, 0.93, 0.85],  # light green
        [0.88, 0.85, 0.97],  # light purple
        [0.97, 0.95, 0.82],  # light yellow
    ]
    # Group target patches into contiguous blocks for labeling
    target_list = sorted(tgt)
    # Assign colors by spatial region
    for p in target_list:
        r, c = divmod(p, grid_size)
        y0, x0 = r * patch_px, c * patch_px
        color_idx = (r // 4 + c // 8) % len(target_colors)
        abstract_bg[y0:y0 + patch_px, x0:x0 + patch_px] = target_colors[color_idx]
        # Add thin border
        abstract_bg[y0, x0:x0 + patch_px] = [0.7, 0.7, 0.7]
        abstract_bg[y0:y0 + patch_px, x0] = [0.7, 0.7, 0.7]

    axes[1, 2].imshow(abstract_bg)
    # Add vector labels on the target regions
    axes[1, 2].text(112, 60, "z = [0.3, -0.1, 0.8, ...]",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#1565C0",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#1565C0", alpha=0.9))
    axes[1, 2].text(112, 160, "abstract features\nnot pixels",
                    ha="center", va="center", fontsize=11,
                    fontstyle="italic", color="#555",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#999", alpha=0.85))
    axes[1, 2].set_title("JEPA Predicts: Representations\n(1024-dim feature vectors)", fontsize=12,
                         color="#1565C0")
    axes[1, 2].axis("off")

    # Row labels
    fig.text(0.02, 0.75, "MAE", fontsize=16, fontweight="bold", color="#E65100",
             rotation=90, va="center")
    fig.text(0.02, 0.3, "JEPA", fontsize=16, fontweight="bold", color="#1565C0",
             rotation=90, va="center")

    plt.suptitle("MAE vs I-JEPA: What Gets Predicted?", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0, 1, 0.93])
    path = OUTPUT_DIR / "02_mae_vs_jepa.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Visualization 8: EMA and collapse prevention
# ---------------------------------------------------------------------------
def plot_ema_explained():
    """Diagram explaining EMA and why it prevents collapse."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Without EMA (collapse)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Without EMA: Collapse", fontsize=15, fontweight="bold", color="#C62828")

    def add_box(ax, x, y, w, h, text, color, fontsize=11, text_color="white"):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, linespacing=1.4)

    add_box(ax, 1, 7, 3.5, 1.5, "Encoder A", "#C62828")
    add_box(ax, 5.5, 7, 3.5, 1.5, "Encoder B", "#C62828")
    ax.annotate("", xy=(5.5, 7.75), xytext=(4.5, 7.75),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=2))
    ax.text(5, 7.2, "both trained", fontsize=9, ha="center", fontstyle="italic")

    add_box(ax, 2, 4, 6, 1.5, "Both encoders converge\nto same trivial output", "#EF9A9A",
            text_color="#333")
    ax.annotate("", xy=(5, 5.5), xytext=(5, 7),
                arrowprops=dict(arrowstyle="-|>", color="#C62828", lw=2))

    add_box(ax, 1.5, 1.5, 7, 1.5, "All images → same vector\nLoss = 0, but useless!", "#F44336")
    ax.annotate("", xy=(5, 3), xytext=(5, 4),
                arrowprops=dict(arrowstyle="-|>", color="#C62828", lw=2))

    # Right: With EMA (stable)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("With EMA: Stable Learning", fontsize=15, fontweight="bold", color="#2E7D32")

    add_box(ax, 1, 7, 3.5, 1.5, "Context\nEncoder", "#1565C0")
    add_box(ax, 5.5, 7, 3.5, 1.5, "Target\nEncoder", "#BF360C")

    ax.annotate("", xy=(5.5, 8.0), xytext=(4.5, 8.0),
                arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=2, linestyle="--"))
    ax.text(5, 8.5, "slow EMA copy", fontsize=9, ha="center", fontstyle="italic", color="#1565C0")
    ax.text(7.25, 6.5, "stop gradient\n(no training)", fontsize=9, ha="center",
            fontstyle="italic", color="#BF360C")

    add_box(ax, 2, 4, 6, 1.5, "Target moves slowly\n→ stable prediction goal", "#C8E6C9",
            text_color="#333")
    ax.annotate("", xy=(5, 5.5), xytext=(5, 7),
                arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2))

    add_box(ax, 1.5, 1.5, 7, 1.5, "Rich, diverse representations\nthat capture real visual structure", "#4CAF50")
    ax.annotate("", xy=(5, 3), xytext=(5, 4),
                arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2))

    plt.suptitle("Why EMA Matters: Preventing Representation Collapse",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = OUTPUT_DIR / "02_ema_explained.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  Demo 02 — I-JEPA Masking Strategy Explained")
    print("=" * 55)

    print("\n  [1/8] Masking overlays on images...")
    plot_masking_on_images()

    print("  [2/8] Mask variations on same image...")
    plot_mask_variations()

    print("  [3/8] Architecture diagram...")
    plot_architecture_diagram()

    print("  [4/8] Method comparison...")
    plot_method_comparison()

    print("  [5/8] Prediction flow...")
    plot_prediction_flow()

    print("  [6/8] Transfer diagram...")
    plot_transfer_diagram()

    print("  [7/8] MAE vs JEPA...")
    plot_mae_vs_jepa()

    print("  [8/8] EMA explained...")
    plot_ema_explained()

    print(f"\n  Done! Check {OUTPUT_DIR}/02_*.png")


if __name__ == "__main__":
    main()
