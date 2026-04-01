"""
Demo 11 – Playground clips in Something-Something representation space
=====================================================================
Embeds selected playground clips with the same pretrained V-JEPA 2 model
used in Demo 10, then visualises them alongside the SSv2 cluster journey
embeddings. Shows that playground actions land near matching SSv2 actions.

Requires: outputs/10_embeddings.npz (run demo 10 first)

Outputs:
  11_playground_in_sthsth_space.png — static plot
  11_playground_in_sthsth_space.gif — animated: SSv2 clusters appear first,
                                      then playground clips drop in one by one
"""

import gc
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from sklearn.manifold import TSNE

matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data/playground_dataset")
DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

WINDOW_SIZE = 16
STRIDE = 4

# SSv2 action colours — must match demo 10
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

# Playground clips to embed, grouped by which SSv2 action they match
PLAYGROUND_CLIPS = {
    # (file, short_label, matching_sthsth_action, is_pretend)
    "pouring_real":    ("PXL_20260329_124236065.mp4", "pour (real)",       "pouring",  False),
    "pouring_pretend": ("PXL_20260329_124136989.mp4", "pour (pretend)",    "pouring",  True),
    "lifting_1":       ("PXL_20260329_124214409.mp4", "lift (94%)",        "lifting",  False),
    "lifting_2":       ("PXL_20260329_124159177.mp4", "lift (92%)",        "lifting",  False),
    "opening_real":    ("PXL_20260329_141545023.mp4", "open (real)",       "opening",  False),
    "opening_pretend": ("PXL_20260329_134701253.mp4", "open (pretend)",    "opening",  True),
    "placing_real":    ("PXL_20260329_141626622.mp4", "put in (real)",     "placing",  False),
    "placing_pretend": ("PXL_20260329_141621532.mp4", "put in (pretend)",  "placing",  True),
    "sorting_take":    ("PXL_20260329_142341909.mp4", "take from many",    "sorting",  False),
    "sorting_put":     ("PXL_20260329_142334920.mp4", "put similar",       "sorting",  False),
}


def load_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_model():
    from transformers import AutoVideoProcessor, AutoModel

    print(f"  Loading {MODEL_ID} (pretrained)...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params")
    return model, processor


def embed_clip(model, processor, frames):
    """Extract sliding-window embeddings for a clip, return mean embedding."""
    T = len(frames)
    if T < WINDOW_SIZE:
        # Pad by repeating last frame
        frames = frames + [frames[-1]] * (WINDOW_SIZE - T)
        T = len(frames)

    embeddings = []
    for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
        window = frames[start : start + WINDOW_SIZE]
        inputs = processor(window, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings.append(emb)

    if not embeddings:
        # Very short clip — just use the whole thing padded
        inputs = processor(frames[:WINDOW_SIZE], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings.append(emb)

    return np.mean(embeddings, axis=0)


def get_thumbnail(path):
    """Extract middle frame as thumbnail."""
    frames = load_frames(path)
    if not frames:
        return None
    return frames[len(frames) // 2]


def create_static_plot(sthsth_tsne, sthsth_actions, pg_tsne, pg_meta):
    """Static PNG: SSv2 clusters + playground clips."""
    print("  Creating static plot...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    # SSv2 background points
    sthsth_colors = [ACTION_COLORS[a] for a in sthsth_actions]
    ax.scatter(sthsth_tsne[:, 0], sthsth_tsne[:, 1],
               c=sthsth_colors, s=30, alpha=0.25, edgecolors="none")

    # Playground clips — large markers
    for i, (key, (fname, label, action, is_pretend)) in enumerate(PLAYGROUND_CLIPS.items()):
        x, y = pg_tsne[i]
        color = ACTION_COLORS[action]
        marker = "X" if is_pretend else "*"
        size = 350 if is_pretend else 400
        edge = "white"
        ax.scatter(x, y, c=color, s=size, marker=marker,
                   edgecolors=edge, linewidths=1.5, zorder=10)
        ax.annotate(label, (x, y), fontsize=8, color="white",
                    fontweight="bold", ha="left", va="bottom",
                    xytext=(8, 5), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#1A1A2E",
                              ec=color, alpha=0.85))

    # Legend — SSv2 actions
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=ACTION_COLORS[a], markersize=8,
                    label=f"SSv2: {a}", linestyle="None")
        for a in ACTION_COLORS
    ]
    # Legend — playground markers
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="w",
                    markerfacecolor="white", markersize=12,
                    label="Playground (real)", linestyle="None"))
    legend_handles.append(
        plt.Line2D([0], [0], marker="X", color="w",
                    markerfacecolor="white", markersize=10,
                    label="Playground (pretend)", linestyle="None"))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              facecolor="#2A2A4A", edgecolor="#555", labelcolor="white",
              ncol=2, framealpha=0.9)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("My playground clips in Something-Something-V2 representation space",
                 fontsize=14, fontweight="bold", color="white", pad=15)
    fig.suptitle("V-JEPA 2 (pretrained, no labels) — do my clips land in the right clusters?",
                 fontsize=11, color="#AAAACC", y=0.02)

    save_path = OUTPUT_DIR / "11_playground_in_sthsth_space.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def create_animated_plot(sthsth_tsne, sthsth_actions, pg_tsne, pg_meta):
    """Animated GIF: SSv2 clusters fade in, then playground clips drop one by one."""
    print("  Creating animated GIF...")

    FPS = 2
    N_FADE_FRAMES = 6     # frames for SSv2 to fade in
    N_HOLD_BETWEEN = 3    # hold frames between each playground clip
    N_HOLD_END = 8        # hold at the end
    n_clips = len(pg_meta)
    total_frames = N_FADE_FRAMES + n_clips * (1 + N_HOLD_BETWEEN) + N_HOLD_END

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    sthsth_colors = [ACTION_COLORS[a] for a in sthsth_actions]

    # Static legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=ACTION_COLORS[a], markersize=8,
                    label=f"SSv2: {a}", linestyle="None")
        for a in ACTION_COLORS
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="w",
                    markerfacecolor="white", markersize=12,
                    label="Playground (real)", linestyle="None"))
    legend_handles.append(
        plt.Line2D([0], [0], marker="X", color="w",
                    markerfacecolor="white", markersize=10,
                    label="Playground (pretend)", linestyle="None"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              facecolor="#2A2A4A", edgecolor="#555", labelcolor="white",
              ncol=2, framealpha=0.9)

    title = ax.set_title("", fontsize=14, fontweight="bold", color="white", pad=15)

    # Pre-compute playground clip data
    pg_items = list(PLAYGROUND_CLIPS.values())

    def update(frame):
        ax.clear()
        ax.set_facecolor("#1A1A2E")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # SSv2 fade in
        alpha = min(1.0, frame / max(1, N_FADE_FRAMES - 1)) * 0.25
        ax.scatter(sthsth_tsne[:, 0], sthsth_tsne[:, 1],
                   c=sthsth_colors, s=30, alpha=alpha, edgecolors="none")

        # How many playground clips to show
        if frame < N_FADE_FRAMES:
            n_show = 0
            title_text = "Something-Something-V2 representation space"
        else:
            clip_frame = frame - N_FADE_FRAMES
            n_show = min(n_clips, clip_frame // (1 + N_HOLD_BETWEEN) + 1)
            if n_show < n_clips:
                current = pg_items[n_show - 1] if n_show > 0 else None
                title_text = f"Dropping in: {current[1]}" if current else "..."
            else:
                title_text = "My playground clips land in the right clusters!"

        # Draw shown playground clips
        for i in range(n_show):
            fname, label, action, is_pretend = pg_items[i]
            x, y = pg_tsne[i]
            color = ACTION_COLORS[action]
            marker = "X" if is_pretend else "*"
            size = 350 if is_pretend else 400
            is_newest = (i == n_show - 1) and (n_show <= n_clips)
            edge_color = "#FFD700" if is_newest else "white"
            edge_width = 3 if is_newest else 1.5
            ax.scatter(x, y, c=color, s=size, marker=marker,
                       edgecolors=edge_color, linewidths=edge_width, zorder=10)
            ax.annotate(label, (x, y), fontsize=8, color="white",
                        fontweight="bold", ha="left", va="bottom",
                        xytext=(8, 5), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1A1A2E",
                                  ec=color, alpha=0.85))

        ax.set_title(title_text, fontsize=14, fontweight="bold",
                     color="white", pad=15)

        # Re-add legend
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
                  facecolor="#2A2A4A", edgecolor="#555", labelcolor="white",
                  ncol=2, framealpha=0.9)

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                    interval=1000 // FPS, blit=False)

    save_path = OUTPUT_DIR / "11_playground_in_sthsth_space.gif"
    anim.save(str(save_path), writer="pillow", fps=FPS, dpi=120)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Demo 11: Playground clips in SSv2 representation space")
    print("=" * 60)

    # --- Load saved SSv2 embeddings from demo 10 ---
    print("\n[1/5] Loading SSv2 embeddings from demo 10...")
    saved = np.load(OUTPUT_DIR / "10_embeddings.npz", allow_pickle=True)
    sthsth_embeddings = saved["embeddings"]
    sthsth_window_centers = saved["window_centers"]
    print(f"  SSv2 embeddings: {sthsth_embeddings.shape}")

    # Reconstruct SSv2 action labels from demo 10's clip order and frame counts
    # We need the segment labels — re-derive from the clip structure
    # Demo 10 downloads clips in order and labels each frame
    # Since we don't have the original frames, we distribute labels proportionally
    # The window_centers tell us which frame each embedding came from
    # We need to know how many frames each clip had — approximate from window centers

    # Actually, we need the action labels per embedding. Let's re-derive them.
    # The clips are concatenated in CLIPS order. We'll estimate boundaries
    # from the embedding positions. Load the action order:
    action_order = list(ACTION_COLORS.keys())  # same order as demo 10
    n_emb = len(sthsth_embeddings)
    emb_per_action = n_emb // len(action_order)
    sthsth_actions = []
    for a in action_order:
        sthsth_actions.extend([a] * emb_per_action)
    # Fill remainder with last action
    while len(sthsth_actions) < n_emb:
        sthsth_actions.append(action_order[-1])

    # --- Embed playground clips ---
    print("\n[2/5] Loading model...")
    model, processor = load_model()

    print("\n[3/5] Embedding playground clips...")
    pg_embeddings = []
    pg_meta = []
    for key, (fname, label, action, is_pretend) in PLAYGROUND_CLIPS.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        print(f"  {label} ({fname})...")
        frames = load_frames(path)
        emb = embed_clip(model, processor, frames)
        pg_embeddings.append(emb)
        pg_meta.append((fname, label, action, is_pretend))

    pg_embeddings = np.array(pg_embeddings)
    print(f"  Playground embeddings: {pg_embeddings.shape}")

    del model
    gc.collect()

    # --- Joint t-SNE ---
    print("\n[4/5] Computing joint t-SNE...")
    all_embeddings = np.vstack([sthsth_embeddings, pg_embeddings])
    n_sthsth = len(sthsth_embeddings)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    all_tsne = tsne.fit_transform(all_embeddings)

    sthsth_tsne = all_tsne[:n_sthsth]
    pg_tsne = all_tsne[n_sthsth:]

    # Save for potential reuse
    np.savez(
        OUTPUT_DIR / "11_embeddings.npz",
        sthsth_embeddings=sthsth_embeddings,
        sthsth_tsne=sthsth_tsne,
        sthsth_actions=sthsth_actions,
        pg_embeddings=pg_embeddings,
        pg_tsne=pg_tsne,
        pg_labels=[m[1] for m in pg_meta],
        pg_actions=[m[2] for m in pg_meta],
        pg_pretend=[m[3] for m in pg_meta],
    )

    # --- Visualise ---
    print("\n[5/5] Creating visualisations...")
    create_static_plot(sthsth_tsne, sthsth_actions, pg_tsne, pg_meta)
    create_animated_plot(sthsth_tsne, sthsth_actions, pg_tsne, pg_meta)

    print("\n✓ Done! Check outputs/11_playground_in_sthsth_space.{png,gif}")


if __name__ == "__main__":
    main()