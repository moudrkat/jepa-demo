"""
Demo 11 – Playground clips in Something-Something representation space
=====================================================================
Embeds SSv2 clips AND selected playground clips together using sliding-window
embeddings (per-window, not averaged), then visualises them in a joint t-SNE.
Shows that playground actions land near matching SSv2 actions.

Outputs:
  11_playground_in_sthsth_space.png — static plot
  11_playground_in_sthsth_space.gif — animated: SSv2 clusters appear first,
                                      then playground clips drop in one by one
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
from sklearn.manifold import TSNE

matplotlib.use("Agg")

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data/playground_dataset")
DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

WINDOW_SIZE = 16
STRIDE = 4

# ── SSv2 clips (same as demo 10) ──

BASE_URL = (
    "https://huggingface.co/datasets/Nojah/"
    "limited_something_something_v2/resolve/main/videos"
)

STHSTH_CLIPS = {
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

# ── Playground clips ──

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

# Lighter shades for playground points (same hue family, distinguishable)
PLAYGROUND_COLORS = {
    "pouring":      "#ff6b8a",
    "folding":      "#7de67d",
    "transferring": "#7b9cf0",
    "unscrewing":   "#ffb87a",
    "placing":      "#c77ddb",
    "lifting":      "#7aeeff",
    "opening":      "#ff7af0",
    "sorting":      "#e0ff7a",
}


def download_video(url):
    suffix = Path(url).suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


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


def extract_windowed_embeddings(model, processor, frames):
    """Extract per-window embeddings from a list of frames."""
    T = len(frames)
    if T < WINDOW_SIZE:
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
        inputs = processor(frames[:WINDOW_SIZE], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings.append(emb)

    return embeddings


def create_static_plot(sthsth_tsne, sthsth_actions, pg_tsne, pg_actions,
                       pg_labels, pg_pretend):
    """Static PNG: SSv2 clusters + playground per-window points."""
    print("  Creating static plot...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    # SSv2 background points
    sthsth_colors = [ACTION_COLORS[a] for a in sthsth_actions]
    ax.scatter(sthsth_tsne[:, 0], sthsth_tsne[:, 1],
               c=sthsth_colors, s=30, alpha=0.3, edgecolors="none")

    # Playground points — per-window, with different markers for real/pretend
    for i in range(len(pg_tsne)):
        x, y = pg_tsne[i]
        action = pg_actions[i]
        pretend = pg_pretend[i]
        color = PLAYGROUND_COLORS[action]
        marker = "X" if pretend else "*"
        size = 120 if pretend else 150
        ax.scatter(x, y, c=color, s=size, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=10, alpha=0.85)

    # Labels — one per clip group, placed at centroid
    seen = {}
    for i in range(len(pg_tsne)):
        label = pg_labels[i]
        if label not in seen:
            seen[label] = []
        seen[label].append(pg_tsne[i])

    for label, points in seen.items():
        cx, cy = np.mean(points, axis=0)
        action = None
        for i, l in enumerate(pg_labels):
            if l == label:
                action = pg_actions[i]
                break
        ax.annotate(label, (cx, cy), fontsize=8, color="white",
                    fontweight="bold", ha="left", va="bottom",
                    xytext=(8, 5), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#1A1A2E",
                              ec=PLAYGROUND_COLORS.get(action, "white"),
                              alpha=0.85))

    # Legend — SSv2 actions
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

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("My playground clips in Something-Something-V2 representation space",
                 fontsize=14, fontweight="bold", color="white", pad=15)
    fig.suptitle("V-JEPA 2 (pretrained, no labels) — per-window embeddings",
                 fontsize=11, color="#AAAACC", y=0.02)

    save_path = OUTPUT_DIR / "11_playground_in_sthsth_space.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def create_animated_plot(sthsth_tsne, sthsth_actions, pg_tsne, pg_actions,
                         pg_labels, pg_pretend, clip_boundaries):
    """Animated GIF: SSv2 clusters fade in, then playground clips appear one by one."""
    print("  Creating animated GIF...")

    FPS = 2
    N_FADE_FRAMES = 6
    N_HOLD_BETWEEN = 3
    N_HOLD_END = 8
    n_clips = len(clip_boundaries)
    total_frames = N_FADE_FRAMES + n_clips * (1 + N_HOLD_BETWEEN) + N_HOLD_END

    # Build clip info for animation
    clip_info = []
    for label, (start, end, action, pretend) in clip_boundaries.items():
        clip_info.append((label, start, end, action, pretend))

    sthsth_colors = [ACTION_COLORS[a] for a in sthsth_actions]

    # Legend handles
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

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#1A1A2E")

    def update(frame):
        ax.clear()
        ax.set_facecolor("#1A1A2E")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # SSv2 fade in
        alpha = min(1.0, frame / max(1, N_FADE_FRAMES - 1)) * 0.3
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
                current_label = clip_info[n_show - 1][0] if n_show > 0 else ""
                title_text = f"Dropping in: {current_label}"
            else:
                title_text = "My playground clips in SSv2 action space!"

        # Draw playground clips shown so far
        for ci in range(n_show):
            label, start, end, action, pretend = clip_info[ci]
            is_newest = (ci == n_show - 1) and (n_show <= n_clips)
            color = PLAYGROUND_COLORS[action]
            marker = "X" if pretend else "*"
            size = 120 if pretend else 150
            edge_color = "#FFD700" if is_newest else "white"
            edge_width = 1.5 if is_newest else 0.8
            a = 0.95 if is_newest else 0.7

            ax.scatter(pg_tsne[start:end, 0], pg_tsne[start:end, 1],
                       c=color, s=size, marker=marker,
                       edgecolors=edge_color, linewidths=edge_width,
                       zorder=10, alpha=a)

            # Label at centroid
            cx = pg_tsne[start:end, 0].mean()
            cy = pg_tsne[start:end, 1].mean()
            ax.annotate(label, (cx, cy), fontsize=8, color="white",
                        fontweight="bold", ha="left", va="bottom",
                        xytext=(8, 5), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1A1A2E",
                                  ec=color, alpha=0.85))

        ax.set_title(title_text, fontsize=14, fontweight="bold",
                     color="white", pad=15)
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
    print("        (per-window embeddings, all extracted together)")
    print("=" * 60)

    # ── Load model ──
    print("\n[1/5] Loading model...")
    model, processor = load_model()

    # ── Extract SSv2 embeddings ──
    print("\n[2/5] Extracting SSv2 clip embeddings...")
    all_sthsth_embeddings = []
    all_sthsth_actions = []

    for action, url in STHSTH_CLIPS.items():
        print(f"  {action}...")
        path = download_video(url)
        frames = load_frames(path)
        embs = extract_windowed_embeddings(model, processor, frames)
        all_sthsth_embeddings.extend(embs)
        all_sthsth_actions.extend([action] * len(embs))
        print(f"    {len(frames)} frames → {len(embs)} windows")

    print(f"  Total SSv2: {len(all_sthsth_embeddings)} window embeddings")

    # ── Extract playground embeddings ──
    print("\n[3/5] Extracting playground clip embeddings...")
    all_pg_embeddings = []
    all_pg_actions = []
    all_pg_labels = []
    all_pg_pretend = []
    clip_boundaries = {}  # label → (start_idx, end_idx, action, pretend)

    for key, (fname, label, action, is_pretend) in PLAYGROUND_CLIPS.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        print(f"  {label} ({fname})...")
        frames = load_frames(path)
        embs = extract_windowed_embeddings(model, processor, frames)
        start = len(all_pg_embeddings)
        all_pg_embeddings.extend(embs)
        all_pg_actions.extend([action] * len(embs))
        all_pg_labels.extend([label] * len(embs))
        all_pg_pretend.extend([is_pretend] * len(embs))
        clip_boundaries[label] = (start, start + len(embs), action, is_pretend)
        print(f"    {len(frames)} frames → {len(embs)} windows")

    print(f"  Total playground: {len(all_pg_embeddings)} window embeddings")

    del model
    gc.collect()

    # ── Joint t-SNE ──
    print("\n[4/5] Computing joint t-SNE...")
    sthsth_arr = np.array(all_sthsth_embeddings)
    pg_arr = np.array(all_pg_embeddings)
    all_embeddings = np.vstack([sthsth_arr, pg_arr])
    n_sthsth = len(sthsth_arr)

    perplexity = min(30, len(all_embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                random_state=42)
    all_tsne = tsne.fit_transform(all_embeddings)

    sthsth_tsne = all_tsne[:n_sthsth]
    pg_tsne = all_tsne[n_sthsth:]

    print(f"  t-SNE done: {len(all_tsne)} points total")

    # Save
    np.savez(
        OUTPUT_DIR / "11_embeddings.npz",
        sthsth_embeddings=sthsth_arr,
        sthsth_tsne=sthsth_tsne,
        sthsth_actions=all_sthsth_actions,
        pg_embeddings=pg_arr,
        pg_tsne=pg_tsne,
        pg_actions=all_pg_actions,
        pg_labels=all_pg_labels,
        pg_pretend=all_pg_pretend,
    )

    # ── Visualise ──
    print("\n[5/5] Creating visualisations...")
    create_static_plot(sthsth_tsne, all_sthsth_actions, pg_tsne,
                       all_pg_actions, all_pg_labels, all_pg_pretend)
    create_animated_plot(sthsth_tsne, all_sthsth_actions, pg_tsne,
                         all_pg_actions, all_pg_labels, all_pg_pretend,
                         clip_boundaries)

    print("\n✓ Done! Check outputs/11_playground_in_sthsth_space.{png,gif}")


if __name__ == "__main__":
    main()