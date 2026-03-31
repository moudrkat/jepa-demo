"""Sliding-window classifier — runs V-JEPA 2 (SSv2) across a long video and shows action predictions over time."""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

matplotlib.use("Agg")

DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
OUTPUT_DIR = Path("outputs")
WINDOW = 16
FPS_GIF = 3
TOP_K = 5


def load_video(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while cap.isOpened():
        if max_frames and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"  {len(frames)} frames, {fps:.0f} fps")
    return frames, fps


def classify_window(model, processor, window_frames):
    """Classify a 16-frame window."""
    n = len(window_frames)
    if n >= 16:
        indices = np.linspace(0, n - 1, 16, dtype=int)
    else:
        indices = list(range(n)) + [n - 1] * (16 - n)
    sampled = [window_frames[i] for i in indices]
    inputs = processor(sampled, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    top_idx = probs.argsort(descending=True)[:TOP_K]
    preds = [(model.config.id2label[i.item()], probs[i].item()) for i in top_idx]
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    video_path = Path(args.video)
    name = args.name or video_path.stem
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n[1/4] Loading video...")
    frames, fps = load_video(video_path)

    print(f"\n[2/4] Loading classifier...")
    from transformers import AutoVideoProcessor, AutoModelForVideoClassification
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
    model.eval().to(DEVICE)
    print(f"  Ready.")

    print(f"\n[3/4] Classifying windows (stride={args.stride})...")
    n_windows = max(1, (len(frames) - WINDOW) // args.stride + 1)
    all_preds = []
    all_centers = []
    all_center_frames = []

    for i, start in enumerate(range(0, len(frames) - WINDOW + 1, args.stride)):
        window = frames[start:start + WINDOW]
        preds = classify_window(model, processor, window)
        center = start + WINDOW // 2
        all_preds.append(preds)
        all_centers.append(center)
        all_center_frames.append(frames[center])
        if (i + 1) % 5 == 0 or i + 1 == n_windows:
            top = preds[0]
            print(f"    [{i+1}/{n_windows}] t={center/fps:.1f}s  {top[0][:50]}  {top[1]:.0%}")

    print(f"\n[4/4] Creating animated GIF...")

    # Build GIF: video frame on left, prediction bars on right
    step = max(1, len(all_preds) // 60)
    frame_indices = list(range(0, len(all_preds), step))
    n_hold = 8
    total_frames = len(frame_indices) + n_hold

    fig, ax_vid = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1A1A2E")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08)

    # Progress bar
    progress_bg = plt.Rectangle((0.05, 0.03), 0.90, 0.02,
                                 transform=fig.transFigure, facecolor="#333355",
                                 edgecolor="none", zorder=10)
    progress_fill = plt.Rectangle((0.05, 0.03), 0.0, 0.02,
                                   transform=fig.transFigure, facecolor="#4FC3F7",
                                   edgecolor="none", zorder=11)
    fig.patches.extend([progress_bg, progress_fill])

    def update(i):
        idx = min(i, len(frame_indices) - 1)
        data_idx = frame_indices[idx]
        preds = all_preds[data_idx]
        frame = all_center_frames[data_idx]
        t_sec = all_centers[data_idx] / fps
        frac = (idx + 1) / len(frame_indices)

        ax_vid.clear()
        ax_vid.imshow(frame)
        ax_vid.set_xticks([])
        ax_vid.set_yticks([])
        for spine in ax_vid.spines.values():
            spine.set_color("#4FC3F7")
            spine.set_linewidth(2)

        # Top prediction as text overlay
        top_label = preds[0][0]
        top_prob = preds[0][1]
        fig.suptitle(f"{top_label}  —  {top_prob:.0%}",
                     fontsize=22, fontweight="bold", color="white", y=0.95)
        ax_vid.set_title(f"t = {t_sec:.1f}s", fontsize=13,
                         fontweight="bold", color="#4FC3F7", loc="left")

        progress_fill.set_width(0.90 * frac)

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // FPS_GIF, blit=False)
    save_path = OUTPUT_DIR / f"sliding_{name}.gif"
    anim.save(str(save_path), writer="pillow", fps=FPS_GIF, dpi=100)
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # Also save raw predictions as text
    txt_path = OUTPUT_DIR / f"sliding_{name}.txt"
    with open(txt_path, "w") as f:
        for center, preds in zip(all_centers, all_preds):
            t = center / fps
            f.write(f"t={t:.1f}s  {preds[0][0]}  {preds[0][1]:.1%}\n")
    print(f"  Saved: {txt_path}")


if __name__ == "__main__":
    main()