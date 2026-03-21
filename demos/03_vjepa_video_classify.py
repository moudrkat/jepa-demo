"""
Demo 03 — V-JEPA 2 Video Action Recognition
=============================================

Loads pretrained V-JEPA 2 (ViT-L) from HuggingFace and classifies actions
in video clips.  The model is fine-tuned on Something-Something-V2, so it
excels at fine-grained hand/object interaction recognition.

Run:
    python demos/03_vjepa_video_classify.py
"""

import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

# Sample videos — SSv2-style hand-object interactions
SAMPLE_VIDEOS = {
    "dipping_brush": "https://huggingface.co/datasets/Nojah/limited_something_something_v2/resolve/main/videos/102148.webm",
    "picking_up_pens": "https://huggingface.co/datasets/Nojah/limited_something_something_v2/resolve/main/videos/103874.webm",
    "pushing_object": "https://huggingface.co/datasets/Nojah/limited_something_something_v2/resolve/main/videos/106248.webm",
}


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------
def load_video_opencv(path_or_url, max_frames=64):
    """Load video frames using OpenCV. Returns numpy array (T, H, W, 3)."""
    import cv2
    import tempfile
    import urllib.request

    if str(path_or_url).startswith("http"):
        print(f"  Downloading video...")
        suffix = Path(path_or_url).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        urllib.request.urlretrieve(path_or_url, tmp.name)
        video_path = tmp.name
    else:
        video_path = str(path_or_url)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {path_or_url}")

    print(f"  Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})")
    return np.stack(frames)


def sample_frames(video, n_frames=16):
    """Uniformly sample n_frames from video. Returns (T, H, W, 3)."""
    total = len(video)
    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        # Repeat last frame if video is too short
        indices = list(range(total)) + [total - 1] * (n_frames - total)
        indices = np.array(indices)
    return video[indices]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model():
    from transformers import AutoVideoProcessor, AutoModelForVideoClassification
    print(f"  Loading {MODEL_ID} ...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
    model.eval()
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_labels = model.config.num_labels
    print(f"  Model loaded ({n_params:.0f}M params, {n_labels} action classes)")
    return model, processor


def classify_video(model, processor, video_frames, top_k=10):
    """Run inference on video frames. Returns list of (label, prob)."""
    inputs = processor(video_frames, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    top_indices = probs.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        label = model.config.id2label[idx.item()]
        prob = probs[idx].item()
        results.append((label, prob))
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_classification_result(video_frames, predictions, video_name, save_path):
    """Show sampled frames + top predictions as bar chart."""
    n_frames_to_show = min(8, len(video_frames))
    frame_indices = np.linspace(0, len(video_frames) - 1, n_frames_to_show, dtype=int)

    fig = plt.figure(figsize=(16, 8))

    # Top: frame strip
    for i, fi in enumerate(frame_indices):
        ax = fig.add_subplot(2, n_frames_to_show, i + 1)
        ax.imshow(video_frames[fi])
        ax.axis("off")
        ax.set_title(f"Frame {fi}", fontsize=9)

    # Bottom: bar chart of predictions
    ax_bar = fig.add_subplot(2, 1, 2)
    labels = [p[0] for p in predictions[:8]]
    probs = [p[1] for p in predictions[:8]]

    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(labels))]
    bars = ax_bar.barh(range(len(labels)), probs, color=colors, edgecolor="white")
    ax_bar.set_yticks(range(len(labels)))
    ax_bar.set_yticklabels(labels, fontsize=11)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Confidence", fontsize=12)
    ax_bar.set_xlim(0, 1)
    ax_bar.grid(axis="x", alpha=0.2)

    # Add probability text on bars
    for bar, prob in zip(bars, probs):
        ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}", va="center", fontsize=10)

    plt.suptitle(f"V-JEPA 2 Action Recognition — {video_name}",
                 fontsize=17, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_frame_grid(video_frames, save_path):
    """Show all 16 input frames in a grid."""
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    for i, ax in enumerate(axes.flat):
        if i < len(video_frames):
            ax.imshow(video_frames[i])
            ax.set_title(f"t={i}", fontsize=9)
        ax.axis("off")

    plt.suptitle("V-JEPA 2 Input: 16 Uniformly Sampled Frames",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  Demo 03 — V-JEPA 2 Video Classification")
    print("=" * 55)

    model, processor = load_model()

    for name, url in SAMPLE_VIDEOS.items():
        print(f"\n  Processing: {name}")

        # Load and sample frames
        raw_video = load_video_opencv(url)
        sampled = sample_frames(raw_video, n_frames=16)

        # Classify
        print("  Running inference (this may take ~30-60 sec on CPU)...")
        predictions = classify_video(model, processor, sampled)

        print(f"  Top prediction: {predictions[0][0]} ({predictions[0][1]:.1%})")
        for label, prob in predictions[:5]:
            print(f"    {prob:6.1%}  {label}")

        # Visualize
        plot_classification_result(
            sampled, predictions, name,
            OUTPUT_DIR / f"03_classify_{name}.png",
        )
        plot_frame_grid(sampled, OUTPUT_DIR / f"03_frames_{name}.png")

    del model
    gc.collect()
    print(f"\n  Done! Check {OUTPUT_DIR}/03_*.png")


if __name__ == "__main__":
    main()
