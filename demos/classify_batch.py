"""Quick batch classifier — runs V-JEPA 2 (SSv2 fine-tuned) on a list of videos."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

DEVICE = "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"


def load_video(path, max_frames=64):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sample_frames(frames, n=16):
    total = len(frames)
    if total >= n:
        indices = np.linspace(0, total - 1, n, dtype=int)
    else:
        indices = list(range(total)) + [total - 1] * (n - total)
        indices = np.array(indices)
    return [frames[i] for i in indices]


def main():
    from transformers import AutoVideoProcessor, AutoModelForVideoClassification

    video_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/playground_dataset")
    videos = sorted(video_dir.glob("*.mp4"))

    print(f"Loading {MODEL_ID}...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
    model.eval().to(DEVICE)
    print(f"Ready. {len(videos)} videos.\n")

    print(f"{'Video':>45}  {'Top-1 prediction':>50}  {'Conf':>5}  {'#2':>40}  {'Conf':>5}")
    print("-" * 155)

    for v in videos:
        frames = load_video(v)
        if len(frames) < 4:
            print(f"{v.name:>45}  {'(too short)':>50}")
            continue
        sampled = sample_frames(frames)
        inputs = processor(sampled, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        top2 = probs.argsort(descending=True)[:2]
        l1 = model.config.id2label[top2[0].item()]
        p1 = probs[top2[0]].item()
        l2 = model.config.id2label[top2[1].item()]
        p2 = probs[top2[1]].item()
        print(f"{v.name:>45}  {l1:>50}  {p1:>5.1%}  {l2:>40}  {p2:>5.1%}")


if __name__ == "__main__":
    main()