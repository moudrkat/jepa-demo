"""Stitch prediction GIFs into a single LinkedIn video with title cards."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

OUTPUT_DIR = Path("outputs")
W, H = 1400, 900
FPS = 3
TITLE_HOLD = FPS * 3  # 3 seconds per title card
BG_COLOR = (26, 26, 46)  # #1A1A2E

SEGMENTS = [
    # --- Prologue ---
    {"title": "I took V-JEPA 2 to a playground", "subtitle": "First, we had some fun with cubes.", "gif": None, "hold": FPS * 3},
    # --- Opener: cubes on JEPA paper ---
    {"title": "", "subtitle": "", "gif": "sliding_jepa_paper_opener.gif", "hold": 0, "max_frames": 27},
    # --- Watering can ---
    {"title": "Then I poured from an empty watering can", "subtitle": 'V-JEPA 2: "Pretending to pour,\nbut something is empty" \u2014 83%', "gif": None, "hold": FPS * 3},
    {"title": "", "subtitle": "", "gif": "11_prediction_watering_can.gif", "hold": 0, "max_frames": 14},
    # --- Showing empty ---
    {"title": "I showed it an empty toolbox", "subtitle": 'V-JEPA 2: "Showing that something\nis empty" \u2014 99.5%', "gif": None, "hold": FPS * 3},
    {"title": "", "subtitle": "", "gif": "11_prediction_showing_empty.gif", "hold": 0, "max_frames": 12},
    # --- Showing inside ---
    {"title": "I showed it a ball inside a tube", "subtitle": 'V-JEPA 2: "Showing that something\nis inside something" \u2014 98%', "gif": None, "hold": FPS * 3},
    {"title": "", "subtitle": "", "gif": "11_prediction_showing_inside.gif", "hold": 0, "max_frames": 11},
    # --- DO / PRETEND pair ---
    {"title": "I tried to fool the model. Same toy. Same hand.\nOnce real. Once pretend.", "subtitle": "", "gif": None, "hold": FPS * 3},
    {"title": "I put a ball into a box", "subtitle": 'V-JEPA 2: "Putting something into something" \u2014 72%', "gif": None, "hold": FPS * 3},
    {"title": "", "subtitle": "", "gif": "11_prediction_real_put_bag.gif", "hold": 0},
    {"title": "I only pretended to put a ball into a box", "subtitle": 'V-JEPA 2: "Pretending to put something\ninto something" \u2014 92%', "gif": None, "hold": FPS * 3},
    {"title": "", "subtitle": "", "gif": "11_prediction_pretend_put_into.gif", "hold": 0,
     "freeze_at": 8, "freeze_text": "Got it!\n\"Pretending to put something\ninto something\"", "freeze_hold": FPS * 2},
    # --- Outro ---
    {"title": "Now I can't stop thinking:\nwhat is the future of play?", "subtitle": "And is it in the representation space? :)", "gif": None, "hold": FPS * 5},
    {"title": "Starring:\nmany colorful toys\nand this paper", "subtitle": "Revisiting Feature Prediction for Learning\nVisual Representations from Video\narxiv.org/abs/2404.08471", "gif": None, "hold": FPS * 5},
]


def make_title_card(title, subtitle, w=W, h=H):
    """Create a title card image."""
    img = Image.new("RGB", (w, h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Try to load a nice font, fall back to default
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except OSError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    title_lines = title.split("\n")
    y = h // 2 - 80
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=font_big)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, y), line, fill="white", font=font_big)
        y += 60

    # Subtitle
    y += 20
    sub_lines = subtitle.split("\n")
    for line in sub_lines:
        bbox = draw.textbbox((0, 0), line, font=font_small)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, y), line, fill=(150, 180, 220), font=font_small)
        y += 45

    return np.array(img)


def load_gif_frames(gif_path):
    """Load all frames from a GIF."""
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy().convert("RGB").resize((W, H), Image.LANCZOS)
            frames.append(np.array(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


def main():
    out_path = OUTPUT_DIR / "linkedin_post_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))

    for seg in SEGMENTS:
        print(f"  {seg['title'][:50]}...")

        # Title card (skip if no title text)
        if seg["title"]:
            card = make_title_card(seg["title"], seg["subtitle"])
            card_bgr = cv2.cvtColor(card, cv2.COLOR_RGB2BGR)
            hold = seg["hold"] if seg["hold"] > 0 else TITLE_HOLD
            for _ in range(hold):
                writer.write(card_bgr)

        # GIF frames
        if seg["gif"]:
            gif_path = OUTPUT_DIR / seg["gif"]
            if gif_path.exists():
                frames = load_gif_frames(gif_path)
                if "max_frames" in seg:
                    frames = frames[:seg["max_frames"]]
                elif seg["gif"].startswith("11_prediction_"):
                    # Trim hold frames at end (last 6 frames repeat)
                    frames = frames[:16]
                print(f"    {len(frames)} frames from {seg['gif']}")
                freeze_at = seg.get("freeze_at")
                for fi, frame in enumerate(frames):
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
                    # Freeze and overlay text at specific frame
                    if freeze_at is not None and fi == freeze_at:
                        freeze_text = seg.get("freeze_text", "")
                        freeze_hold = seg.get("freeze_hold", FPS * 2)
                        # Create overlay: take the current frame, add text on top
                        overlay = Image.fromarray(frame.copy())
                        draw = ImageDraw.Draw(overlay)
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
                        except OSError:
                            font = ImageFont.load_default()
                        # Semi-transparent banner at bottom
                        draw.rectangle([(0, H - 180), (W, H)], fill=(26, 26, 46, 230))
                        # Text
                        text_lines = freeze_text.split("\n")
                        ty = H - 170
                        for tl in text_lines:
                            bbox = draw.textbbox((0, 0), tl, font=font)
                            tw = bbox[2] - bbox[0]
                            draw.text(((W - tw) // 2, ty), tl, fill=(79, 195, 247), font=font)
                            ty += 45
                        overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
                        for _ in range(freeze_hold):
                            writer.write(overlay_bgr)
            else:
                print(f"    WARNING: {gif_path} not found!")

    writer.release()

    # Re-encode with ffmpeg for better compatibility
    final_path = OUTPUT_DIR / "linkedin_vjepa_playground.mp4"
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(out_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "medium",
        str(final_path)
    ], capture_output=True)

    out_path.unlink()  # remove temp
    print(f"\n  Done! {final_path}")
    print(f"  Size: {final_path.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()