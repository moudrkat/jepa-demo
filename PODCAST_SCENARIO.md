# JEPA Podcast Scenario

**Topic:** What is JEPA and why it thinks differently than LLMs
**Format:** ~10 minute podcast segment with visuals
**Visuals:** All in `outputs/` folder
**Video:** `outputs/linkedin_vjepa_playground.mp4` (93s)

---

## Visual inventory

| File | Type | What it shows |
|------|------|---------------|
| `02_mae_vs_jepa.png` | PNG | MAE reconstructs pixels vs JEPA predicts representations |
| `07_mae_vs_jepa.png` | PNG | Side-by-side MAE vs JEPA on cat, dog, airplane, ship |
| `07_patch_zoom.png` | PNG | Zoomed single-image comparison — blurry pixel prediction |
| `08_tsne_animation.gif` | GIF | Random dots organizing into semantic clusters |
| `08_tsne_thumbnails.png` | PNG | t-SNE with actual image thumbnails |
| `10_cluster_journey.gif` | GIF | Video + dot moving through representation space |
| `linkedin_vjepa_playground.mp4` | MP4 | Playground demo: real vs pretend actions, sliding window |

---

## The scenario

### INTRO — Everyone knows LLMs, but... (1-2 min)

**Say:**
"So everyone knows ChatGPT. You type a question, it predicts the next word. And the next. And the next. That's how large language models work — they predict the next token in a sequence. And it's incredibly powerful for text.

But here's the thing — when you try to do the same with images and video, predicting the next *pixel*... it doesn't work as well. Imagine trying to predict the exact shade of green of the next pixel of grass. Or the exact reflection on a car hood. That's insanely hard, and most of it is meaningless noise.

This is where JEPA comes in — Joint Embedding Predictive Architecture. It's Yann LeCun's answer to this problem, and the idea is beautifully simple: **don't predict pixels. Predict meaning.**"

---

### ACT 1 — Pixels vs representations (2-3 min)

![MAE vs JEPA zoomed on a single cat image](outputs/07_patch_zoom.png)

**Say:**
"Let me show you what I mean. Here's a cat image. On the left, you have a generative approach — like what LLMs do, but for images. You hide parts of the image, and the model tries to reconstruct the exact pixels. See? It's blurry. It gets the rough shape but wastes enormous effort predicting fur texture, exact lighting, color gradients — stuff that's basically noise.

On the right: JEPA. It doesn't try to reconstruct pixels at all. Instead, it predicts the *representation* — a compressed, abstract description of what should be there. Not 'these specific brown pixels' but 'this is part of a cat's face.'"

![Full side-by-side: MAE pixel reconstruction vs JEPA feature activation](outputs/07_mae_vs_jepa.png)

"Here's the same comparison on four images — cat, dog, airplane, ship. Left: pixel reconstruction, blurry and wasteful. Right: JEPA's feature activation — see how it lights up on the meaningful parts? The cat's face, the dog's body, the airplane's shape. It ignores the background noise entirely.

This is the fundamental difference. LLMs predict the next *token*. Pixel models predict the next *pixel*. JEPA predicts the next *concept*."

![MAE predicts pixels, JEPA predicts representations](outputs/02_mae_vs_jepa.png)

"Here's the diagram. Left: predict pixels — hard, noisy, wasteful. Right: predict representations — efficient, semantic, useful. That's the whole idea."

---

### ACT 2 — What does JEPA actually learn? The representation space (2-3 min)

![t-SNE animation — clusters forming from random noise](outputs/08_tsne_animation.gif)

**Say:**
"Okay, but does this actually work? Let me show you. I took 300 images — cats, dogs, airplanes, trucks, ships — and ran them through JEPA. The model has *never seen labels*. Nobody told it what a cat is. Nobody told it what a ship is.

Each dot here is one image. Watch what happens when we visualize the model's internal representation space..."

*Let the GIF play — dots organize into clusters.*

"Animals on one side, vehicles on the other. Cats near dogs. Trucks near cars. The model organized all of this by itself, just from learning to predict representations."

![t-SNE with actual image thumbnails](outputs/08_tsne_thumbnails.png)

"And here's the same thing with actual thumbnails so you can see it's real. Horses together, ships together, cars together. The model doesn't know the *word* 'horse' — but it knows these images share something fundamental.

This is what representation space looks like. Instead of storing pixels, the model builds an internal map of *concepts*. Similar things live close together. Different things live far apart. It's like the model built its own mental dictionary — without anyone teaching it the words."

---

### ACT 3 — From images to video: V-JEPA (2-3 min)

![Cluster journey: video + dot moving through representation space](outputs/10_cluster_journey.gif)

**Say:**
"Now here's where it gets really interesting. The same idea works for video — that's V-JEPA. And in video, the representation space isn't just about *what things look like*, it's about *what's happening*.

Watch this. On the left, the actual video playing — someone pouring, folding, stacking. On the right, where the model's representation is in that abstract space. Every time the action changes, the dot *jumps* to a completely different region. Pouring lives here, folding lives there. The model discovered the structure of actions — no labels, no supervision."

---

### ACT 4 — The playground: can it tell real from fake? (2-3 min)

*Play `outputs/linkedin_vjepa_playground.mp4`*

**Say:**
"So I had to try this myself. I took V-JEPA 2 to a playground — literally, my kids' toys on the floor — and I tested something: can it tell the difference between a *real* action and a *pretended* one?

Watch. Same hand, same toy, same background. In one video I actually put a ball into a bag. In the other I just pretend to — same motion, but I don't let go. To a pixel-prediction model, these look almost identical. Same colors, same shapes, same movement.

But V-JEPA? It knows. It sees through the fake. Because it's not comparing pixels — it understood the *intent*. The ball ended up somewhere different, and in representation space, that matters.

And then I let it run on a longer video — sixty seconds of my kids just playing — and it segments the whole thing into actions in real time. No prompting, no tricks. It just... understands what's happening."

---

### CLOSING — Why this matters (1 min)

**Say:**
"So to bring it back: LLMs predict the next word. That works brilliantly for text. But for the visual world — images, video — predicting pixels is a dead end. Too much noise, too little meaning.

JEPA flips it: predict *representations*, not pixels. And the result is a model that builds an actual understanding of the visual world — it knows that cats are like dogs, it knows when an action changes, and it can even tell when you're faking it.

This is Yann LeCun's bet on how we get to real world models — AI that doesn't just generate text, but actually *understands* what it sees. We're still early. But the playground experiment convinced me: this thing gets it."

---

## Quick reference: show order

| # | Visual | When |
|---|--------|------|
| 1 | `07_patch_zoom.png` | Pixel prediction is blurry and wasteful |
| 2 | `07_mae_vs_jepa.png` | Side-by-side: pixels vs features |
| 3 | `02_mae_vs_jepa.png` | The core diagram: pixel vs representation |
| 4 | `08_tsne_animation.gif` | Representation space — clusters forming |
| 5 | `08_tsne_thumbnails.png` | Proof: actual thumbnails in clusters |
| 6 | `10_cluster_journey.gif` | Video: dot moving through action space |
| 7 | `linkedin_vjepa_playground.mp4` | The playground demo — real vs pretend |

## Tips for recording

- **The LLM comparison is your anchor** — keep coming back to "LLMs predict words, pixel models predict pixels, JEPA predicts meaning"
- **The playground video is your closer** — it's personal, visual, and the "faking it" moment is the wow
- **Let the t-SNE GIF breathe** — pause and let people watch the clusters form before explaining
- **Keep the pixel vs representation contrast visual** — point at the blurry MAE reconstruction vs the clean JEPA activation map