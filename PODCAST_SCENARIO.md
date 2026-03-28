# JEPA Podcast Scenario

**Topic:** Why JEPA is different and where it's useful
**Format:** Video podcast with screen-shared visuals
**Duration:** ~15-20 minutes
**Visuals:** All in `outputs/` folder, ready to show

---

## Visual inventory

| File | Type | What it shows |
|------|------|---------------|
| `08_tsne_animation.gif` | GIF | Random dots organizing into semantic clusters |
| `09_vjepa_live_prediction.gif` | GIF | Single video with live prediction bars |
| `09_vjepa_multi_video.gif` | GIF | 3 videos classified simultaneously |
| `02_method_comparison.png` | PNG | Three paradigms: Contrastive vs Generative vs JEPA |
| `02_architecture.png` | PNG | I-JEPA architecture diagram |
| `02_mae_vs_jepa.png` | PNG | MAE reconstructs pixels vs JEPA predicts representations |
| `02_masking_on_images.png` | PNG | How JEPA masking works on real images |
| `07_mae_vs_jepa.png` | PNG | Side-by-side MAE vs JEPA on cat, dog, airplane, ship |
| `07_patch_zoom.png` | PNG | Zoomed single-image comparison |
| `08_tsne_final.png` | PNG | Final t-SNE clusters with labels |
| `08_tsne_thumbnails.png` | PNG | t-SNE with actual image thumbnails |
| `04_progressive.png` | PNG | Confidence growing as video progresses |
| `05_timeline.png` | PNG | Action boundaries discovered without labels (fine-tuned) |
| `06_timeline.png` | PNG | Same but pretrained — still finds structure |

---

## The scenario

### INTRO — The hook (1-2 min)

> **Show:** `08_tsne_animation.gif`

**Say:**
"So here's something wild. I took 300 random images — cats, dogs, airplanes, trucks, ships — and fed them through a model that has never been told what any of these things are. No labels. No human supervision. Just raw images during training.

And watch what happens when we visualize what the model learned..."

*Let the GIF play — dots start random, then organize into clean clusters.*

"Animals end up on one side, vehicles on the other. Cats near dogs, trucks near cars. The model figured this out on its own. This is JEPA — and today I want to show you why it's such a different approach to AI vision."

---

### ACT 1 — The problem: how do you learn without labels? (3-4 min)

> **Show:** `02_method_comparison.png`

**Say:**
"So the fundamental challenge: you have billions of images and videos on the internet. Labeling them is insanely expensive. So how do you get a model to learn useful things from unlabeled data? There are three main approaches.

**Contrastive** — SimCLR, DINO — you take the same image, crop it two ways, and say 'these should be similar, everything else should be different.' Works well, but you need careful tricks to avoid collapse, and you need negative pairs.

**Generative** — like MAE, Masked Autoencoders — you hide 75% of the image and make the model reconstruct the missing pixels. Simple and elegant. But here's the problem..."

> **Show:** `07_patch_zoom.png`

"Look at this cat. MAE masks random patches and tries to reconstruct the exact pixels. See the reconstruction on the right? It's blurry. It gets the rough shape but wastes enormous capacity trying to predict exact fur texture, exact lighting, exact color gradients. Stuff that's basically random noise at the pixel level.

**JEPA takes a completely different approach.** It says: don't predict pixels at all. Predict the *meaning*."

> **Show:** `07_mae_vs_jepa.png`

"Here's the same four images — cat, dog, airplane, ship — processed by both models. Left side: MAE, pixel reconstruction. Right side: JEPA's feature activation map — it shows *what the model pays attention to*.

See how JEPA lights up on the cat's face, the dog's body, the airplane's shape? It's not wasting capacity on background grass or sky texture. It's learning the *concept*."

---

### ACT 2 — How JEPA actually works (3-4 min)

> **Show:** `02_architecture.png`

**Say:**
"The architecture is surprisingly clean. You take an image, split it into patches — like a grid. Then you split those patches into two groups: context and targets.

The context encoder — a Vision Transformer — processes the visible patches. Then a small predictor network tries to predict *the representations* of the target patches. Not the pixels — the abstract features.

There's a clever trick here: the target encoder is an exponential moving average of the context encoder. It moves slowly, giving the predictor a stable target to aim for. Without this, both encoders would collapse to outputting the same thing for every image."

> **Show:** `02_masking_on_images.png`

"And notice the masking strategy. Unlike MAE which randomly drops patches everywhere, JEPA masks *contiguous blocks*. This forces the model to understand spatial relationships — what kind of thing *should* be in that region, not just interpolate from nearby pixels."

> **Show:** `02_mae_vs_jepa.png`

"So to really hammer this home: MAE asks 'what color are the missing pixels?' JEPA asks 'what *concept* belongs in the missing region?' That's a fundamentally different learning signal."

---

### ACT 3 — The proof: clusters from nothing (2-3 min)

> **Show:** `08_tsne_animation.gif` (replay)

**Say:**
"Back to this animation. This is I-JEPA — trained only on ImageNet — applied to STL-10, a completely different dataset it has never seen. Ten categories: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.

No fine-tuning. No labels. Just extract features and run t-SNE."

> **Show:** `08_tsne_thumbnails.png`

"And here's the same thing but with actual image thumbnails. You can see — the horses are together, the cars are together, ships are together. The model doesn't know the word 'horse.' But it knows that these images share something fundamental.

This is what learning in representation space gives you. The model builds an internal world model of visual concepts."

---

### ACT 4 — Video: where it gets really interesting (4-5 min)

> **Show:** `09_vjepa_live_prediction.gif`

**Say:**
"Now JEPA isn't just for images. V-JEPA 2 extends this to video — and this is where the practical applications explode.

Watch this GIF. A hand is dipping a brush into paint. The model sees more and more of the video, and the prediction bars update in real time. By halfway through, it already knows what's happening."

> **Show:** `09_vjepa_multi_video.gif`

"Here's three different actions side by side. And look at the granularity — the model doesn't just say 'a hand doing something.' It distinguishes between 'taking one of many similar things on the table' versus 'picking something up.' These are subtle differences that require understanding *intent*, not just motion."

> **Show:** `04_progressive.png`

"This chart makes it concrete. At 25% of the video — you've barely started watching — the model is already 64% confident. By 50%, it's at 99%. It's anticipating the action before it finishes.

Think about what this means for robotics, for autonomous systems. You don't need to wait for an action to complete to understand what's happening."

---

### ACT 5 — The killer demo: discovering structure without labels (2-3 min)

> **Show:** `05_timeline.png`

**Say:**
"This might be the most impressive result. We took 8 different action videos — pouring, folding, transferring, placing, and so on — concatenated them into one long sequence. Then we ran V-JEPA's embeddings through simple k-means clustering. No labels. No supervision.

Top bar: the ground truth — what action is actually happening at each moment. Bottom bar: what k-means discovered from JEPA's features alone.

The boundaries *align*. The model found where one action ends and another begins, purely from the structure of the representations."

> **Show:** `06_timeline.png`

"And here's the same experiment with the pretrained model — no fine-tuning at all, just pure self-supervised learning. It's noisier, but the temporal structure is still there. The model learns that 'pouring' is fundamentally different from 'folding' without anyone ever telling it what those words mean."

---

### CLOSING — Why this matters (2-3 min)

**Say:**
"So why should you care about JEPA? Three reasons.

**One: efficiency.** By not wasting capacity on pixel prediction, JEPA learns more useful features with less compute. The representations transfer better to new tasks.

**Two: video understanding.** This is the big one. Video is the next frontier — there's orders of magnitude more video data than image data, and most of it is unlabeled. JEPA can learn from all of it.

**Three: the path to world models.** This is Yann LeCun's vision — and he's the architect behind JEPA. If you want a model that understands how the world works, it needs to predict at the level of *concepts*, not pixels. You don't simulate every atom to predict that a ball will roll off a table. You reason at an abstract level. That's what JEPA does.

The immediate applications? Robotics — a robot that anticipates actions before they complete. Video surveillance — understanding events, not just detecting objects. Content understanding at scale — automatically segmenting and categorizing video without human labeling.

We're still early. But JEPA represents a genuine paradigm shift in how we think about self-supervised learning. Not pixels. Representations."

---

## Quick reference: show order

| # | Visual | When |
|---|--------|------|
| 1 | `08_tsne_animation.gif` | Hook — watch clusters form |
| 2 | `02_method_comparison.png` | Three paradigms |
| 3 | `07_patch_zoom.png` | MAE blurriness close-up |
| 4 | `07_mae_vs_jepa.png` | Full side-by-side comparison |
| 5 | `02_architecture.png` | How JEPA works |
| 6 | `02_masking_on_images.png` | Block masking strategy |
| 7 | `02_mae_vs_jepa.png` | Pixel vs representation prediction |
| 8 | `08_tsne_animation.gif` | Replay — proof of concept |
| 9 | `08_tsne_thumbnails.png` | Thumbnails at cluster positions |
| 10 | `09_vjepa_live_prediction.gif` | Live video prediction |
| 11 | `09_vjepa_multi_video.gif` | Three videos at once |
| 12 | `04_progressive.png` | Confidence evolution chart |
| 13 | `05_timeline.png` | Discovered action boundaries (fine-tuned) |
| 14 | `06_timeline.png` | Discovered action boundaries (pretrained) |

## Tips for recording

- **GIFs**: open them in a browser tab (they loop automatically) or use an image viewer that supports animation
- **PNGs**: full-screen them, zoom into details while talking
- **Pacing**: let the animated GIFs play for a few seconds before talking over them — the visual tells the story
- **The t-SNE GIF is your strongest opener** — it's visually striking and immediately shows the "wow" moment
- **The multi-video GIF is your strongest mid-section beat** — it shows practical utility in real time