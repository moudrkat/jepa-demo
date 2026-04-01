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
| `linkedin_vl_jepa_post.webm` | WEBM | Screencast of Delong Chen's VL-JEPA LinkedIn post + video |

---

## The scenario

### COLD OPEN — The LinkedIn post that started this (30-60s)

*Play screencast `outputs/linkedin_vl_jepa_post.webm`*

**Source:** [Delong Chen's VL-JEPA post (Dec 2025)](https://www.linkedin.com/posts/chendelong_vl-jepa-joint-embedding-predictive-architecture-activity-7406715572022865920-xdSr)

**Say:**
"I saw this post from Delong Chen — he's one of the researchers working with Yann LeCun — and it stopped me in my tracks. They released VL-JEPA, a vision-language model, and the key line is this: instead of predicting the next token like ChatGPT does, it **predicts continuous embeddings** — abstract representations of what the text *means*.

And I thought: okay, I need to explain why that matters. Because this is not just a technical detail — this is a fundamentally different way of thinking about AI. And it turns out there's a whole *family* of models behind this idea. So let me take you through it — from a single image all the way to robots."

---

### INTRO — Everyone knows LLMs, but... (1-2 min)

**Say:**
"So everyone knows ChatGPT. You type a question, it predicts the next word. And the next. And the next. That's how large language models work — they predict the next token in a sequence. And it's incredibly powerful for text.

But here's the thing — when you try to do the same with images and video, predicting the next *pixel*... it doesn't work as well. Imagine trying to predict the exact shade of green of the next pixel of grass. Or the exact reflection on a car hood. That's insanely hard, and most of it is meaningless noise.

This is where JEPA comes in — Joint Embedding Predictive Architecture. It's Yann LeCun's answer to this problem, and the idea is beautifully simple: **don't predict pixels. Predict meaning.**

And this one idea spawned a whole family of models — each one pushing further:

- **I-JEPA** — understands images. Knows a cat is like a dog.
- **V-JEPA** — understands video. Knows what actions are happening.
- **VL-JEPA** — understands video *and* language. Can talk about what it sees.
- **V-JEPA 2.1** — understands video precisely enough to **control a robot**.

Same core idea, four directions. Let me show you how it builds up."

---

### ACT 1 — Masking: fill-in-the-blank for images (2-3 min)

*Model: **I-JEPA** — images only*

![MAE vs JEPA: masking and prediction](outputs/02_mae_vs_jepa.png)

**Say:**
"So how do these models learn without anyone labeling the images? It's basically a fill-in-the-blank exercise — but only during *training*. You take an image, split it into a grid of small patches — like tiles — and you *hide* some of them. Then you tell the model: guess what's missing. If it can guess correctly over millions of images, it must have learned something about how the visual world works. No labels needed.

Once it's trained, the masking is gone — you give it a complete image or video and it just understands it. Like a student who practiced fill-in-the-blank to learn English, but now reads whole books.

That training trick is called *masking* — and here's where it gets interesting: *what* you hide and *what* you ask the model to predict — that changes everything."

![MAE vs JEPA zoomed on a single cat image](outputs/07_patch_zoom.png)

"Look at this cat. The approach on the left — MAE, Masked Autoencoder — hides random patches scattered all over the image. And then it asks: reconstruct the exact pixels. What color was that fur? What shade was that shadow? The result? Blurry. It gets the rough shape but wastes enormous effort on noise — exact textures, lighting, color gradients. And because the hidden patches are scattered, the model can often cheat by just blending nearby colors.

I-JEPA does it differently. It hides *large blocks* — not random scattered patches. If the entire bottom half of the image is gone, you can't just interpolate from neighbors. You have to actually *think*: 'I see a cat's head up here, so there should be a body down there.'

And then the crucial part: I-JEPA doesn't even try to predict pixels. It predicts a *representation* — an abstract description. Not 'these specific brown pixels' but 'this is part of a cat.'"

![Full side-by-side: MAE pixel reconstruction vs JEPA feature activation](outputs/07_mae_vs_jepa.png)

"Here's the same comparison on four images — cat, dog, airplane, ship. The middle column shows what MAE reconstructed — blurry pixels. The right column shows what I-JEPA focuses on — see how it lights up on the meaningful parts? The cat's face, the dog's body, the airplane's shape. It ignores the background noise entirely.

LLMs predict the next *word*. Pixel models predict the next *pixel*. JEPA predicts the next *concept*."

---

### ACT 2 — What does I-JEPA actually learn? The representation space (2-3 min)

*Model: **I-JEPA** — images only*

![t-SNE animation — clusters forming from random noise](outputs/08_tsne_animation.gif)

**Say:**
"Okay, but does this actually work? Let me show you. I took 300 images — cats, dogs, airplanes, trucks, ships — and ran them through I-JEPA. The model has *never seen labels*. Nobody told it what a cat is. Nobody told it what a ship is.

Each dot here is one image. Watch what happens when we visualize the model's internal representation space..."

*Let the GIF play — dots organize into clusters.*

"Animals on one side, vehicles on the other. Cats near dogs. Trucks near cars. The model organized all of this by itself, just from learning to predict representations."

![t-SNE with actual image thumbnails](outputs/08_tsne_thumbnails.png)

"And here's the same thing with actual thumbnails so you can see it's real. Horses together, ships together, cars together. The model doesn't know the *word* 'horse' — but it knows these images share something fundamental.

This is what representation space looks like. Instead of storing pixels, the model builds an internal map of *concepts*. Similar things live close together. Different things live far apart. It's like the model built its own mental dictionary — without anyone teaching it the words."

---

### ACT 3 — From images to video: V-JEPA (2-3 min)

*Model: **V-JEPA** → **V-JEPA 2** — video understanding*

![Cluster journey: video + dot moving through representation space](outputs/10_cluster_journey.gif)

**Say:**
"Now here's where it gets really interesting. The same idea — predict representations, not pixels — but applied to video. That's V-JEPA. And in video, the representation space isn't just about *what things look like*, it's about *what's happening*.

Watch this. On the left, the actual video playing — someone pouring, folding, stacking. On the right, where the model's representation is in that abstract space. Every time the action changes, the dot *jumps* to a completely different region. Pouring lives here, folding lives there. The model discovered the structure of actions — no labels, no supervision.

And the next version, V-JEPA 2, pushed this further — not just recognizing actions, but *predicting* what happens next and *planning*. Which is exactly what I wanted to test."

---

### ACT 4 — The playground: can it tell real from fake? (2-3 min)

*Model: **V-JEPA 2** — the playground experiment*

*Play `outputs/linkedin_vjepa_playground.mp4`*

**Say:**
"So I had to try this myself. I took V-JEPA 2 to a playground — literally, my kids' toys on the floor — and I tested something: can it tell the difference between a *real* action and a *pretended* one?

Watch. Same hand, same toy, same background. In one video I actually put a ball into a bag. In the other I just pretend to — same motion, but I don't let go. To a pixel-prediction model, these look almost identical. Same colors, same shapes, same movement.

But V-JEPA 2? It knows. It sees through the fake. Because it's not comparing pixels — it understood the *intent*. The ball ended up somewhere different, and in representation space, that matters.

And then I let it run on a longer video — sixty seconds of my kids just playing — and it segments the whole thing into actions in real time. No prompting, no tricks. It just... understands what's happening."

---

### CLOSING — The family tree, and where it's going (1-2 min)

**Say:**
"So let's zoom out. We started with one idea — predict representations, not pixels — and look where it went:

**I-JEPA** learned to understand images. It knows a cat is like a dog, not like a truck. That's step one.

**V-JEPA** took that to video. It understands *actions* — pouring, folding, stacking — and it organizes them in representation space, all without labels.

**V-JEPA 2** went deeper — prediction and planning. It can tell when you're faking an action. It sees the *intent*, not just the motion.

**VL-JEPA** — that's the post I showed you at the beginning — adds *language*. Same idea: don't predict the next token, predict the *meaning* of the text. Now it can understand video and talk about it.

And the latest one, **V-JEPA 2.1**, from just a few weeks ago? It makes the representations so spatially precise that a robot can use them to *pick up objects*. Twenty points better at grasping than the previous version. Same idea — predict meaning — but now precise enough to guide a hand.

One core idea. Images, video, language, robotics. That's the JEPA family.

This is Yann LeCun's bet on how we get to real world models — AI that doesn't just generate text, but actually *understands* what it sees. And after taking it to a playground with my kids... I think he might be right."

---

## Quick reference: show order

| # | Visual | When |
|---|--------|------|
| 0 | `linkedin_vl_jepa_post.webm` | Cold open — screencast of the post that sparked this |
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

---

## TODO

- [ ] Embed own playground clips into the Something-Something-V2 representation space (from ACT 3) and see if they land near matching actions — bridge between ACT 3 and ACT 4

---

## References

### Papers used in this podcast

| # | Model | Paper | Link |
|---|-------|-------|------|
| 1 | **I-JEPA** | Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023) | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| 2 | **MAE** (contrast) | He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021) | [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) |
| 3 | **V-JEPA** | Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video" (2024) | [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) |
| 4 | **V-JEPA 2** | Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" (2025) | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |

### Referenced in cold open

| # | Model | Paper | Link |
|---|-------|-------|------|
| 5 | **VL-JEPA** | Chen et al., "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language" (2025) | [arXiv:2512.10942](https://arxiv.org/abs/2512.10942) |

### Datasets used in demos

| # | Dataset | Used in | Link |
|---|---------|---------|------|
| 6 | **ImageNet** | I-JEPA was trained on it (self-supervised, no labels used) | [image-net.org](https://www.image-net.org/) |
| 7 | **STL-10** | t-SNE visualisation (ACT 2) — 300 images the model never saw during training | [cs.stanford.edu/~acoates/stl10](https://cs.stanford.edu/~acoates/stl10/) |
| 8 | **Something-Something-V2** | Cluster journey (ACT 3) + V-JEPA 2 fine-tuning for playground classification | [20bn/something-something-v2](https://developer.qualcomm.com/software/ai-datasets/something-something) |

### Background / further reading

| # | Model | Paper | Link |
|---|-------|-------|------|
| 9 | JEPA concept | LeCun, "A Path Towards Autonomous Machine Intelligence" (2022) | [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf) |
| 10 | **V-JEPA 2.1** | Mur-Labadia et al., "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning" (2026) | [arXiv:2603.14482](https://arxiv.org/abs/2603.14482) |
| 11 | **CLIP** | Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021) | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |

### V-JEPA vs VL-JEPA — what's the difference?

- **V-JEPA** (Video JEPA) — self-supervised, learns from video only, predicts visual representations. No language. This is what the playground demo uses.
- **VL-JEPA** (Vision-Language JEPA) — extends JEPA to text, predicts continuous embeddings of language instead of generating tokens autoregressively. Combines vision + language.
- The podcast demos use **V-JEPA 2**. The cold open references Chen's **VL-JEPA** post as the hook — "they're now doing this with language too, but let me show you how it works with just vision first."