# JEPA Demo — Joint Embedding Predictive Architecture

A collection of visual demos showcasing Meta's **I-JEPA** (image) and **V-JEPA 2** (video) models.
Built for a video podcast presentation.

> **Want the full story?** Read [ARTICLE.md](ARTICLE.md) — a visual deep-dive into how JEPA works, with inline results from every demo.

## What is JEPA?

JEPA learns visual representations by **predicting abstract representations** of masked
image/video regions — not pixels. Unlike generative models (MAE, Sora), JEPA works entirely
in latent space, which lets it focus on high-level semantics rather than low-level details.
Unlike contrastive methods (SimCLR, DINO), it doesn't need hand-crafted augmentations or
negative pairs.

```
  Image/Video
       │
       ▼
 ┌───────────┐        ┌───────────┐
 │  Context   │        │  Target   │
 │  Encoder   │        │  Encoder  │  (EMA, no gradients)
 │   (ViT)    │        │   (ViT)   │
 └─────┬──────┘        └─────┬─────┘
       │                     │
       ▼                     ▼
 ┌───────────┐         target patch
 │ Predictor │         embeddings
 │ (small    │── L2 ──▶ (stop-grad)
 │  ViT)     │  loss
 └───────────┘
```

**Key insight**: By predicting *representations* instead of pixels, JEPA ignores unpredictable
low-level details (exact textures, lighting) and learns *what matters* — object identity,
spatial relationships, motion dynamics.

## Demos

| # | Script | What it shows | Status |
|---|--------|---------------|--------|
| 01 | `demos/01_ijepa_representations.py` | Load pretrained I-JEPA ViT-H/14, extract features from Flowers102, visualize t-SNE clustering, similarity retrieval, and pairwise heatmap | Tested |
| 02 | `demos/02_ijepa_masking_explained.py` | Visual explainer of I-JEPA's multi-block masking strategy — no model needed, pure visualization | Tested |
| 03 | `demos/03_vjepa_video_classify.py` | V-JEPA 2 action recognition on hand-object interaction clips (SSv2 — dipping, picking up, pushing) | Tested |
| 04 | `demos/04_vjepa_action_anticipation.py` | Progressive reveal — watch confidence rise as more of a hand-object video is shown | Tested |
| 05 | `demos/05_vjepa_cluster_analysis.py` | Temporal clustering — slide a window across concatenated action clips, cluster V-JEPA 2 embeddings with k-means, visualize with t-SNE | Tested |
| 06 | `demos/06_vjepa_cluster_pretrained.py` | Same clustering but with the **base pretrained** V-JEPA 2 (no fine-tuning) — tests what self-supervised learning alone captures | Tested |

## Setup

```bash
make setup
source .venv/bin/activate
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Demo 01 downloads the Flowers102 dataset (~350 MB) on first run and the I-JEPA ViT-H/14 weights from HuggingFace (~2.4 GB).
Demos 03/04 download V-JEPA 2 weights (~1-2 GB) and sample videos from Something-Something V2 (hand-object interactions).
Models are cached in `~/.cache/huggingface/`.

## Running

```bash
make run-01   # I-JEPA representations
make run-02   # I-JEPA masking explainer
make run-03   # V-JEPA 2 video classification
make run-04   # V-JEPA 2 action anticipation
make run-05   # V-JEPA 2 temporal clustering (fine-tuned)
make run-06   # V-JEPA 2 temporal clustering (pretrained, no fine-tuning)
make run-all  # Run everything
```

Or run scripts directly:

```bash
python demos/01_ijepa_representations.py
python demos/02_ijepa_masking_explained.py
python demos/03_vjepa_video_classify.py
python demos/04_vjepa_action_anticipation.py
python demos/05_vjepa_cluster_analysis.py
python demos/06_vjepa_cluster_pretrained.py
```

Results are saved to `outputs/`.

## Contributing

```bash
make setup              # One-time setup
make lint               # Check code style (ruff)
make format             # Auto-format code
make help               # Show all available targets
```

## Hardware

- Works on **CPU** (no GPU required)
- ~6 GB RAM recommended
- Demo 01: ~3 min (feature extraction on 200 images)
- Demo 02: ~5 sec (visualization only, no model)
- Demo 03: ~2-3 min (3 video clips, inference on each)
- Demo 04: ~3-5 min (4 progressive inference passes on one clip)

## Roadmap

**Embedding interpolation** — use the pretrained backbone to interpolate between action embeddings and visualize smooth trajectories through representation space, testing whether the self-supervised model learns a continuous manifold of physical states (see the interpolation discussion in [ARTICLE.md](ARTICLE.md)).

The big next step is a **real-time video demo** — something anyone can try with just a webcam:

- **Live action narrator** — point your webcam at your desk, pick up objects, push things around, and V-JEPA 2 narrates what you're doing in real time ("Taking one of many similar things on the table", "Pushing something so it almost falls off...")
- **V-JEPA + LLM combo** — feed V-JEPA 2's action predictions into a talking LLM that commentates your actions in natural language, like a live sports narrator for everyday tasks
- **"Can the AI tell what you're doing?"** — a game mode where you perform actions and try to fool the model, or match a target action it gives you
- **Webcam similarity search** — live I-JEPA features from your camera, matched against a dataset in real time
- **Interactive masking playground** — draw masks on images and see what the model predicts

If you have ideas for a killer use case, open an issue!

## Article

See [ARTICLE.md](ARTICLE.md) for a visual deep-dive into how JEPA works — with inline images from the demo outputs.

See also [JEPA_VS_PHYSICS.md](JEPA_VS_PHYSICS.md) — a companion essay exploring JEPA through the lens of statistical physics, renormalization, and coarse-graining. *(Note: this essay was generated by Claude and has not been fully reviewed — read with appropriate skepticism.)*

## Papers

- **I-JEPA**: [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) (CVPR 2023)
- **V-JEPA**: [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) (2024)
- **V-JEPA 2**: [Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) (2025)

## Credits

Pretrained models by Meta FAIR. Architecture by Yann LeCun, Mahmoud Assran, Adrien Bardes et al.
