# JEPA Demo — Joint Embedding Predictive Architecture

A hands-on collection of demos for learning about and playing with Meta's **I-JEPA** (image) and **V-JEPA 2** (video) models. Run them yourself — no GPU needed.

> **Want the full story?** Read [ARTICLE.md](ARTICLE.md) — a visual deep-dive into how JEPA works, with inline results from every demo.
>
> **Heard the podcast?** See [PODCAST_SCENARIO.md](PODCAST_SCENARIO.md) for a map of what was covered and which demos go with each section.

## What is JEPA?

JEPA learns visual representations by **predicting abstract representations** of masked
image/video regions — not pixels. Unlike generative models (MAE, Sora), JEPA works entirely
in latent space, which lets it focus on high-level semantics rather than low-level details.

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

## See it in action

300 images, no labels — I-JEPA organizes them into semantic clusters by itself:

![t-SNE animation — clusters forming from random noise](outputs/08_tsne_animation.gif)

V-JEPA 2 traces a path through representation space as actions change in a video:

![Cluster journey — dot moves through action space](outputs/10_cluster_journey.gif)

## Demos

### Core demos

| # | Script | What it shows |
|---|--------|---------------|
| 01 | `demos/01_ijepa_representations.py` | I-JEPA feature extraction, t-SNE clustering, similarity retrieval |
| 02 | `demos/02_ijepa_masking_explained.py` | How I-JEPA's multi-block masking works (pure visualization, no model) |
| 03 | `demos/03_vjepa_video_classify.py` | V-JEPA 2 action recognition on hand-object clips |
| 04 | `demos/04_vjepa_action_anticipation.py` | Watch confidence rise as more of a video is revealed |
| 05 | `demos/05_vjepa_cluster_analysis.py` | Temporal clustering of actions with V-JEPA 2 (fine-tuned) |
| 06 | `demos/06_vjepa_cluster_pretrained.py` | Same clustering without fine-tuning — what self-supervised learning alone captures |

### Visualizations

| # | Script | What it shows |
|---|--------|---------------|
| 07 | `demos/07_mae_vs_jepa_comparison.py` | Side-by-side: MAE pixel reconstruction vs JEPA abstract representations |
| 08 | `demos/08_animated_tsne.py` | Animated t-SNE — watch clusters form from random noise |
| 09 | `demos/09_vjepa_video_gif.py` | V-JEPA 2 live prediction bar chart updating frame by frame |
| 10 | `demos/10_vjepa_cluster_journey.py` | Video + dot tracing through representation space side by side |

### Try it yourself

| # | Script | What it shows |
|---|--------|---------------|
| 11 | `demos/11_your_own_video.py` | Feed any video into V-JEPA 2, get progressive predictions |
| 12 | `demos/12_your_own_latent_space.py` | Feed a long video, get temporal clustering and a journey through latent space |

### Utilities

| Script | What it does |
|--------|-------------|
| `demos/classify_batch.py` | Batch-classify a list of videos |
| `demos/classify_sliding.py` | Sliding-window classification across a long video |

## Playground Dataset

We recorded 89 short clips of hand-object interactions with children's toys and classified them with V-JEPA 2.
The dataset is available as a [GitHub release](https://github.com/moudrkat/jepa-demo/releases/tag/playground-dataset-v1) (22 MB, 256x256 videos + metadata CSV).

```bash
gh release download playground-dataset-v1 -p "playground-dataset-256.zip"
```

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
Demos 03+ download V-JEPA 2 weights (~1-2 GB) and sample videos from Something-Something V2.
Models are cached in `~/.cache/huggingface/`.

## Running

```bash
make run-01   # I-JEPA representations
make run-02   # I-JEPA masking explainer
make run-03   # V-JEPA 2 video classification
make run-04   # V-JEPA 2 action anticipation
make run-05   # V-JEPA 2 temporal clustering (fine-tuned)
make run-06   # V-JEPA 2 temporal clustering (pretrained)
make run-all  # Run everything
```

Or run scripts directly: `python demos/01_ijepa_representations.py`

Results are saved to `outputs/`.

## Hardware

- Works on **CPU** (no GPU required)
- ~6 GB RAM recommended
- Most demos run in 2-5 minutes

## Ideas & contributions welcome

Some directions we'd love to explore — PRs, issues, and half-baked ideas all welcome:

- **Real-time webcam demo** — narrate actions live, or turn it into a game ("can you fool the model?")
- **V-JEPA + LLM combo** — feed action predictions into an LLM for natural-language commentary
- **Embedding interpolation** — smooth trajectories through representation space (see [ARTICLE.md](ARTICLE.md))
- **Interactive masking playground** — draw masks on images, see what the model predicts

Have a different idea? [Open an issue](../../issues) — we'd love to hear it.

```bash
make lint               # Check code style (ruff)
make format             # Auto-format code
make help               # Show all available targets
```

## Further reading

- [ARTICLE.md](ARTICLE.md) — visual deep-dive into how JEPA works, with inline demo results
- [FAQ.md](FAQ.md) — from basics to deep theory, with visuals
- [JEPA_VS_PHYSICS.md](JEPA_VS_PHYSICS.md) — JEPA through the lens of statistical physics *(generated by Claude, read with skepticism)*

## Papers

- **I-JEPA**: [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) (CVPR 2023)
- **V-JEPA**: [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) (2024)
- **V-JEPA 2**: [Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) (2025)

## License

MIT. Pretrained models by Meta FAIR under their respective licenses.