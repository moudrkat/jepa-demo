# What was in the podcast

This repo was built alongside a podcast episode explaining the JEPA model family.
Here's a quick map of what was covered and which demos correspond to each part.

---

## The story arc

**News hook** — Meta released V-JEPA 2.1, which nearly doubled image segmentation
performance and improved robotic grasping by 10 points — same architecture, just
better training.

**The core idea** — LLMs predict the next token. Generative vision models (MAE, Sora)
predict the next pixel. JEPA predicts *representations* — abstract meaning, not
low-level details. That one idea spawned a whole model family.

**The JEPA family:**

| Model | What it does |
|-------|-------------|
| **I-JEPA** | Understands images — knows a cat is like a dog, not like a truck |
| **V-JEPA / V-JEPA 2** | Understands video — recognizes and predicts actions |
| **VL-JEPA** | Adds language — can talk about what it sees |
| **V-JEPA 2.1** | Precise enough to control a robot |

---

## Demo map

Each section of the podcast corresponds to demos you can run yourself:

### Representation space (I-JEPA)
*"300 images, no labels — and the model sorted cats next to dogs, trucks next to cars, all by itself."*

- `demos/01_ijepa_representations.py` — t-SNE clustering, similarity retrieval
- `demos/02_ijepa_masking_explained.py` — how the masking training works

### From images to video (V-JEPA)
*"Every time the action changes, the dot jumps to a completely different region."*

- `demos/05_vjepa_cluster_analysis.py` — temporal clustering of video actions
- `demos/06_vjepa_cluster_pretrained.py` — same clustering without fine-tuning

### The playground experiment (V-JEPA 2)
*"Same hand, same toy, same background. In one video I actually put a ball into a bag. In the other I just pretend to. V-JEPA 2 knows the difference."*

- `demos/03_vjepa_video_classify.py` — action recognition on hand-object clips
- `demos/04_vjepa_action_anticipation.py` — progressive reveal of confidence

---

## References

### Papers discussed

| Model | Paper | Link |
|-------|-------|------|
| **I-JEPA** | Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023) | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| **MAE** (contrast) | He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021) | [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) |
| **V-JEPA** | Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video" (2024) | [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) |
| **V-JEPA 2** | Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" (2025) | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |
| **VL-JEPA** | Chen et al., "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language" (2025) | [arXiv:2512.10942](https://arxiv.org/abs/2512.10942) |
| **V-JEPA 2.1** | Mur-Labadia et al., "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning" (2026) | [arXiv:2603.14482](https://arxiv.org/abs/2603.14482) |

### Datasets used in demos

| Dataset | Used in | Link |
|---------|---------|------|
| **STL-10** | t-SNE visualisation — 300 images the model never saw during training | [cs.stanford.edu/~acoates/stl10](https://cs.stanford.edu/~acoates/stl10/) |
| **Something-Something-V2** | Video action recognition + clustering | [20bn/something-something-v2](https://developer.qualcomm.com/software/ai-datasets/something-something) |

### Background

| Paper | Link |
|-------|------|
| LeCun, "A Path Towards Autonomous Machine Intelligence" (2022) | [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf) |