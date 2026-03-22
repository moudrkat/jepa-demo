# JEPA Through the Eyes of a Physicist

## The Familiar Problem

You have a system with an enormous number of degrees of freedom. The microscopic
dynamics are intractable — there are too many variables, too many possible
configurations, and most of the fine-grained detail is irrelevant to the
quantities you actually care about.

In physics, this is every statistical mechanics problem you've ever solved.

In machine learning, this is every image and every video. A 224×224 RGB image
has 150,528 degrees of freedom (pixels). A 2-second video clip at 30 fps has
millions. Most of those degrees of freedom encode noise, lighting accidents,
compression artifacts — things that carry no information about what's actually
happening in the scene.

The question is the same in both domains: **how do you find the right
description at the right level of abstraction?**

## LeCun's Energy-Based Framework

The connection between JEPA and physics is not an analogy imposed from outside.
LeCun builds his entire framework on energy-based models, and he is explicit
about the physics roots.

In his foundational paper **"A Tutorial on Energy-Based Learning"** (LeCun,
Chopra & Hadsell, 2006), he defines the core machinery:

- An **energy function** $E(x, y)$ over input-output pairs — low energy for
  compatible pairs, high energy for incompatible ones
- The **Gibbs-Boltzmann distribution** to convert energy to probability:
  $P(Y|X) = e^{-\beta E(Y,X)} / Z$
- A parameter $\beta$ he calls "an arbitrary positive constant akin to an
  **inverse temperature**"
- A normalizing constant he calls "**the partition function, by analogy with
  similar concepts in statistical physics**"

This isn't a loose metaphor. The mathematical framework is identical to
statistical mechanics. LeCun chose it deliberately.

He returned to this framework in depth at the **Les Houches Summer School on
Statistical Physics and Machine Learning** (2022), where the resulting lecture
notes — **"Introduction to Latent Variable Energy-Based Models"** (Dawid &
LeCun, 2023) — were published in the *Journal of Statistical Mechanics: Theory
and Experiment*. In these notes he explicitly states that marginalization over
latent variables leads to "a formula known from statistical physics, where $F$
is the **free energy**" — using the term in its precise thermodynamic sense.

The energy-based viewpoint is not a side remark in LeCun's work. It is the
foundation. JEPA is an energy-based architecture.

## JEPA as Learned Coarse-Graining

JEPA (Joint Embedding Predictive Architecture) is a specific energy-based
architecture where prediction happens entirely in latent space.

The **encoder** maps the high-dimensional input (pixels) into a
lower-dimensional latent space:

$$
z = f_\theta(x) \quad \text{where } x \in \mathbb{R}^{150528}, \; z \in \mathbb{R}^{1024}
$$

The **predictor** operates entirely in this latent space — it never sees pixels:

$$
\hat{z}_{\text{target}} = g_\phi(z_{\text{context}}, \text{position})
$$

The training objective is to minimize the prediction error in latent space:

$$
\mathcal{L} = \| \hat{z}_{\text{target}} - z_{\text{target}} \|^2
$$

The critical insight: **the encoder and the prediction objective are learned
jointly.** The system doesn't just compress — it discovers which degrees of
freedom are predictable and retains exactly those. Information that can't be
predicted from context (the exact texture of a leaf, the precise RGB value of a
shadow) gets discarded automatically because it can't reduce the loss.

This is coarse-graining not by a physicist's intuition about what matters, but by
a mathematical criterion: **keep what's predictable, discard what's not.**

### The Analogy to Renormalization

The connection between deep learning and the renormalization group (RG) was
first made precise by **Mehta & Schwab (2014)** in "An exact mapping between the
Variational Renormalization Group and Deep Learning." They showed that
restricted Boltzmann machines performing unsupervised learning on Ising model
configurations implement a mapping that is *exactly* equivalent to a
variational RG scheme. This is not a LeCun result — but it establishes
that the structural similarity between deep representation learning and RG
is mathematically grounded, not just a vague analogy.

In Wilsonian RG, you integrate out short-wavelength fluctuations to obtain an
effective theory at longer wavelengths. The procedure systematically discards
high-frequency modes that don't contribute to long-range physics. What remains
is a description that's simpler but captures the essential structure — the
universality class, the critical exponents, the phase diagram.

JEPA's encoder does something structurally similar:

| Renormalization Group | JEPA Encoder |
|---|---|
| Integrate out short-wavelength modes | Discard pixel-level detail |
| Retain long-range correlations | Retain semantic structure |
| Effective theory at coarser scale | Latent representation |
| Universality: different microscopic systems, same macroscopic behavior | Transfer: different images, same representation for same concept |

The **transfer learning** results make this concrete. I-JEPA trained on ImageNet
(everyday objects) produces clean semantic clusters when applied to flower
photographs it has never seen. A sunflower and a daisy both activate "radial
petal pattern" in representation space, even though their pixel-level details are
completely different — different microscopic configurations mapping to the same
macroscopic class.

### What Gets Thrown Away (And Why That's the Point)

Consider what a generative model (MAE) must do: reconstruct every masked pixel.
This is equivalent to building a microscopic simulator — you need the full
microstate. If a leaf has a particular vein pattern, the model must reproduce
it. If there's a shadow with a specific gradient, the model must get it right.

The capacity the model spends modeling these details is capacity not spent on
understanding what the image *means*.

JEPA sidesteps this entirely. By predicting in representation space, it is free
to build a description where a leaf is just "a leaf" — not a specific arrangement
of green pixels. The exact vein pattern, the precise shadow, the JPEG
compression artifacts: all integrated out. Gone. Irrelevant.

This is why JEPA achieves better downstream performance with less compute. It's
not doing more with less — it's doing *less*, and that less is exactly the right
thing. Just as a physicist doesn't need to track every molecule to predict when
water boils, JEPA doesn't need to predict every pixel to understand what's in an
image.

## The Energy Landscape and Collapse

### Shaping the Potential

In LeCun's framework, the model's job is to shape the energy function $E(x,y)$
so that the landscape has the right structure — deep wells for correct
associations, barriers between incorrect ones. This is the same problem as
designing an effective potential in physics.

The Gibbs-Boltzmann distribution connects the energy to probabilities:

$$
P(y|x) = \frac{e^{-\beta E(x,y)}}{Z(x)}, \quad Z(x) = \int e^{-\beta E(x,y)} \, dy
$$

A key insight from LeCun's framework: you don't always need to compute the
partition function $Z$. Energy-based models are *more general* than probabilistic
models precisely because they can work with un-normalized energies. In many
cases, you only need to find energy minima — not integrate over the full
landscape.

### The Collapse Problem: A Flat Potential

The fundamental failure mode of energy-based models is **collapse**: the model
learns to assign low energy everywhere. In physics terms, this is a flat
potential — no structure, no barriers, no wells. $E(x,y) = \text{const}$
trivially satisfies "low energy for valid pairs" because *everything* has low
energy. But the representation is useless.

Different approaches deal with this differently:

**Contrastive learning** (SimCLR, CLIP) explicitly pushes up the energy on
negative samples — "this image and this wrong caption should have HIGH energy."
You're manually sculpting the repulsive part of the potential. It works, but
requires carefully choosing negatives and scales poorly — you're defining your
potential point by point.

**JEPA's solution** is the asymmetric architecture with EMA. No negative samples
needed. No explicit repulsion. The collapse is prevented by *architecture*, not
by data.

### EMA: The Slowly-Evolving Potential

The target encoder in JEPA is updated as an exponential moving average of the
context encoder:

$$
\theta_{\text{target}} \leftarrow \alpha \, \theta_{\text{target}} + (1 - \alpha) \, \theta_{\text{context}}, \quad \alpha = 0.996
$$

with a stop-gradient: no backpropagation flows through the target encoder.

The target encoder defines the potential landscape that the context encoder is
learning within. That landscape evolves, but slowly — 0.4% per step. The
context encoder can track it, find the structure, settle into meaningful
representations.

If you updated the target encoder instantly ($\alpha = 0$), both encoders would
be identical and could collapse together — a potential that deforms faster than
the system can respond. If you never updated it ($\alpha = 1$), the targets
would be frozen and meaningless — a static potential that doesn't improve.

The EMA interpolates between these extremes: the potential changes slowly enough
that the system always approximately tracks the evolving equilibrium, but fast
enough that the equilibrium improves over time. Physicists will recognize
this pattern from simulated annealing, Born-Oppenheimer molecular dynamics,
and adiabatic quantum computation — all rely on separating fast and slow
dynamics.

## Loss Surfaces as Spin-Glass Hamiltonians

LeCun has co-authored work that makes the physics connection fully explicit at
the level of mathematical proof.

**"The Loss Surfaces of Multilayer Networks"** (Choromanska, Henaff, Mathieu,
Ben Arous & LeCun, 2015) maps the loss function of a neural network to the
**Hamiltonian of a spherical spin-glass model**. Using random matrix theory from
statistical mechanics, they show that:

- Critical points (local minima, saddle points) of the loss form a layered
  structure analogous to the energy levels of a spin glass
- For large networks, local minima concentrate near the global minimum —
  explaining why gradient descent works despite the non-convex landscape
- The index (number of negative eigenvalues of the Hessian) decreases as you
  go deeper in the loss landscape

This is genuine statistical physics applied to deep learning, not analogy.

In the follow-up **"Entropy-SGD: Biasing Gradient Descent Into Wide Valleys"**
(Chaudhari, Choromanska, Soatto, LeCun et al., 2017), they use **Langevin
dynamics** — a stochastic differential equation from statistical mechanics — to
bias optimization toward wide, flat minima in the energy landscape. The
algorithm explicitly computes a **local entropy** (the log-volume of
low-energy configurations near a point) to favor solutions that are robust to
perturbation.

## The Latent Space of V-JEPA

When V-JEPA processes video, it extracts a sequence of embeddings — one per
temporal window. This sequence traces a path through a learned representation
space.

### What We Observe: Clusters and Trajectories

Run V-JEPA on a video of someone pouring water, then folding paper, then
opening a bottle. The embedding path moves through three distinct regions
of latent space. If you project with t-SNE or UMAP, these regions are visibly
separated — clusters with clear boundaries.

Within a cluster, small perturbations (different camera angle, different hand,
different cup) don't change the cluster membership — the path stays in the same
region. The transitions between actions — the moment the hand stops pouring and
starts folding — correspond to the path crossing between regions.

### Why This Happens

The V-JEPA training objective — predict the representation of a masked
spatio-temporal region from visible context — forces the encoder to build a
space where temporal neighbors have similar representations (otherwise
prediction would fail). Actions that maintain consistent dynamics (pouring
continues to look like pouring) produce representations that stay in the same
neighborhood. Action boundaries, where dynamics change abruptly, produce
jumps.

The model was never told about actions, categories, or dynamics. The cluster
structure emerges because **temporal predictability in latent space requires it**.
Representations that group similar dynamics together are precisely the ones that
minimize the prediction loss.

### The Physics Reading

A physicist looking at these embedding trajectories will naturally see:

- The clusters as **basins** in an energy landscape
- The transitions as **barrier crossings**
- The smooth within-cluster evolution as motion within a potential well
- Repetitive actions as approximately **periodic orbits**
- The latent dimensions as **slowly-varying quantities** that stay approximately
  constant within an action and change between actions

These are observations about the geometry of the learned space, not claims about
the training algorithm. But they're also not coincidences. The energy-based
framework that JEPA is built on — with its explicit energy functions,
Gibbs-Boltzmann distributions, and collapse prevention — produces
representations whose geometry mirrors the structures physicists use to
describe dynamical systems. The reason may be that both are solutions to the
same underlying problem: finding the right low-dimensional description of a
high-dimensional system.

## Why Prediction in Latent Space Changes Everything

Consider predicting the next frame of a video in pixel space. A ball is flying
through the air. Where will it be in the next frame?

A physicist would say: that depends on the velocity and forces. Given those,
the prediction is deterministic (or at least narrowly constrained).

A generative model working in pixel space faces a different problem. Even if
the ball's trajectory is deterministic, the *pixels* are not. The ball might be
lit differently. A cloud might cast a shadow. The background might shift due to
camera jitter. The JPEG compression might differ. There are infinitely many
pixel-level "next frames" that are all consistent with the same physical
trajectory.

This is the **problem of irrelevant degrees of freedom** contaminating the
prediction. In pixel space, you can't separate "where does the ball go?" from
"what exact shade of shadow does the table cast?" They're entangled in the same
representation.

JEPA resolves this by predicting in a learned space where irrelevant degrees of
freedom have already been integrated out. The ball's position, velocity, and
interaction with other objects are retained. The shadow, the compression
artifacts, the background clutter are gone.

This is the same principle that makes generalized coordinates more powerful than
Cartesian coordinates for constrained systems. By choosing the right variables,
the constraints become invisible and the dynamics become simple. JEPA learns
those variables from data.

In physics, finding the right variables is often the hardest and most creative
part of the problem. JEPA automates this: the "right variables" are the ones
that minimize prediction error in the self-supervised task. The encoder learns
a coordinate transformation from pixel space to a space where dynamics are
simple and predictable.

## The Bigger Picture: World Models

LeCun's stated goal — laid out in **"A Path Towards Autonomous Machine
Intelligence"** (2022) — is not image classification or video understanding.
It's **world models**: systems that learn an internal model of how the world
works and use it to plan, predict, and reason.

He writes that "many types of reasoning can be viewed as forms of **energy
minimization**" and that a value of $y$ "that defies the **laws of physics**
should result in a high energy value." The energy function *is* the world model:
it assigns low energy to states that are physically plausible and high energy to
states that are not.

JEPA is the representation-learning part of this program: learn the right state
space — the right variables, the right level of description. V-JEPA 2 begins to
address the next step: learning the dynamics within that space, demonstrated
through zero-shot robot planning and intuitive physics benchmarks.

The audacious bet is that self-supervised prediction in latent space can
discover not just what objects are, but how they behave. Not just actions,
but dynamics. Not just recognition, but understanding.

Whether this program succeeds is an open question. But the first results —
a model that discovers the structure of human action, organizes it into a
meaningful geometry, and generalizes to unseen scenarios without any labels —
suggest that building machine intelligence on the mathematics of energy and
statistical mechanics is not just aesthetically pleasing. It may be the right
foundation.

---

*This article accompanies the [JEPA Demo repository](.), specifically the
[temporal cluster analysis demo](demos/05_vjepa_cluster_analysis.py) and the
[phase-space visualization demo](demos/06_vjepa_phase_space.py), which make
these connections visible.*

### References

**LeCun's energy-based framework:**
- LeCun, Y., Chopra, S. & Hadsell, R. "A Tutorial on Energy-Based Learning." *Predicting Structured Data*, MIT Press, 2006. [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
- Dawid, A. & LeCun, Y. "Introduction to Latent Variable Energy-Based Models: A Unifying Perspective." *J. Stat. Mech.* 2024, 104011. Lecture notes from the Les Houches Summer School on Statistical Physics and Machine Learning, 2022. [arXiv:2306.02572](https://arxiv.org/abs/2306.02572)
- LeCun, Y. "A Path Towards Autonomous Machine Intelligence." 2022. [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- LeCun, Y. "Intriguing Connections Between Deep Learning and Physics." Morris Loeb Lectures, Harvard Physics Department, 2019.

**Loss landscapes and statistical mechanics (LeCun co-author):**
- Choromanska, A., Henaff, M., Mathieu, M., Ben Arous, G. & LeCun, Y. "The Loss Surfaces of Multilayer Networks." AISTATS 2015. [arXiv:1412.0233](https://arxiv.org/abs/1412.0233)
- Chaudhari, P., Choromanska, A., Soatto, S., LeCun, Y. et al. "Entropy-SGD: Biasing Gradient Descent Into Wide Valleys." ICLR 2017. [arXiv:1611.01838](https://arxiv.org/abs/1611.01838)

**The JEPA architecture:**
- Assran, M. et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- Bardes, A. et al. "Revisiting Feature Prediction for Learning Visual Representations from Video." 2024. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
- Bardes, A. et al. "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." 2025. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)

**Deep learning and renormalization group (other authors):**
- Mehta, P. & Schwab, D. J. "An exact mapping between the Variational Renormalization Group and Deep Learning." 2014. [arXiv:1410.3831](https://arxiv.org/abs/1410.3831)