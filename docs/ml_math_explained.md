# ML, From Scratch — Space Disaster Mapper Study Notes

A ground-up explanation of every model, equation, and idea we used, written for
someone who *just* started machine learning. Read top to bottom. Equations render
in Obsidian/Cursor (they use LaTeX `$...$`).

---

## 0. What does "machine learning" even mean?

A **model** is just a function with tunable knobs:

$$ \hat{y} = f(x; \theta) $$

- $x$ = the input (a satellite image)
- $\hat{y}$ = the prediction (a water mask)
- $\theta$ = the **parameters** ("knobs") — numbers the model can change

**Learning** = automatically searching for the values of $\theta$ that make $\hat{y}$
match the true answer $y$ as closely as possible. That's the whole game. Everything
below is *how* we measure "closely" and *how* we search.

- A **heuristic** has $\theta$ fixed by a human (no learning).
- A **neural network** has millions of $\theta$ found by training.

---

## 1. Our data, as math

One satellite image is a 3-D block of numbers (a **tensor**):

$$ X \in \mathbb{R}^{C \times H \times W}, \quad C=13,\; H=W=512 $$

- $C = 13$ **channels** (Sentinel-2 sees 13 colors, including infrared we can't)
- $H, W$ = height, width in pixels

The label (mask) is one number per pixel — water or not:

$$ Y \in \{0, 1\}^{H \times W} $$

So **segmentation = classify every single pixel**. A 512×512 image is 262,144
tiny yes/no decisions. That's why we need something more powerful than a formula.

---

## 2. Baseline 1 — the NDWI heuristic (NO learning)

Before ML, remote-sensing scientists used physics. **Water absorbs near-infrared
(NIR) light but reflects green light.** So this index lights up over water:

$$ \text{NDWI} = \frac{G - \text{NIR}}{G + \text{NIR}} $$

(G = green band, NIR = near-infrared band; McFeeters, 1996.)

- Over water: $G$ high, NIR low → NDWI close to $+1$
- Over land/plants: NIR high → NDWI negative

Then we **threshold**: pixel is water if $\text{NDWI} > c$ (a cutoff like 0).

Dividing by $(G + \text{NIR})$ is **normalization** — it cancels out how bright the
whole scene is, so the index means the same thing in shadow or sunlight. No $\theta$
is learned here; a human picked the formula and cutoff. This is our "can a neural
net beat a smart guess?" reference.

---

## 3. The neuron — the atom of a neural network

A neuron takes inputs $x_1..x_n$, weights them, adds a bias, and squashes the result.

**Step 1 — weighted sum (linear part):**

$$ z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}\cdot\mathbf{x} + b $$

The weights $w_i$ and bias $b$ are the learnable knobs $\theta$.

**Step 2 — activation (non-linear part):** without this, stacking neurons would
just be one big linear equation (useless). Common choices:

$$ \text{ReLU}(z) = \max(0, z) \qquad \sigma(z) = \frac{1}{1+e^{-z}} $$

- **ReLU** keeps positives, zeros out negatives — cheap and works great in hidden layers.
- **Sigmoid** $\sigma$ squashes any number into $(0,1)$ — perfect for "probability."

A network is just **many** of these neurons wired in layers. "Deep learning" = many layers.

---

## 4. Convolution — why CNNs are made for images

A normal neuron connected to all 262,144 pixels would need 262,144 weights *per neuron*
— insane, and it would ignore that nearby pixels are related. **Convolution** fixes both.

We slide a small **kernel** (filter) $K$, e.g. $3\times3$, across the image and compute a
dot product at each spot:

$$ O[i,j] = \sum_{m}\sum_{n} X[i+m,\; j+n]\; K[m,n] \; + \; b $$

(Across multiple input channels, we also sum over channels.)

Two superpowers:
1. **Locality** — each output looks only at a small neighborhood (like the eye does).
2. **Weight sharing** — the *same* small kernel is reused everywhere, so we learn only
   $3\times3 = 9$ weights to detect an "edge" *anywhere* in the image. Tiny and powerful.

The kernel values are **learned**. Early layers learn edges; deeper layers combine
edges into textures, then into "this region looks like a river."

**Output size** after a convolution:

$$ H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1 $$

$k$ = kernel size, $p$ = padding (pixels added at borders), $s$ = stride (step size).

---

## 5. The U-Net — our actual model

U-Net is the standard network for segmentation. It's an **encoder–decoder** shaped
like a "U":

```
   INPUT (13×128×128)
        │
   ENCODER (contract)                DECODER (expand)
   conv+ReLU ─┐                        ┌─► upsample+conv ─► OUTPUT (1×128×128)
   maxpool    │  skip connection       │
   conv+ReLU ─┼──────────────────────► ┤
   maxpool    │  skip connection       │
        └────────── bottleneck ────────┘
```

**Encoder (downsampling)** — repeatedly convolve + **max-pool** to shrink the image
and distill *what* is present. Max-pooling takes the strongest response in each window:

$$ P[i,j] = \max_{(m,n)\in \text{window}} O[i+m,\; j+n] $$

Shrinking gives later layers a bigger "field of view" for cheap.

**Bottleneck** — smallest, most abstract representation ("there's water-like texture here").

**Decoder (upsampling)** — grow the image back to full size (via **transposed
convolution** / interpolation) so we can say *where* each pixel's label is.

**Skip connections** (the crossbars of the U) — copy fine detail from the encoder
directly to the matching decoder level. Without them, upsampling gives blurry blobs;
with them, water edges stay crisp. This is U-Net's key trick.

**Final layer** — a $1\times1$ convolution maps features to **one** number per pixel
(the logit $z$), then sigmoid turns it into a probability.

---

## 6. From logit to probability — sigmoid

The network's raw output per pixel is a **logit** $z \in (-\infty, \infty)$. Sigmoid maps it to a probability:

$$ p = \sigma(z) = \frac{1}{1+e^{-z}} \in (0,1) $$

- $z = 0 \Rightarrow p = 0.5$ (unsure)
- big positive $z \Rightarrow p \to 1$ ("water")
- big negative $z \Rightarrow p \to 0$ ("not water")

We call a pixel water if $p \ge 0.5$.

---

## 7. Loss — measuring "how wrong"

To learn, we need a single number for how bad a prediction is. For yes/no problems
we use **Binary Cross-Entropy (BCE)**. For one pixel with truth $y\in\{0,1\}$ and
predicted probability $p$:

$$ \mathcal{L} = -\big[\, y\log p + (1-y)\log(1-p) \,\big] $$

Read it as two cases:
- If truth $y=1$: loss $= -\log p$. Predict $p=1$ → loss 0. Predict $p\to0$ → loss $\to\infty$. (Confident and wrong = huge penalty.)
- If truth $y=0$: loss $= -\log(1-p)$. Symmetric.

Total loss = **average** over all pixels. (`BCEWithLogitsLoss` in PyTorch takes the
logit $z$ directly and applies sigmoid + BCE together for numerical stability.)

**The beautiful fact** (why this pairs with sigmoid): the gradient of BCE with respect
to the logit simplifies to just

$$ \frac{\partial \mathcal{L}}{\partial z} = p - y $$

"How much to nudge" = "prediction minus truth." Clean and intuitive. Remember this — it
explains our model's failure in §10.

---

## 8. Learning — gradient descent + backpropagation

We want $\theta$ that minimizes the loss. We can't try every combination (millions of
knobs), so we **roll downhill** on the loss surface.

**Gradient descent update rule:**

$$ \theta \leftarrow \theta - \eta \,\nabla_\theta \mathcal{L} $$

- $\nabla_\theta \mathcal{L}$ = the **gradient**, the direction of steepest *increase* of loss
- minus sign = go the *opposite* way (downhill)
- $\eta$ = **learning rate**, the step size (we used $10^{-3}$)

**Backpropagation** is how we get that gradient efficiently. The loss depends on the
output, which depends on the last layer, which depends on the previous layer… Using the
**chain rule** of calculus, we multiply the local derivatives backward through the
network, layer by layer:

$$ \frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z}\cdot \frac{\partial z}{\partial w} $$

One forward pass to predict, one backward pass to assign "blame" to each weight, then
one nudge. Repeat for every batch, every **epoch** (one full pass over the data). We ran
8 epochs; that's why loss fell $0.75 \to 0.48$.

**AdamW** (our optimizer) is gradient descent with two upgrades:
- **Momentum** — average recent gradients so we don't zig-zag: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
- **Adaptive step size** — scale each weight's step by how noisy its gradient is: $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$

$$ \theta \leftarrow \theta - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon} - \eta\lambda\theta $$

(The last term is **weight decay** $\lambda$ — gently shrinks weights to avoid overfitting.)

---

## 9. Evaluation metrics — the math of "how good"

After training, we score predictions against truth using four pixel counts:

| | Predicted water | Predicted not |
|---|---|---|
| **Actually water** | TP (true positive) | FN (false negative) |
| **Actually not** | FP (false positive) | TN (true negative) |

$$ \text{Precision} = \frac{TP}{TP+FP} \quad\text{(of pixels I called water, how many were?)} $$

$$ \text{Recall} = \frac{TP}{TP+FN} \quad\text{(of real water pixels, how many did I catch?)} $$

$$ \text{IoU} = \frac{TP}{TP+FP+FN} \quad\text{(overlap ÷ union — the segmentation gold standard)} $$

$$ \text{Dice/F1} = \frac{2\,TP}{2\,TP+FP+FN} $$

IoU and Dice both ignore the huge TN count, so they **can't be fooled** by a model that
just says "no water everywhere." They're linked by:

$$ \text{Dice} = \frac{2\,\text{IoU}}{1+\text{IoU}} $$

**Why accuracy lies here:** $\text{Accuracy} = \frac{TP+TN}{\text{all}}$. With 98.2%
non-water pixels, predicting "never water" scores 98.2% accuracy but **IoU = 0**. That
gap is the whole reason we report IoU/Dice.

---

## 10. Why our U-Net collapsed to "no water" (the math of the trap)

Recall the gradient per pixel: $\dfrac{\partial \mathcal{L}}{\partial z} = p - y$.

Average it over an image that is **98.2% zeros**. The average target $\bar{y}\approx0.018$.
Plain BCE weights every pixel equally, so the fastest way to shrink *total* loss is to push
$p$ down almost everywhere — the rare $y=1$ pixels are drowned out. With a small model and
few epochs, it settles at $p<0.5$ for **every** pixel → predicts no water → IoU 0.

It's not a bug; it's the loss doing exactly what we told it to. We told it the wrong thing.

**The fix (next step):** make water pixels "count more."

**Option A — weighted BCE** with a positive weight $w^+ > 1$:

$$ \mathcal{L} = -\big[\, w^{+}\, y\log p + (1-y)\log(1-p) \,\big] $$

Set $w^+ \approx \frac{\#\text{negative}}{\#\text{positive}} \approx 55$ so the classes balance.

**Option B — soft Dice loss**, which optimizes overlap directly (and ignores TN):

$$ \mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon} $$

A common, robust choice is **BCE + Dice combined**. That's what we'll try next.

---

## 11. What we used, in one table

| Piece | Type | Math core |
|---|---|---|
| NDWI heuristic | Fixed formula (no ML) | $(G-\text{NIR})/(G+\text{NIR})$ |
| Convolution | Feature extractor | sliding dot product |
| ReLU / Sigmoid | Activations | $\max(0,z)$ / $1/(1+e^{-z})$ |
| Tiny U-Net | CNN, encoder–decoder + skips | stacked convolutions |
| BCE | Loss | $-[y\log p+(1-y)\log(1-p)]$ |
| AdamW | Optimizer | momentum + adaptive step |
| IoU / Dice | Metrics | overlap ÷ union |

---

*Next experiment: swap plain BCE for weighted-BCE + Dice, retrain, and watch IoU climb
off zero. See `data/results/main_table.md` for live numbers.*
