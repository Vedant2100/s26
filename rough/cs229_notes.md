# CS 229 — Complete Notes
> Spring 2026 · Ver Steeg · All lectures, quizzes, assignments

---

## 1. ML Basics
### Probability
- Normalization: discrete PMF sums to 1; PDF integrates to 1. p(z) ≥ 0 always (PDF CAN exceed 1)
- Marginalization: P(X=0) = Σ_y P(X=0,Y=y)
- Conditioning: P(Y|X) = P(X,Y)/P(X)
- Bayes: p(θ|D) ∝ p(D|θ)·p(θ)
- MAP: argmax posterior — uses prior as regularizer but ignores weight uncertainty

### Softmax
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
- logits (-log2, +log2) → (1/5, 4/5). logits (0, log3) → (1/4, 3/4)
- Temperature T: divide logits by T. T→0 = argmax. T→∞ = uniform.

### PyTorch ⚠️ missed
- `A * B` = element-wise. `A @ B` = matrix multiply.
- `loss.backward()` → stores gradients in `.grad`. Does NOT update params, zero grads, or free memory.
- Training: forward → loss → `zero_grad()` → `backward()` → `step()`

### GD vs SGD
- GD: full dataset, deterministic, more likely stuck in local minima
- SGD: mini-batches, noise = exploration, less likely to get stuck

---

## 2. Tokenization & Embeddings
- BPE: merge most frequent adjacent char pairs iteratively
- New words → subwords: "ghostable" → "ghost-able"
- `nn.Embedding(V, d)`: integer ID → dense vector. Output: (seq_len, d)
- HW0: ord(c) − 96 mapping (a=1, z=26, pad=0). tokenize("cat") = [3,1,20,0,…]
- HW0 NameMLP: Embedding → **Flatten** → Linear(240,128) → ReLU → Linear(128,64) → ReLU → Linear(64,2)

---

## 3. Transformers
### Self-Attention
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
- Complexity: O(N²·d) — all N² pairwise dot products
- √d_k scaling: TEMPERATURE (prevents saturation at init), NOT normalization ⚠️

### Positional Embeddings
- Removing them: model can't distinguish word order (permutation-invariant)
- **RoPE**: rotate Q,K so dot product depends on relative position. Used in LLaMA, Qwen.
- **ALiBi**: subtract |i-j| from attention scores
- **NoPE**: no positional embedding (used in HW3 ViT — ||z_t|| already encodes t)

### KV Cache
- Cache past K,V at inference — no recompute. Memory: O(N·d·L)
- TurboQuant: 4-bit KV cache. Apply random rotation before quantizing → better precision.

### BERT vs GPT ⚠️ missed
| | GPT | BERT |
|---|---|---|
| Style | Decoder-only, causal | Encoder-only, bidirectional |
| Pretraining | Next-token prediction | Masked LM (MLM) |
| Probability model | **Yes** | **No** |
| MLM | — | Mask 15%: 80%→[MASK], 10%→random, 10%→unchanged |

---

## 4. LLM Training Pipeline
1. **Pretraining**: next-token prediction. Cross-entropy. Causal mask. BOS token required.
2. **SFT**: fine-tune on (instruction, response) pairs. Same loss.
3. **RLHF**: align to human preferences.
4. **Distillation + Quantization**: compress.
- Catastrophic forgetting fix: pretrain on A∪B∪C jointly, then fine-tune.

### PPO / DPO / GRPO
- **PPO**: reward model from ranked preferences, maximize reward with RL, KL penalty, clipped objective
- **DPO**: no reward model, directly optimize on ranked pairs (simpler)
- **GRPO** (DeepSeek): generate G responses, rule-based rewards r₁…rG, advantage = (rᵢ − r̄)/σᵣ, clip updates. No value model. Verifiable rewards: math correctness, code runs, `<think>` tags.

---

## 5. Prompting & LoRA
### HW1 Results
| Method | Overall | Level 1 |
|---|---|---|
| Base | 4.7% | 10% |
| System | 11.3% | 30% |
| CoT | 16.0% | 38% |
| ICL | 10.7% | 28% |
| **LoRA** | **36.7%** | **96%** |

### LoRA
$$\Delta W = BA,\quad A\in\mathbb{R}^{r\times d_\text{in}},\ B\in\mathbb{R}^{d_\text{out}\times r}$$
$$\text{trainable params} = r(d_\text{in}+d_\text{out}) \ll d_\text{in}d_\text{out}$$
- Merge: W' = W + BA. Zero inference overhead.
- QLoRA: quantize frozen base to 4-bit, LoRA adapters in float16.
- **Rank-1 matrix**: outer product of two vectors ⚠️ (Quiz 2)

---

## 6. Scaling Laws & Goodhart
- Test loss: power law in model params N and data D
- **Chinchilla**: compute-optimal = scale N and D equally. Most models undertrained. "Optimal requires underfitting."
- **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure"
- Paperclip optimizer: AI maximizes paperclip count → converts solar system
- LLM Goodharting: benchmark leakage, test set optimization, Claude aware of being tested

---

## 7. Computer Vision & CNNs
### Convolution
$$|y_\text{valid}| = |x| - |h| + 1 \qquad \text{2D same: output size = input size}$$
- Stride s: output = ⌊(|x|−|h|)/s⌋ + 1
- Example: input (32,32,3), 3×3 kernel, same padding, 4 channels → output (32,32,4)

### Key Concepts
- 1×1 conv: linear transform across channels per pixel (channel mixing, no spatial)
- **Fully convolutional**: NO FC layers → works on any input size
- **Rotation invariance**: output unchanged (classification)
- **Rotation equivariance**: output rotates with input (segmentation) ⚠️

### Architectures
- **ResNet**: F(x)+x skip connections. Trains very deep networks.
- **U-Net**: encoder-decoder + skip connections. Segmentation + denoiser.
- **ViT (HW3)**: 8×8 patches, 16 patches per 32×32 image. No spatial inductive bias.
- **Deep Image Prior**: untrained U-Net is sufficient prior for reconstruction.

---

## 8. Contrastive Learning
### InfoNCE Loss ⚠️ missed
$$\mathcal{L} = -\log\frac{\exp(z_a\cdot z_p)}{\exp(z_a\cdot z_p)+\sum_k\exp(z_a\cdot z_k^-)}$$
- Minimize = push anchor toward positive, away from negatives

### SimCLR
- Augmentations: random crop, color jitter, grayscale, h-flip, Gaussian blur
- Large batch critical — more negatives = better representations

### CLIP
- Positive: (image, its caption). Negatives: all other combos in batch.
- Zero-shot: embed image + class names, pick highest cosine similarity.

### DINO
- Teacher = EMA of student weights. No teacher gradients.
- Student predicts teacher's pseudo-label distribution from different augmented views.
- ViT learns to segment objects without labels.
- DINOv2: + iBOT loss (masked patch prediction) + Gram Anchoring.

---

## 9. Uncertainty & Calibration
### Two Types
- **Aleatoric** (data): irreducible — noise, missing features. Persists with infinite data.
- **Epistemic** (model): reducible — many models fit data. Vanishes with infinite data.

### ECE ⚠️ missed (Q4 and Q5)
$$\text{ECE} = \sum_b \frac{n_b}{N}\,|\text{acc}(b) - \text{conf}(b)|$$
- Binary: use P(positive class) as confidence (allows full [0,1], not just [0.5,1])
- Overconfident: confidence > accuracy. Underconfident: confidence < accuracy.
- ECE=0 ≠ accurate. Always predict base rate → perfect calibration but useless.

### HW2 Results
| Method | ECE | Notes |
|---|---|---|
| Base | 0.128 | Overconfident |
| MC Dropout | 0.102 | 30 passes, avg probs |
| Label Smoothing | 0.040 | `label_smoothing=0.1` |
| **Temp Scaling T=3** | **0.037** | Post-hoc, no retraining |

### OOD Problems
- Neural nets: high confidence on OOD inputs (adversarial, noise, random images)
- ImageNet-C: accuracy drops with corruption, model stays overconfident
- Ideal: high uncertainty far from training data. Neural nets fail at this.

---

## 10. Bayesian Deep Learning
$$p(\theta|D) \propto p(D|\theta)\cdot p(\theta) \qquad \text{BMA: } p(y|x,D) = \int p(y|x,\theta)\,p(\theta|D)\,d\theta$$

### Approximations
- **Laplace**: Gaussian N(θ_MAP, H⁻¹) using Hessian H around MAP
- **Variational Bayes**: learn q(θ) ≈ p(θ|D) by minimizing KL. Didn't work well in practice.
- **SWA**: average weights along SGD trajectory. Better generalization than single endpoint.
- **SWAG**: fit Gaussian over SWA iterates → sample for BMA
- **Deep Ensembles**: retrain with different seeds, average predictions. Best in practice. Eliminates double descent for SGD.

---

## 11. Optimization Dynamics
### LR Scaling ⚠️ missed
$$\eta_\text{new} = \eta_\text{old}\times\frac{B_\text{new}}{B_\text{old}} \qquad \text{(4× batch → 4× LR)}$$

### SGD as Langevin Dynamics
$$\theta_{t+1} = \theta_t - \eta\nabla\mathcal{L}(\theta_t) + \sqrt{2\eta}\,\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,I)$$
- Stationary distribution ∝ exp(−L/T), T ∝ η/batch_size
- **Fokker-Planck**: governs density evolution under stochastic dynamics
- Why SGD generalizes: explores landscape, prefers wide flat minima

### Momentum
$$v_t = \mu v_{t-1} + \eta g_t,\quad \theta_t = \theta_{t-1} - v_t$$
- Extra memory for v_t. Smooths noise. Accelerates consistent directions.
- Nesterov: evaluate gradient at predicted next position. Optimal convergence for convex.

### Adam
$$m_t = \beta_1 m_{t-1}+(1-\beta_1)g_t \qquad v_t = \beta_2 v_{t-1}+(1-\beta_2)g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\varepsilon}\hat{m}_t$$
- Adaptive in (A) time and (B) per-parameter. Noisy params → smaller effective LR.

---

## 12. Generative Models
### Flow Matching (HW3)
$$z_t = tx + (1-t)\varepsilon \qquad v = \frac{x-z_t}{1-t} \qquad \mathcal{L} = \mathbb{E}\!\left[\left\|\frac{\hat{x}-x}{1-t}\right\|^2\right]$$
$$\text{Euler: } z_{k+1} = z_k + \frac{1}{T}\cdot\frac{\hat{x}-z_k}{1-t_k} \qquad t_k = k/T$$
- Time sampling: logit-normal t=σ(μ+σ₀z), μ=−0.8, σ₀=0.8
- T=10–50 steps sufficient (flow matching straight paths >> diffusion)

### Score Matching / Diffusion
- Score function: s(x) = ∇_x log p(x)
- Denoising score matching: train to predict noise ε from z_t. MSE loss = score matching.
- DDPM forward: q(z_t|z_{t-1}) = N(√(1−β_t)·z_{t-1}, β_tI)
- DDPM = VAE where encoder = fixed noising, decoder = U-Net. LVLB = Σ MSE terms at each noise level.

### VAE
$$\text{ELBO} = \underbrace{\mathbb{E}_q[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_\text{KL}(q_\phi(z|x)\|p(z))}_{\text{rate/regularization}}$$
$$D_\text{KL}(\mathcal{N}(\mu,\sigma^2)\|\mathcal{N}(0,1)) = \tfrac{1}{2}(\mu^2+\sigma^2-\log\sigma^2-1)$$
- Reparameterization: z = μ(x) + σ(x)·ε, ε~N(0,I). Allows backprop through sampling.
- β-VAE: weight KL by β to trade rate vs distortion.

### VQ-VAE
- Discrete codebook of K embeddings. Encoder → nearest codebook entry → index.
- Loss: reconstruction + commitment loss + codebook loss. Straight-through estimator.
- **VQ-GAN**: VQ-VAE + GAN loss → sharper reconstruction. Used in Stable Diffusion.

### GAN
$$\min_G\max_D\,\mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$$
- No encoder, no probability model. Sharp images. Mode collapse. Training instability.

---

## 13. Graph Neural Networks
### Tasks
- Node classification, Link prediction, Graph-level prediction
- Inductive = new graphs at test time. Transductive = same graph.

### GCN (Message Passing)
$$h_i^{(l+1)} = \sigma\!\left(W\cdot\text{mean}_{j\in\mathcal{N}(i)\cup\{i\}}h_j^{(l)}\right)$$
- L layers → L-hop neighborhood aggregation
- Over-smoothing: L→∞, all nodes → same representation

### Pitfalls & Solutions
- **MixHop**: aggregate at multiple hop distances simultaneously
- **GraphSAGE**: sample nodes + neighbors per step. Inductive. Generalizes to new nodes.
- **Decoupled**: precompute aggregations, train MLP separately. Much faster.

### Skip-gram (DeepWalk)
- Random walks → Word2Vec embedding. Transductive only.

---

## 14. Formula Sheet

| Concept | Formula |
|---|---|
| Softmax | $\text{softmax}(z_i) = e^{z_i}/\sum_j e^{z_j}$ |
| Attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| Attention complexity | $O(N^2 d)$ |
| ECE | $\sum_b (n_b/N)\|acc(b)-conf(b)\|$ |
| InfoNCE | $-\log(e^{z_a\cdot z_p}/(e^{z_a\cdot z_p}+\sum e^{z_a\cdot z_k^-}))$ |
| Flow noising | $z_t = tx+(1-t)\varepsilon$ |
| Velocity | $v=(x-z_t)/(1-t)$ |
| Flow loss | $\mathbb{E}[\|(\hat{x}-x)/(1-t)\|^2]$ |
| Euler step | $z_{k+1}=z_k+(1/T)\cdot v_k$ |
| Momentum | $v_t=\mu v_{t-1}+\eta g_t$ |
| Adam m | $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$ |
| Adam v | $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$ |
| VAE ELBO | $\mathbb{E}_q[\log p(x|z)] - D_{KL}(q\|p)$ |
| KL Gaussians | $(1/2)(\mu^2+\sigma^2-\log\sigma^2-1)$ |
| LoRA params | $r(d_\text{in}+d_\text{out})$ |
| LR scaling | $\eta_\text{new}=\eta_\text{old}\times B_\text{new}/B_\text{old}$ |
| Valid conv | $|y|=|x|-|h|+1$ |
| GCN | $h_i^{l+1}=\sigma(W\cdot\text{mean}_{j\in\mathcal{N}(i)}h_j^l)$ |
| Bayes | $p(\theta|D)\propto p(D|\theta)p(\theta)$ |
