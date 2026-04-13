LoRA trains 0.1% of parameters with ~90% of full fine-tuning's benefit. This democratizes fine-tuning: a 65B model on a single 48GB GPU. But this efficiency comes with hidden costs—wrong rank choices, poor layer targeting, and quantization artifacts can silently destroy quality.

## Learning Progression (Easy -> Hard)[#](#learning-progression-easy-hard)

Use this sequence as you read:

1. Start with `Prerequisites: What You Should Know` to build core intuition and shared vocabulary.
2. Move to `Why Low-Rank Works` to understand the mechanism behind the intuition.
3. Apply the idea in `Mathematical Derivation of LoRA` with concrete examples or implementation details.
4. Challenge your understanding in the failure-mode section and check what breaks first.
5. Then zoom out to scale-level tradeoffs so the same concept holds at larger model and system sizes.
6. Map the concept to production constraints to understand how teams make practical tradeoffs.

## Prerequisites: What You Should Know[#](#prerequisites-what-you-should-know)

*Flow bridge: Start here; this section establishes the base mental model for the rest of the lesson.*

▶Refresher: Matrix Factorization and SVD

If `W` is a `d\_out × d\_in` matrix, we can decompose it using Singular Value Decomposition (SVD):

```
W = U Σ V^T
```

where:

* `U` is `d\_out × d\_out` (left singular vectors)
* `Σ` is diagonal with singular values `σ\_1 ≥ σ\_2 ≥ ... ≥ σ\_r`
* `V^T` is `d\_out × d\_in` (right singular vectors transposed)

**Key insight:** If we truncate to rank `r`, we keep only the top-r singular values:

```
W_r = U_r Σ_r V_r^T
```

This is the best rank-r approximation of `W` in Frobenius norm. The singular values tell you how much "importance" each dimension contributes.

▶Refresher: Gradient Flow During Training

During fine-tuning, we compute gradients `dW = ∂L/∂W` with respect to the loss. These gradients also have a structure—they're not random noise.

**Key observation:** When you train a model on a new task, the gradient updates concentrate in a low-rank subspace. This is because:

1. The task signal is sparse (only relevant to some features)
2. Most base knowledge is already captured in the pretrained weights
3. Fine-tuning is learning a "task-specific adapter," not rewriting the model

So `dW` often has effective rank much smaller than min(d\_out, d\_in).

▶Refresher: Hardware Memory Hierarchy

Fine-tuning memory is dominated by:

1. **Model weights:** `d\_out × d\_in × 2 bytes` (fp16) or `× 0.5 bytes` (int4)
2. **Gradients:** Same size as weights (only for trainable params)
3. **Optimizer state:** `× 4 bytes × 2` for Adam (m and v vectors)

For a 7B model in fp16:

* Weights: 7B × 2 = 14 GB
* Gradients: 14 GB
* Optimizer: 28 GB
* **Total: ~56 GB** (needs A100 80GB)

With LoRA (0.1% params):

* Weights: 14 GB (frozen, no gradient)
* LoRA weights: 0.014 GB
* LoRA gradients: 0.014 GB
* LoRA optimizer: 0.056 GB
* **Total: ~14 GB** (fits on single RTX 3090)

## Instructor Lens[#](#instructor-lens)

## Why Low-Rank Works[#](#why-low-rank-works)

*Flow bridge: Building on Prerequisites: What You Should Know, this section adds the next layer of conceptual depth.*

low\_rank\_intuition.pycpu-only

```
import numpy as np

def analyze_weight_updates(original_dim=4096, hidden_dim=4096, num_samples=1000):
  """
  Demonstrate that fine-tuning updates are approximately low-rank.
  """
  np.random.seed(42)

  # Simulate weight updates from fine-tuning
  # (In practice, this would be W_finetuned - W_pretrained)
  weight_updates = np.random.randn(original_dim, hidden_dim) * 0.1

  # SVD to find the effective rank
  U, S, Vt = np.linalg.svd(weight_updates, full_matrices=False)

  # How much variance is captured by top-k singular values?
  total_variance = np.sum(S**2)
  cumulative_variance = np.cumsum(S**2) / total_variance

  print("Effective Rank of Weight Updates")
  print("=" * 60)
  print("Weight matrix shape: %d x %d" % (original_dim, hidden_dim))
  print("Total parameters: %s" % format(original_dim * hidden_dim, ','))
  print()
  print("Variance captured by top-k singular values:")
  print()

  for k in [1, 4, 8, 16, 32, 64, 128]:
      if k <= len(cumulative_variance):
          var_captured = cumulative_variance[k-1]
```

LoRA: W' = W + BA·alpha/r

Full Fine-Tuning

W (d\_out × d\_in)  
All params updated  
~16B gradients+opt

W (frozen)  
No gradient

B (d\_out × r)  
rank=8

A (r × d\_in)  
rank=8

B×A (d\_out × d\_in)  
Low-rank update

alpha/r  
scaling

Output  
W + (B×A)·α/r

## Mathematical Derivation of LoRA[#](#mathematical-derivation-of-lora)

*Flow bridge: Building on Why Low-Rank Works, this section adds the next layer of conceptual depth.*

**The Problem:** Standard fine-tuning updates every weight matrix `W ∈ ℝ^(d\_out × d\_in)`. For a 7B model, that's 7B parameters × 4 bytes = 28 GB just for FP32 weights, plus 28 GB gradients, plus 56 GB optimizer state (Adam momentum + variance) = 112 GB.

**The Observation (Hu et al., 2021):** When fine-tuning on task-specific data, the weight update `ΔW = W\_finetuned - W\_pretrained` has low intrinsic rank. That is, `ΔW` can be well-approximated by a rank-r factorization.

**The Solution:** Instead of computing `W' = W + ΔW`, parameterize `ΔW = BA` where:

* `B ∈ ℝ^(d\_out × r)`
* `A ∈ ℝ^(r × d\_in)`
* `r` is much smaller than `d_in` and `d_out`

Then the forward pass becomes:

```
y = (W + BA · α/r) x^T
```

where `α/r` is a scaling factor (typically `α = 16` or `32`, and we divide by rank to keep updates at similar magnitude regardless of rank).

**Why the scaling `α/r`?** If we don't scale, doubling the rank would double the update magnitude, changing the optimization trajectory. By dividing by rank, we stabilize learning rates across different rank choices.

**Parameter Count:**

* Standard: `d\_out × d\_in` parameters
* LoRA: `r × d\_in + d\_out × r = r(d\_in + d\_out)` parameters
* Compression: `(d\_out × d\_in) / (r(d\_in + d\_out))`

For `d\_in = d\_out = 4096, r = 8`:

* Standard: 16.8M parameters
* LoRA: 65K parameters
* **Compression: 258x**

**Initialization matters:**

* `A` initialized with Kaiming normal: `N(0, √(2/d\_in))`
* `B` initialized to zero: `B = 0`

Why? At initialization, `BA = 0`, so `W' = W` (identity mapping). Training then gradually learns the task-specific update.

lora\_math\_derivation.pycpu-only

```
import numpy as np

print("=" * 70)
print("LoRA Mathematical Derivation")
print("=" * 70)
print()

# Setup
d_out, d_in = 4096, 4096
ranks = [4, 8, 16, 32, 64]
alpha = 16  # standard choice

print("Layer shape: %d x %d" % (d_out, d_in))
print("Full parameters: %s" % format(d_out * d_in, ','))
print()

print("LoRA with alpha=%d:" % alpha)
print("%-6s %-15s %-12s %-12s" % ('Rank', 'LoRA Params', '% of Full', 'Scaling a/r'))
print("-" * 50)

for r in ranks:
  lora_params = r * (d_out + d_in)
  pct = 100 * lora_params / (d_out * d_in)
  scaling = alpha / r
  print("%-6d %-15s %.3f%%       %.3f" % (r, format(lora_params, ','), pct, scaling))

print()
print("Key insight: Scaling factor α/r keeps update magnitude stable.")
print()
```

lora\_implementation.pycpu-only

```
import numpy as np

class LoRALinear:
  """
  LoRA-wrapped linear layer.

  Instead of W' = W (full fine-tune), we have:
  W' = W + (B @ A) * alpha/rank

  where:
  - W: frozen original weights (d_out, d_in)
  - A: trainable (rank, d_in), initialized with Kaiming
  - B: trainable (d_out, rank), initialized with zeros
  - alpha/rank: scaling factor (typically alpha=16)
  """

  def __init__(self, d_in, d_out, rank=8, alpha=16):
      self.d_in = d_in
      self.d_out = d_out
      self.rank = rank
      self.alpha = alpha
      self.scaling = alpha / rank

      # Original weights (pretrained, frozen)
      self.W = np.random.randn(d_out, d_in) * 0.02

      # LoRA matrices (trainable)
      # A: Kaiming initialization (variance based on input dim)
      self.A = np.random.randn(rank, d_in) * np.sqrt(2.0 / d_in)
      # B: Zero initialization (so initially W' = W exactly)
```

## Practical LoRA Configuration[#](#practical-lora-configuration)

*Flow bridge: Building on Mathematical Derivation of LoRA, this section adds the next layer of conceptual depth.*

lora\_training\_setup.pycpu-only

```
import numpy as np

class TransformerLoRA:
  """
  LoRA applied to a Transformer attention head.
  """

  def __init__(self, hidden_dim=512, rank=8, target_layers=["q_proj", "v_proj"]):
      self.hidden_dim = hidden_dim
      self.rank = rank
      self.target_layers = target_layers
      self.alpha = 16
      self.scaling = self.alpha / rank

      # Frozen base weights (pretrained)
      self.W_q = np.random.randn(hidden_dim, hidden_dim) * 0.02
      self.W_k = np.random.randn(hidden_dim, hidden_dim) * 0.02
      self.W_v = np.random.randn(hidden_dim, hidden_dim) * 0.02
      self.W_o = np.random.randn(hidden_dim, hidden_dim) * 0.02

      # LoRA adapters (trainable)
      self.lora_params = {}
      if "q_proj" in target_layers:
          self.lora_params["q"] = {
              "A": np.random.randn(rank, hidden_dim) * np.sqrt(2.0 / hidden_dim),
              "B": np.zeros((hidden_dim, rank))
          }
      if "v_proj" in target_layers:
          self.lora_params["v"] = {
              "A": np.random.randn(rank, hidden_dim) * np.sqrt(2.0 / hidden_dim),
```

## QLoRA: LoRA + 4-Bit Quantization[#](#qlora-lora-4-bit-quantization)

*Flow bridge: Building on Practical LoRA Configuration, this section adds the next layer of conceptual depth.*

### How 4-Bit Quantization Works[#](#how-4-bit-quantization-works)

Standard fp16 stores each weight as 16 bits (sign + exponent + mantissa). We can achieve 4 bits per weight by:

1. **Grouping weights:** Divide weight matrix into groups (e.g., 64 weights per group)
2. **Computing scale:** For each group, find `scale = (max - min) / 15` (15 is `2^4 - 1`)
3. **Quantizing:** Round each weight to nearest integer in `[0, 15]`
4. **Storing:** Original value = `(quantized\_int - 8) \* scale`

This reduces a 512×512 weight matrix from 512 bytes (fp16) to 64 bytes (int4) + overhead for scales.

### Double Quantization[#](#double-quantization)

The scale factors themselves take memory (typically fp32, 4 bytes per scale). QLoRA quantizes these too:

* Scale factor = `(value - zero\_point) \* scale\_of\_scale`
* Now: 4 bits for weights + 2 bits for scale factors ≈ **0.625 bytes/parameter**

quantization\_mechanics.pycpu-only

```
import numpy as np

def quantize_4bit(weights, group_size=64):
  """
  Simulate 4-bit quantization with grouping.
  """
  shape = weights.shape
  num_groups = (shape[-1] + group_size - 1) // group_size

  quantized = np.zeros_like(weights, dtype=np.int8)
  scales = []

  for i in range(num_groups):
      start = i * group_size
      end = min((i + 1) * group_size, shape[-1])
      group = weights[:, start:end]

      # Compute scale: (max - min) / 15
      min_val = np.min(group)
      max_val = np.max(group)
      scale = (max_val - min_val) / 15.0

      if scale == 0:
          scale = 1e-6

      # Quantize to [0, 15]
      centered = (group - min_val) / scale
      quantized[:, start:end] = np.clip(np.round(centered), 0, 15).astype(np.int8)

      scales.append(scale)
```

## Memory Comparison: Full vs LoRA vs QLoRA[#](#memory-comparison-full-vs-lora-vs-qlora)

*Flow bridge: Building on QLoRA: LoRA + 4-Bit Quantization, this section adds the next layer of conceptual depth.*

memory\_estimation.pycpu-only

```
import numpy as np

def estimate_memory(model_size_B, method="full_fp16", batch_size=1, seq_len=512):
  """
  Estimate GPU memory for different fine-tuning methods.

  Components:
  1. Model weights
  2. Gradients (only for trainable params)
  3. Optimizer state (Adam: m and v vectors for each trainable param)
  4. Activations (batch × seq_len × hidden × num_layers)
  """
  params = model_size_B * 1e9
  hidden_dim = 4096  # typical for LLMs

  # Activation memory (rough estimate)
  num_layers = 32  # typical
  activation_mem = batch_size * seq_len * hidden_dim * num_layers * 2 / 1e9  # fp16

  if method == "full_fp16":
      # All params: weights + gradients + optimizer (Adam)
      # Adam: 1 (param) + 1 (grad) + 2 (m, v optimizer states) = 4x param size
      model_mem = params * 2 / 1e9  # fp16
      grad_mem = params * 2 / 1e9
      opt_mem = params * 4 * 2 / 1e9  # m and v in fp32
      return {
          "model": model_mem,
          "gradients": grad_mem,
          "optimizer": opt_mem,
          "activations": activation_mem,
```

## Rank and Target Selection Guide[#](#rank-and-target-selection-guide)

*Flow bridge: Building on Memory Comparison: Full vs LoRA vs QLoRA, this section adds the next layer of conceptual depth.*

rank\_selection.pycpu-only

```
import numpy as np

def estimate_quality(rank, num_targets=2, task_complexity=50):
  """
  Rough estimate of fine-tuning quality vs rank.

  Quality increases with rank but with diminishing returns.
  Complexity determines how much capacity is needed.
  """
  capacity = rank * num_targets  # effective capacity
  quality = 1 - np.exp(-capacity / (task_complexity * 1.5))
  return min(quality, 0.98)  # cap at 98% (can't exceed full fine-tune)

def estimate_cost(rank, num_targets=2):
  """Relative cost (memory, compute)."""
  return rank * num_targets  # linear scaling

print("Rank Selection: Performance vs Cost Tradeoff")
print("=" * 70)
print()

ranks = [4, 8, 16, 32, 64]

for num_targets in [1, 2, 4]:
  target_name = 'q_only' if num_targets == 1 else ('q_v' if num_targets == 2 else 'all_linear')
  print("\nTargets: %d (e.g., %s)" % (num_targets, target_name))
  print("%-6s %-12s %-10s %-12s" % ('Rank', 'Quality', 'Cost', 'Efficiency'))
  print("-" * 40)

  for rank in ranks:
```

## Comparison: Full vs LoRA vs QLoRA[#](#comparison-full-vs-lora-vs-qlora)

*Flow bridge: Building on Rank and Target Selection Guide, this section adds the next layer of conceptual depth.*

method\_comparison.pycpu-only

```
import numpy as np

def compare_methods():
  """
  Compare fine-tuning methods across multiple dimensions.
  """
  methods = {
      "Full FP16": {
          "memory": 1.0,
          "speed": 0.5,
          "quality": 1.0,
          "hardware": "Multi-GPU / A100 80GB",
          "typical_use": "Final production training",
          "convergence": "Baseline",
          "adapter_merge": "N/A",
      },
      "Full BF16": {
          "memory": 1.0,
          "speed": 0.65,
          "quality": 0.98,
          "hardware": "Multi-GPU / A100 80GB",
          "typical_use": "Production with numerical stability",
          "convergence": "Better than FP16",
          "adapter_merge": "N/A",
      },
      "LoRA FP16": {
          "memory": 0.4,
          "speed": 0.8,
          "quality": 0.94,
          "hardware": "Single A100 40GB",
```

## Break It: Common Failure Modes[#](#break-it-common-failure-modes)

*Flow bridge: Now that the core mechanism is clear, stress-test it under realistic failure conditions.*

break\_it\_failure\_modes.pycpu-only

```
import numpy as np

print("=" * 80)
print("COMMON LoRA FAILURE MODES")
print("=" * 80)
print()

# Failure Mode 1: Rank Too Low
print("1. UNDERFITTING: Rank Too Low")
print("-" * 80)

task_complexities = {
  "simple_formatting": 20,
  "instruction_following": 50,
  "reasoning_and_code": 120,
}

for task_name, complexity in task_complexities.items():
  print("\nTask: %s (complexity = %d)" % (task_name, complexity))
  print("  Recommended minimum rank: %d" % max(4, complexity // 8))

  ranks = [4, 8, 16, 32]
  print("  %-8s %-12s %-12s %-15s" % ('Rank', 'Capacity', 'Performance', 'Verdict'))
  print("  %s" % ('-'*52))

  for rank in ranks:
      capacity = rank * 2
      performance = min(1.0, 1 - np.exp(-capacity / complexity))
      verdict = "OK" if performance > 0.85 else "UNDERFITTING"
      print("  %-8d %-12d %-12s %-15s" % (rank, capacity, "%.1f%%" % (performance*100), verdict))
```

## Scale Thought Experiment: Hardware vs Model Size[#](#scale-thought-experiment-hardware-vs-model-size)

*Flow bridge: With the local mechanism in place, extend it to larger model, context, and system scales.*

Viable Approach

Hardware Tier

RTX 3090  
24GB

A100 40GB

A100 80GB

8x A100

7B QLoRA

30B QLoRA  
13B LoRA

65B QLoRA  
70B LoRA

Full FT  
any size

**Recommended approach by model size:**

| Model Size | Consumer GPU | Single A100 | Multi-GPU |
| --- | --- | --- | --- |
| **1B** | Full FT | Full FT | Full FT |
| **7B** | QLoRA (rank=8-16) | LoRA FP16 (rank=16-32) | Full FT (best quality) |
| **13B** | QLoRA (rank=16) | LoRA FP16 (rank=16-32) | Full FT or LoRA |
| **30B** | QLoRA (rank=16+) | QLoRA (rank=16-32) | LoRA or Full FT |
| **65B** | QLoRA (rank=32+) | QLoRA (rank=32-64) | LoRA FP16 or Full FT |
| **175B+** | Not feasible | QLoRA + tricks | Full FT + parallelism |

**Key tradeoffs:**

* **Small rank (4-8):** Fast, low memory, but underfits on complex tasks
* **Medium rank (16-32):** Sweet spot for most SFT tasks
* **Large rank (64+):** Better quality, more memory, usually unnecessary

## Production Reality: What Big Labs Actually Do[#](#production-reality-what-big-labs-actually-do)

*Flow bridge: Carry these tradeoffs into production constraints and team-level operating decisions.*

### Microsoft Research (LoRA paper, 2021)[#](#microsoft-research-lora-paper-2021)

The original LoRA paper showed:

* **10,000x parameter reduction** on GPT-3 (175B)
* Q and V projection adaptation sufficient for 95%+ quality
* Training time 25% faster than full fine-tuning (less gradient computation)
* Adapters are **mergeable** (you can create multiple task-specific adapters)

Real constraint they didn't emphasize: only works well for task-specific fine-tuning, not base pretraining.

### Hugging Face / Meta (QLoRA, 2023)[#](#hugging-face-meta-qlora-2023)

QLoRA breakthrough allowed:\*\*

* 65B Llama on **single 48GB GPU** (previously needed 8x A100)
* 4-bit NormalFloat quantization (better than standard int4)
* **Adapter stacking**: multiple QLoRA adapters in sequence
* Community inference: Ollama, llama.cpp brought quantized models to laptops

Lessons from practice:

* Rank=16 is usually sufficient even for 65B models
* Double quantization adds minimal quality loss
* Gradient checkpointing + QLoRA = feasible on consumer GPUs

### Industry standard workflow[#](#industry-standard-workflow)

```
Experimentation phase (QLoRA):
  • Rapid iteration on task/data/hyperparams
  • Rank=8-16, Q+V projections
  • 30-60 min training on single A100
  → Validate on dev set

Quality tuning phase (LoRA FP16):
  • Slightly higher rank (16-32)
  • Same layers
  • Monitor for overfitting
  → Small validation improvement (1-2%)

Production phase (Full FT if budget allows):
  • Multi-GPU setup
  • Full fine-tuning with all layers
  • Final quality boost (3-5% over LoRA)
  • But: only if revenue justifies GPU cost

Deployment:
  • QLoRA-trained adapters can be quantized to 4-bit for inference
  • Adapter weights are ~0.5% of base model size
  • Can serve 100+ task-specific adapters from single model instance
```

### Common gotchas we've seen[#](#common-gotchas-weve-seen)

1. **Rank selection:** Start rank=8. If loss plateaus, go to 16. Rarely need rank=64.
2. **Layer targeting:** Q+V is default. Adding K or O usually helps <1% but risks instability.
3. **Learning rate:** LoRA trains faster. Use 2-3x higher LR than you'd use for full FT.
4. **Quantization quality loss:** ~2-5% with 4-bit. Mitigate with higher rank or 8-bit.
5. **Adapter merging:** Done by `merged = base + (lora\_b @ lora\_a) \* scaling`. Simple linear combination.

## Research Hooks & Open Questions[#](#research-hooks-open-questions)

*Flow bridge: Use this practical baseline to frame the open research questions that remain unresolved.*

**Adapter merging and composition:**
We know `W' = W + (B @ A)`. What if you train two adapters `(B1, A1)` and `(B2, A2)` on different tasks? Can you merge them? Combine them? Train a "router" to blend them?

Practical implication: Single base model could serve 100+ tasks with small adapter weights.

**Optimal rank per layer:**
LoRA assumes uniform rank across all adapted layers. But transformer attention layers near input might need different rank than deeper layers. Can we learn rank per layer automatically during training?

Paper idea: Use magnitude pruning on singular values to determine layer-wise rank.

**Why low-rank works (theory):**
Empirically, LoRA works. But why? Is it because:

1. Task signal is inherently low-rank?
2. Pretrained models have learned a low-rank task manifold?
3. Fine-tuning trajectory is constrained to low dimensions?

A theoretical understanding could guide rank selection and improve adaptation efficiency.

**LoRA + distillation:**
Can you distill a full fine-tuned model into a QLoRA adapter? This would give you the quality of full FT with QLoRA's efficiency.

**Multi-adapter inference:**
Some tasks need multiple specialized skills (coding + math). Can you compose multiple trained adapters efficiently at inference time?

**Structured pruning + LoRA:**
LoRA solves parameter efficiency. What about FLOPs? Can you combine LoRA with structured pruning for compute-efficient inference?

## Summary: When to Use What[#](#summary-when-to-use-what)

*Flow bridge: Building on Research Hooks & Open Questions, this section adds the next layer of conceptual depth.*

summary\_decision\_tree.pycpu-only

```
print("=" * 80)
print("EFFICIENT FINE-TUNING DECISION TREE")
print("=" * 80)
print()

decision_tree = """
CHOOSE YOUR PATH:

1. Do you have multi-GPU infrastructure (8+ A100s)?
 YES → Full fine-tuning (maximum quality, justify cost)
 NO  → Go to 2

2. Do you have single A100 or similar (40-80GB)?
 YES → LoRA FP16 with rank=16-32 (good quality, single GPU)
 NO  → Go to 3

3. Do you have consumer GPU (RTX 3090/4090 24GB)?
 YES → QLoRA 4-bit with rank=8-16 (democratizes fine-tuning)
 NO  → Go to 4

4. Do you have CPU or very limited GPU?
 YES → QLoRA on 8-bit quantized model, or smaller base model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER TARGETING (What to adapt):
 • Default: Q + V projections (query + value)
 • If underfitting: Add O projection (output)
 • Rarely needed: All linear layers
 • NEVER adapt: K projection alone (breaks attention)
```

## Takeaways[#](#takeaways)

*Flow bridge: Building on Summary: When to Use What, this section adds the next layer of conceptual depth.*

**The Core Win:** LoRA reduces fine-tuning from prohibitively expensive (multi-GPU, weeks) to accessible (single consumer GPU, hours).

**Why it works:** The task-specific update lives in a low-rank subspace. You don't need to update all 70B parameters—just 0.1% of them, strategically placed.

**The tradeoff:** ~5-10% quality loss vs full fine-tuning. For most tasks, this is worth it.

**The failure modes to watch:**

1. Rank too low → underfitting, plateau on training loss
2. Wrong layers → breaks attention mechanics
3. Quantization artifacts → only visible on edge cases
4. Bad hyperparameters → convergence issues

**Best practice workflow:**

* Experiment with QLoRA (fast feedback loop)
* Move to LoRA FP16 if quality critical
* Full fine-tune only if revenue justifies GPU cost

---

**Next lesson:** We've covered SFT (Supervised Fine-Tuning) — teaching models *what* good responses look like. But SFT doesn't teach *preference*. For that, we need RLHF (Reinforcement Learning from Human Feedback) and reward models that learn to score responses by quality.

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. Can you do this without notes: Understand why fine-tuning updates have low intrinsic rank?
2. Can you do this without notes: Derive LoRA mathematically and implement from scratch?
3. Can you do this without notes: Analyze QLoRA's quantization strategy (4-bit, double quantization)?