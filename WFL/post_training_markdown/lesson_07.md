In this tutorial, you will derive the Bradley-Terry preference model from first principles, implement the reward model loss function and analyze its gradients, build a reward model architecture with an LM backbone and scalar head, and diagnose training failures through loss curves and preference consistency checks.

## Prerequisites Refresher[#](#prerequisites-refresher)

▶Sigmoid function basics

The sigmoid (logistic function) maps any real number to [0, 1]:

σ(x) = 1 / (1 + e^(-x))

Key properties:

* σ(0) = 0.5
* σ(∞) ≈ 1
* σ(-∞) ≈ 0
* d/dx σ(x) = σ(x)(1 - σ(x))

In Bradley-Terry, we use σ to convert a scalar reward difference into a probability. The larger the difference r(A) - r(B), the closer σ(r(A) - r(B)) gets to 1.

▶Cross-entropy loss for binary classification

When training binary classifiers, we use cross-entropy loss:

L = -[y \* log(p) + (1-y) \* log(1-p)]

Where:

* y ∈ {0, 1} is the true label
* p ∈ [0, 1] is the model's predicted probability

For Bradley-Terry: y=1 (chosen is always "correct"), so:
L = -log(σ(r\_chosen - r\_rejected))

This naturally penalizes high-magnitude errors.

▶Transformer hidden states

A transformer encodes a token sequence into hidden states:

* Input: [token\_1, token\_2, ..., token\_N]
* Output: [h\_1, h\_2, ..., h\_N] where h\_i ∈ ℝ^d

Each h\_i is a learned representation of the context up to position i. The **last token's hidden state** h\_N contains information about the entire sequence.

For reward modeling, we typically use h\_N (last position) because it's attended to all previous tokens.

## From Preferences to Rewards[#](#from-preferences-to-rewards)

bradley\_terry.py

```
import numpy as np

def sigmoid(x):
  """Sigmoid function: σ(x) = 1 / (1 + e^(-x))"""
  return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def bradley_terry_probability(reward_a, reward_b):
  """
  Bradley-Terry model: probability that A is preferred to B.

  P(A > B) = σ(r(A) - r(B)) = 1 / (1 + e^(-(r(A) - r(B))))

  Intuition:
  - If r(A) >> r(B): P(A > B) ≈ 1
  - If r(A) << r(B): P(A > B) ≈ 0
  - If r(A) = r(B):  P(A > B) = 0.5
  """
  return sigmoid(reward_a - reward_b)

# Demonstration
print("Bradley-Terry Model")
print("=" * 60)
print()
print("P(A > B) = σ(r(A) - r(B))")
print()

examples = [
  (5.0, 2.0, "A much better than B"),
  (3.0, 3.0, "A and B equally good"),
  (2.0, 4.0, "B better than A"),
```

## Mathematical Derivation[#](#mathematical-derivation)

bt\_derivation.py

```
"""
Bradley-Terry Derivation from First Principles
==============================================

Starting point: Thurstone's Law of Comparative Judgment (1927)
- Each item has a "true quality" μ
- When evaluated, perceived quality is μ + noise
- Noise follows a distribution (originally Gaussian)

Thurstone Case V (equal variance Gaussian):
- P(A > B) = Φ((μ_A - μ_B) / √2σ)  where Φ is normal CDF

Bradley-Terry Modification (1952):
- Use logistic noise instead of Gaussian
- Mathematically cleaner, nearly identical predictions

Logistic assumption:
- Let r_A, r_B be reward values
- Perceived quality: q_A = r_A + ε_A, q_B = r_B + ε_B
- Where ε follows logistic distribution

Key identity:
- If ε_A, ε_B are i.i.d. Gumbel(0, 1), then (ε_A - ε_B) is Logistic
- This gives us: P(A > B) = σ(r_A - r_B)

"""

import numpy as np

def derive_bt_loss():
```

## Scalar Head Design[#](#scalar-head-design)

The reward head is deceptively simple but has important design choices.

Last Hidden State  
h\_N ∈ ℝ^d

Linear Projection  
W ∈ ℝ^(1×d), b ∈ ℝ

Reward Scalar  
r = Wh\_N + b

**Why only a linear head?**

* The LM backbone already extracts features. The head's job is just to produce a scalar summary.
* Non-linear heads (e.g., 2-layer MLP) can overfit and extract spurious patterns.
* Linear simplicity helps with interpretability.

**Initialization matters:**

* Initialize W ≈ N(0, 0.01) -- small values ensure rewards start near 0.
* Initialize b = 0 -- no prior bias.
* This prevents early training dominated by the bias term.

**What about residual connections?**

* Some architectures add r = h\_N + Wh\_N (additive residual).
* Rarely used. Keep it simple.

## Reward Model Architecture (Full Pipeline)[#](#reward-model-architecture-full-pipeline)

Reward Model

Input Tokens  
prompt+response

Language Model  
Backbone

Hidden States  
h\_1...h\_N

Last Token h\_N  
∈ ℝ^d

Linear Head  
W, b

Reward r  
∈ ℝ

reward\_model\_architecture.py

```
import numpy as np

class RewardModel:
  """
  Reward Model Architecture.

  Architecture:
  1. Language Model backbone (pretrained, can fine-tune)
  2. Remove LM head (no token prediction needed)
  3. Add scalar projection head
  4. Take hidden state at last token
  """

  def __init__(self, hidden_dim=4096):
      self.hidden_dim = hidden_dim

      # Pretrained backbone weights (simplified - just attention output)
      self.backbone_weights = np.random.randn(hidden_dim, hidden_dim) * 0.02

      # Scalar head: maps hidden state to single scalar
      self.reward_head = np.random.randn(hidden_dim, 1) * 0.01
      self.reward_bias = np.zeros(1)

  def encode(self, tokens):
      """
      Get hidden state for token sequence.
      (Simplified - real implementation would use transformer)
      """
      # Pretend we have sequence of hidden states
      seq_len = len(tokens)
```

## Loss Function Implementation[#](#loss-function-implementation)

rm\_loss.pycpu-only

```
import numpy as np

def reward_model_loss(r_chosen, r_rejected, margin=0.0):
  """
  Bradley-Terry loss for reward model training.

  L = -log σ(r_chosen - r_rejected - margin)

  Args:
      r_chosen: Reward for chosen response
      r_rejected: Reward for rejected response
      margin: Optional margin (for margin-based training)

  Returns:
      loss: Scalar loss value
      grad_chosen: Gradient w.r.t. r_chosen
      grad_rejected: Gradient w.r.t. r_rejected
  """
  diff = r_chosen - r_rejected - margin

  # Numerically stable sigmoid
  if diff > 0:
      p = 1 / (1 + np.exp(-diff))
      loss = -np.log(p)
  else:
      p = np.exp(diff) / (1 + np.exp(diff))
      loss = -diff + np.log(1 + np.exp(diff))

  # Gradient: dL/d(r_chosen) = -(1 - p) = p - 1
  grad_chosen = p - 1
```

## Sequence-Level vs Token-Level Rewards[#](#sequence-level-vs-token-level-rewards)

Most modern reward models compute **sequence-level** (or **response-level**) rewards: a single scalar for the entire response.

Token-Level Reward

Response: Do you like cats?  
Yes, cats are great!

Hidden states:  
h\_1, h\_2, ... h\_N

Project each h\_i

r = [0.2, 0.5, 0.8, 0.9]  
per-token scores

Sequence-Level Reward

Response: Do you like cats?  
Yes, cats are great!

Hidden states for  
entire response

Take LAST h\_N

Project to scalar r

r = 0.87 for  
entire response

**Sequence-level (standard):**

* One scalar reward for the entire response
* Clean Bradley-Terry training
* What OpenAI, Anthropic use in practice

**Token-level (rare):**

* Separate reward for each position
* Can credit/blame specific words
* Much harder to train, sparse signals
* Only used for special cases (e.g., constrained decoding)

We'll focus on **sequence-level** in this lesson (standard approach).

## Gradient Analysis[#](#gradient-analysis)

gradient\_analysis.pycpu-only

```
import numpy as np

def analyze_gradient_flow():
  """
  Understand what the model learns from each preference pair.
  """
  print("Gradient Flow in Reward Model Training")
  print("=" * 60)
  print()

  scenarios = [
      {
          "name": "Easy distinction",
          "r_chosen": 2.0,
          "r_rejected": -1.0,
          "description": "Model already prefers chosen strongly"
      },
      {
          "name": "Wrong prediction",
          "r_chosen": -0.5,
          "r_rejected": 1.5,
          "description": "Model incorrectly prefers rejected"
      },
      {
          "name": "Uncertain",
          "r_chosen": 0.1,
          "r_rejected": 0.0,
          "description": "Model is unsure"
      },
  ]
```

## Break It: Reversed Labels[#](#break-it-reversed-labels)

break\_it\_reversed.pycpu-only

```
import numpy as np

def simulate_reversed_training(num_steps=100, flip_labels=False):
  """
  Show what happens when preference labels are reversed.
  """
  np.random.seed(42)

  # Simple "true" quality: longer is better (for demonstration)
  def true_quality(response_length):
      return response_length / 50  # Normalize

  # Initialize reward model (single parameter for simplicity)
  w = 0.0  # Weight: reward = w * length

  losses = []
  predictions = []

  for step in range(num_steps):
      # Generate preference pair
      len_a = np.random.randint(10, 100)
      len_b = np.random.randint(10, 100)

      # True preference: longer is better
      if len_a > len_b:
          chosen_len = len_a
          rejected_len = len_b
      else:
          chosen_len = len_b
          rejected_len = len_a
```

## Multi-Objective Reward Models[#](#multi-objective-reward-models)

Real systems often care about multiple dimensions: **helpfulness**, **harmlessness**, **factuality**, **length-of-reasoning**, etc.

Response

Backbone  
LM

h\_N

Split

Head Helpfulness  
W\_h

Head Creativity  
W\_c

Head Safety  
W\_s

r\_helpfulness

r\_creativity

r\_safety

Combine

r\_final = α\*r\_h

+ β\*r\_c + γ\*r\_s

**Multi-head architecture:**

* One backbone (shared feature extraction)
* Multiple reward heads (one per dimension)
* Mix with learnable or fixed weights

**Advantages:**

* Explicit tradeoff control (adjust α, β, γ)
* Train on data that only labels one dimension
* Easier to diagnose which dimension fails

**Challenges:**

* Requires multi-labeled data (expensive)
* Correlated dimensions (helpfulness often correlates with length)
* How to combine? Linear sum? Pareto frontier?

## Diagnosing Training Issues[#](#diagnosing-training-issues)

diagnosing\_rm.pycpu-only

```
import numpy as np

def diagnose_reward_model(train_losses, val_losses, accuracies):
  """
  Diagnose common reward model training issues from metrics.
  """
  issues = []

  # Check for overfitting
  if len(train_losses) > 10:
      late_train_loss = np.mean(train_losses[-10:])
      late_val_loss = np.mean(val_losses[-10:])
      if late_val_loss > late_train_loss * 1.2:
          issues.append(("OVERFITTING", "Val loss >> Train loss. Try more regularization, less epochs, or more data."))

  # Check for underfitting
  if accuracies[-1] < 0.6:
      issues.append(("UNDERFITTING", "Accuracy < 60%. Model isn't learning. Check data quality, learning rate, or model capacity."))

  # Check for label noise
  if np.mean(accuracies) < 0.7 and np.std(accuracies) < 0.02:
      issues.append(("LABEL_NOISE", "Low accuracy with low variance. Labels may be noisy or task too hard."))

  # Check for collapse
  if len(set([round(a, 2) for a in accuracies[-10:]])) == 1:
      issues.append(("COLLAPSE", "Accuracy stuck at single value. Check for degenerate solutions."))

  return issues

# Simulated training run
```

## Break It: Preference Inconsistency[#](#break-it-preference-inconsistency)

What if human raters disagree? A > B and B > C, but C > A (cycle)?

break\_it\_cycles.pycpu-only

```
import numpy as np

def train_with_cycles(consistent=True, num_cycles=10):
  """
  Train reward model on data with preference cycles.
  """
  np.random.seed(42)
  w = np.array([0.0])  # Single parameter for simplicity

  losses_per_cycle = []

  # Create some preferences that VIOLATE transitivity
  prefs_A_B = (1.0, 0.0)  # A > B
  prefs_B_C = (1.0, 0.0)  # B > C

  if not consistent:
      # Cycle: C > A (contradicts transitivity!)
      prefs_C_A = (1.0, 0.0)
  else:
      # Consistent: A > C (as expected from transitivity)
      prefs_C_A = (1.0, 0.0)

  for cycle in range(num_cycles):
      losses = []

      # Train on cyclic preferences
      for (r_chosen, r_rejected) in [prefs_A_B, prefs_B_C, prefs_C_A]:
          r_chosen = r_chosen + w[0]
          r_rejected = r_rejected + w[0] - 0.5
```

## Break It: Prompt Overfitting[#](#break-it-prompt-overfitting)

Can a reward model memorize prompts instead of learning response quality?

break\_it\_overfitting.pycpu-only

```
import numpy as np

def simulate_prompt_overfitting(epochs=50):
  """
  Show how reward model can overfit to prompt patterns.
  """
  np.random.seed(42)

  # Simple: 2 prompts, 2 responses each
  # Real data: helpful responses get reward 1, bad get 0
  # But maybe prompt 1 responses are always longer (spurious)

  # Prompt 1: asks for explanation (get longer, better responses)
  # Prompt 2: asks for yes/no (get short, worse responses)

  # Model learns: "I give high reward to LONG responses"
  # But causality is backwards! Length doesn't cause quality.

  w_length = 0.0  # Weight on length
  w_quality = 0.0  # Weight on actual quality

  train_losses = []
  test_accs = []

  for epoch in range(epochs):
      train_loss = 0

      # TRAIN on Prompt 1 (long = good) and Prompt 2 (short = bad)
      for prompt_id in [1, 2]:
          # True correlation: quality and length are correlated in train
```

## Scale Thought Experiment[#](#scale-thought-experiment)

Data Size

10K pairs  
Fine-tune from SFT

50K pairs  
Train from scratch or  
light fine-tune

500K+ pairs  
Large backbone  
Heavy regularization

Multi-model  
Ensemble

| Data Size | Architecture Choice | Key Consideration |
| --- | --- | --- |
| **10K preferences** | Fine-tune from SFT checkpoint, small model OK | Use same seed distribution as policy will see |
| **50K preferences** | Can train from scratch on smaller backbone | Start splitting train/val (~40/10K) |
| **500K+ preferences** | May need larger backbone, careful regularization | Watch for overfitting to spurious patterns |
| **1M+ preferences** | Multi-model ensemble, multiple architectures | Average predictions, catch reward hacking attempts |

**Key scaling insight:** More data != better RM. A 1B model on 50K carefully-curated preferences beats a 7B model on 500K noisy preferences.

## Production Reality[#](#production-reality)

**OpenAI (InstructGPT):**

* 6B parameter reward model
* Trained on 33K human comparisons
* Separate RMs for helpfulness vs harmlessness (early approach)
* Key detail: RM was trained on curated, high-agreement examples

**Anthropic (Constitutional AI):**

* Reward models as part of alignment pipeline
* AI-generated preferences supplement human data
* Ensemble of RMs to reduce exploitation and improve robustness
* Published research on red-teaming RMs to find failure modes

**Deepseek, Meta (LLaMA):**

* Train on 100K+ paired responses
* RMs used alongside ranking (not just binary comparison)
* Heavy emphasis on data quality over quantity
* Multiple passes of annotation for difficult cases

**Common practices across labs:**

* Start from SFT model (same initialization as policy)
* Use held-out set for periodic calibration checking
* Regularize heavily (L2, dropout on head, early stopping)
* Monitor train/val divergence carefully
* Ensemble multiple RMs to reduce variance
* Explicitly check for reward hacking on red-teaming sets

**Why production RMs are fragile:**

* Policy training will exploit any RM weakness
* Small label errors get amplified via RLHF
* Reward models are known to be unreliable at OOD (out-of-distribution) examples
* Single-task RMs don't generalize across prompt domains

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. A reward model assigns r(chosen) = 2.5 and r(rejected) = 1.0. Compute P(chosen > rejected) using the Bradley-Terry formula. Then compute the loss for this example. What happens to the loss if you double both rewards to 5.0 and 2.0?
2. Your reward model backbone is a 7B parameter transformer with hidden\_dim = 4096. The scalar head is a single linear layer mapping from hidden\_dim to 1. How many trainable parameters does the head add? What fraction of total model parameters is this?
3. During RM training on 50K preference pairs, train accuracy reaches 85% but validation accuracy plateaus at 62%. Diagnose the most likely issue. What are two concrete changes you would make to the training configuration?