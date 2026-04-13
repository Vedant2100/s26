In this tutorial, you will compute KL divergence between a policy and reference model at the token level, implement an adaptive KL controller, and observe how different beta values affect the reward-vs-drift tradeoff. You will simulate RLHF training with and without KL penalty to see mode collapse firsthand. By the end, you will be able to set a KL target, diagnose common KL pathologies (zero, unbounded, oscillating), and choose between penalty and constraint approaches.

## Prerequisites Refresher[#](#prerequisites-refresher)

Before diving into KL penalties, let us refresh the key concepts you will need:

▶What is KL Divergence?

KL divergence (Kullback-Leibler divergence) measures how different one probability distribution is from another. For discrete distributions P and Q:

```
KL(P || Q) = Σ_i P(i) · log(P(i) / Q(i))
```

**Key properties:**

* **Asymmetric**: KL(P || Q) ≠ KL(Q || P)
* **Non-negative**: KL(P || Q) ≥ 0, with equality iff P = Q
* **Unbounded**: Can be arbitrarily large
* **Interprets as**: "Expected log probability ratio under P"

In RLHF, we care about KL(π\_policy || π\_reference) — how much the policy distribution differs from the reference.

▶What is a Probability Distribution Over Tokens?

When we generate text token-by-token, the model outputs a distribution `π(y_t | x, y_prev)` — the probability of each possible next token given:

* The prompt/context x
* The tokens generated so far `y_prev` (all tokens before position t)

We typically work in log-space: log π(y\_t | ...) is more numerically stable and easier to work with mathematically.

For a sequence y = [y\_1, y\_2, ..., y\_n], the joint probability is:

```
P(y | x) = ∏_t π(y_t | x, y_{<t})
log P(y | x) = Σ_t log π(y_t | x, y_{<t})
```

This is why we compute KL *per-token* and sum: KL divergence decomposes additively across the sequence.

▶What is the Reference Model?

The reference model (π\_ref) is typically your **SFT (Supervised Fine-Tuned) checkpoint** — the model after instruction tuning on human demonstrations, before RLHF.

Why the SFT model?

* It already produces coherent, helpful outputs
* It has learned human preferences implicitly
* We want to "improve" it, not replace it

Alternative reference models:

* Earlier RLHF checkpoint (iterative improvement)
* Ensemble of multiple SFT models (diverse baseline)
* Pre-trained base model (more aggressive regularization)

The choice of reference dramatically affects what behaviors are preserved.

▶Why Can't We Just Maximize Reward?

In principle, yes—but the reward model is learned from limited human feedback. It has:

* **Coverage gaps**: Unseen regions of behavior space
* **Overfitting**: Learned spurious correlations
* **Distributional shift**: Training on SFT outputs, test on policy outputs

Maximizing a flawed reward model alone → degenerate solutions that exploit weaknesses.

## The Mode Collapse Problem[#](#the-mode-collapse-problem)

mode\_collapse.pycpu-only

```
import numpy as np

def demonstrate_mode_collapse():
  """
  Show what happens without KL penalty.
  """
  print("Mode Collapse Without KL Penalty")
  print("=" * 60)
  print()

  # Simulate a reward model with exploitable patterns
  def reward_model(response):
      score = 0
      # RM accidentally rewards repetition
      words = response.lower().split()
      if len(words) > 0:
          repetition = 1 - len(set(words)) / len(words)
          score += repetition * 2

      # RM rewards certain phrases
      if "absolutely" in response.lower():
          score += 0.5
      if "definitely" in response.lower():
          score += 0.5

      return score

  # Without KL: policy learns to exploit
  exploiting_responses = [
      "Absolutely absolutely absolutely definitely definitely",
```

## Why KL Constraint is Needed[#](#why-kl-constraint-is-needed)

When you fine-tune via RLHF, you face a fundamental dilemma:

**The Instability Principle:**
The reward model is trained on SFT-generated text. When you update the policy, you enter new regions of behavior space—places the reward model never saw during training. In these regions, the reward model's predictions become unreliable.

Without regularization, the policy does this:

1. Finds outputs the reward model scores highly
2. Exploits artifacts in the reward model
3. Drifts arbitrarily far from baseline
4. Produces coherent-sounding but factually broken outputs

**The KL Penalty Solution:**
Adding a KL penalty says: "You can improve, but you must stay close to the SFT model." This creates a regularization budget—you trade reward points for staying sane.

Think of it like a trust region: the reward model is trustworthy within a certain distance from the SFT distribution. Beyond that distance, the penalty grows, forcing the policy to stop.

## The KL-Regularized Objective[#](#the-kl-regularized-objective)

kl\_objective.pycpu-only

```
import numpy as np

def derive_kl_objective():
  """
  Derive the KL-regularized RLHF objective.
  """
  print("KL-Regularized RLHF Objective")
  print("=" * 60)
  print()

  print("Without KL penalty:")
  print("  J(θ) = E_{y~π_θ}[R(x, y)]")
  print()

  print("With KL penalty:")
  print("  J(θ) = E_{y~π_θ}[R(x, y)] - β · KL(π_θ || π_ref)")
  print()

  print("Where:")
  print("  - π_ref is the reference policy (usually SFT model)")
  print("  - β is the KL coefficient (controls regularization strength)")
  print("  - KL(π_θ || π_ref) = E_{y~π_θ}[log π_θ(y|x) - log π_ref(y|x)]")
  print()

  print("Equivalent per-token formulation:")
  print("  R_total(x,y) = R(x,y) - β · Σ_t [log π_θ(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]")
  print()

  print("Effect:")
  print("  - High KL → penalty increases → policy stays close to reference")
```

### Mathematical Intuition[#](#mathematical-intuition)

The KL-regularized objective is:

```
J(θ) = E_{y ~ π_θ(·|x)}[R(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))]
```

Expanding the KL term:

```
KL(π_θ || π_ref) = E_{y ~ π_θ}[log π_θ(y|x) - log π_ref(y|x)]
```

So the full objective becomes:

```
J(θ) = E_{y ~ π_θ}[R(x, y) - β · log(π_θ(y|x) / π_ref(y|x))]
```

**Intuition:**

* If π\_θ(y|x) > π\_ref(y|x): we are making outputs *more* likely → KL penalty is positive (costs us)
* If `π_θ(y|x) < π_ref(y|x)`: we are making outputs *less* likely → KL penalty is negative (helps us)
* β controls the trade-off strength

When β is large, the penalty dominates, and the policy barely changes from reference. When β is small, the policy can drift further if it finds high rewards.

With KL

No KL

Effective Landscape

KL Landscape

Reward Landscape

Added Constraint

Regularization

Mode Collapse

Stable

Reward R(x,y)

KL Penalty β·KL(π||π\_ref)

Objective J = R - β·KL

Climb reward gradient  
to any height

Climb gradient within  
trust region

Exploitation

Improvement

## Computing KL Divergence[#](#computing-kl-divergence)

In practice, we compute KL on a per-token basis because sequences have variable length. The KL between two full sequences is simply the sum of per-token KLs:

```
KL_sequence = Σ_{t=1}^{T} [log π_θ(y_t | x, y_{<t}) - log π_ref(y_t | x, y_{<t})]
```

This is key: **KL is additive across tokens**. A 100-token sequence has roughly 5x more KL than a 20-token sequence (all else equal), which is why longer outputs get penalized more heavily.

kl\_computation.pycpu-only

```
import numpy as np

def compute_kl_per_token(
  log_probs_policy: np.ndarray,
  log_probs_ref: np.ndarray
) -> np.ndarray:
  """
  Compute per-token KL divergence.

  For each token position t:
    kl_t = log π_θ(y_t|...) - log π_ref(y_t|...)

  Mathematically, this is the "pointwise KL" or "log likelihood ratio".
  """
  return log_probs_policy - log_probs_ref

def compute_total_kl(log_probs_policy: np.ndarray,
                   log_probs_ref: np.ndarray) -> float:
  """
  Compute total KL divergence for a sequence.
  """
  per_token = compute_kl_per_token(log_probs_policy, log_probs_ref)
  return float(np.sum(per_token))

def kl_penalty_reward(reward, log_probs_policy, log_probs_ref, beta):
  """
  Compute KL-penalized reward.

  penalized = original_reward - β * total_kl
  """
```

### Understanding Per-Token KL[#](#understanding-per-token-kl)

Let us break down what's happening at the token level:

token\_level\_kl.pycpu-only

```
import numpy as np

def analyze_token_level_kl():
  """
  Show KL divergence at individual token positions.
  """
  np.random.seed(42)

  print("Token-Level KL Analysis")
  print("=" * 70)
  print()

  seq_len = 10
  log_probs_ref = np.array([-3.2, -3.5, -2.8, -4.1, -3.0,
                            -3.3, -2.9, -3.8, -3.1, -3.4])
  log_probs_policy = np.array([-3.1, -4.2, -2.7, -3.9, -3.5,
                               -3.2, -3.4, -4.0, -3.0, -3.8])

  print("Token | log π_ref | log π_θ | Diff (policy - ref) | Interpretation")
  print("-" * 70)

  total_kl = 0
  for i, (ref, policy) in enumerate(zip(log_probs_ref, log_probs_policy)):
      kl_token = policy - ref
      total_kl += kl_token

      # Interpret the KL at this position
      if kl_token > 0.2:
          interp = "Policy MORE likely (positive KL cost)"
      elif kl_token < -0.2:
```

## Adaptive KL Coefficient[#](#adaptive-kl-coefficient)

Fixed β values have a major problem: **you do not know what the right value is ahead of time**. A β that works for one model/dataset might be terrible for another.

The solution used by OpenAI and Anthropic: **adaptive KL control**. Instead of fixing β, you pick a target KL value (e.g., 6.0 nats) and adjust β dynamically to keep the observed KL near that target.

**The feedback loop:**

1. Run policy gradient step
2. Measure KL divergence between updated policy and reference
3. If KL > target: increase β (add more penalty)
4. If KL < target: decrease β (allow more freedom)
5. Repeat

This is analogous to PID control in robotics—you have a desired setpoint (target KL) and adjust the control signal (β) to maintain it.

adaptive\_kl.pycpu-only

```
import numpy as np

class AdaptiveKLController:
  """
  Adaptively adjust KL coefficient to target a specific KL value.

  This is similar to proportional control in control theory:
  - Error = observed_kl - target_kl
  - Adjust beta proportionally to the error
  """

  def __init__(self, init_beta=0.1, target_kl=6.0, horizon=10000):
      self.beta = init_beta
      self.target_kl = target_kl
      self.horizon = horizon  # Timescale for adaptation

  def update(self, observed_kl):
      """
      Update beta based on observed KL.

      If KL > target: increase beta (penalize divergence more)
      If KL < target: decrease beta (allow policy more freedom)
      """
      # Compute error
      error = observed_kl - self.target_kl

      # Proportional update: scale error by horizon
      # Larger horizon = slower adaptation
      proportional_update = error / self.horizon
```

### Why Adaptive KL Works[#](#why-adaptive-kl-works)

Compute

KL > target

KL < target

KL ≈ target

Continue

Policy Update Step

Compute KL(π\_new || π\_ref)

KL vs Target?

Increase β

Decrease β

Next Update

## KL Penalty vs Constraint[#](#kl-penalty-vs-constraint)

There are two main approaches to regularizing RLHF: **penalty** (what we've been discussing) and **constraint** (TRPO-style).

penalty\_vs\_constraint.pycpu-only

```
def compare_penalty_constraint():
  """
  Compare KL penalty vs KL constraint approaches mathematically
  and practically.
  """
  print("KL Penalty vs KL Constraint")
  print("=" * 70)
  print()

  print("APPROACH 1: KL Penalty (Standard in RLHF)")
  print("-" * 70)
  print("  Objective: J(θ) = E[R] - β · KL(π_θ || π_ref)")
  print()
  print("  Pros:")
  print("    + Simple to implement (just subtract penalty)")
  print("    + Single hyperparameter (β)")
  print("    + Can be computed per-token")
  print("    + Allows trade-off tuning")
  print()
  print("  Cons:")
  print("    - β requires tuning or adaptive control")
  print("    - KL can overshoot if reward signal is strong")
  print("    - No hard guarantee on KL magnitude")
  print()

  print("APPROACH 2: KL Constraint (TRPO-style)")
  print("-" * 70)
  print("  Objective: maximize E[R]")
  print("             subject to: KL(π_θ || π_ref) ≤ δ")
  print()
```

## Break It: No KL Penalty[#](#break-it-no-kl-penalty)

What happens if you remove the KL penalty entirely? The policy finds degenerate solutions that exploit the reward model's weaknesses.

break\_it\_no\_kl.pycpu-only

```
import numpy as np

def simulate_rlhf_with_without_kl():
  """
  Simulate RLHF training with and without KL penalty.
  Shows how the policy diverges from the reference when unregularized.
  """
  np.random.seed(42)

  print("RLHF Training: With vs Without KL Penalty")
  print("=" * 70)
  print()
  print("Scenario: Reward model has a flaw—it gives high scores to")
  print("          repetitive/exploitative outputs, but these are useless.")
  print()

  def simulate_training(use_kl, beta=0.1, num_steps=40):
      # Policy state: probability of producing "good" vs "exploiting" output
      p_good = 0.8  # Start close to reference (SFT model)

      quality_history = []
      reward_history = []
      kl_history = []

      for step in range(num_steps):
          # Reward model (intentionally flawed)
          reward_good = 5.0         # Good outputs score moderately
          reward_exploit = 6.5      # Bad outputs score higher!

          # Current expected reward
```

### Break It: What If β Is Too High?[#](#break-it-what-if-is-too-high)

break\_it\_high\_beta.pycpu-only

```
import numpy as np

def demonstrate_high_beta_problem():
  """
  Show what happens when β is so high that the KL penalty dominates.
  """
  np.random.seed(42)

  print("Break It: KL Coefficient Too High")
  print("=" * 70)
  print()
  print("If β is set too high, the KL penalty dominates the objective,")
  print("and the policy barely moves from the reference model.")
  print()

  def simulate_with_beta(beta, num_steps=30):
      p_good = 0.8

      quality_history = []
      for step in range(num_steps):
          reward_good = 5.0
          reward_exploit = 6.5

          expected_reward = p_good * reward_good + (1 - p_good) * reward_exploit
          kl = max(0, (1 - p_good) * 2.5)

          total_objective = expected_reward - beta * kl

          # Update policy
          delta_p = 0.03
```

## Real-World Failure Modes[#](#real-world-failure-modes)

Here are the most common ways KL penalties go wrong in practice:

▶Failure Mode #1: β Too Small (Reward Hacking)

When β is too small, the KL penalty is negligible, and the policy ignores the regularization. This leads to classic reward hacking:

* **Symptom**: Model outputs high-scoring but incoherent text
* **Example**: Reward model rewards length → outputs 10,000 repetitions of "I agree"
* **Root cause**: β is too low relative to the reward signal magnitude
* **Fix**: Increase β or use adaptive KL with a higher target

The reward model gives scores in range [−5, 5], and β · KL penalty is [0, 0.01]. The policy rightly ignores the penalty.

▶Failure Mode #2: β Too Large (No Learning)

When β is too large, the KL penalty dominates, and the policy barely moves from reference. You pay compute to learn almost nothing.

* **Symptom**: Model outputs are nearly identical to SFT checkpoint
* **Example**: After 10k training steps, reward has barely improved
* **Root cause**: β so high that any update incurs huge penalty
* **Fix**: Decrease β or increase target KL in adaptive controller

You are essentially telling the policy: "Stay exactly where you are." The reward signal cannot overcome the penalty.

▶Failure Mode #3: Target KL Mismatched to Reward Magnitude

Adaptive KL assumes the reward model outputs are calibrated. If the reward model's scale is weird, the target KL might be off.

* **Symptom**: KL bounces wildly; β keeps changing
* **Root cause**: Reward model gives scores in [0, 1] while target KL is 6
* **Fix**: Normalize rewards; match target KL to typical reward magnitudes

Example: If your RM gives scores [0, 0.1], a target KL of 6 is impossibly high. Expected reward gain might be 0.01 while KL penalty is 0.6. Reduce target to 1–2.

▶Failure Mode #4: Reference Model Drift

If you update your reference model mid-training, the KL baseline shifts. Suddenly, old policies look close, and new updates get penalized harshly.

* **Symptom**: KL spikes unexpectedly; training destabilizes
* **Root cause**: Reference updated but policy hasn't caught up
* **Fix**: Keep reference model fixed, or gradually interpolate to new reference

OpenAI handles this by keeping reference fixed during a training run. Only update it between major rounds.

## Scale Thought Experiment: What Changes at Scale?[#](#scale-thought-experiment-what-changes-at-scale)

| Scenario | β Guidance | Why |
| --- | --- | --- |
| **Small model** (125M) | 0.05–0.15 | Reward model is crude; needs more regularization |
| **Medium model** (7B) | 0.10–0.20 | Better balance between learning and stability |
| **Large model** (70B) | 0.15–0.30 | Reward model is more reliable; can afford higher β |
| **Long context** (8k+ tokens) | Higher β | Longer sequences → more total KL; higher penalty |
| **Adaptive (any scale)** | 0.01–1.0 (auto-tuned) | Industry standard; removes manual tuning |

**Why adaptive is crucial at scale:**

* Different datasets have different natural KL ranges
* Different reward models have different difficulty
* As policy improves, KL behavior changes
* Manual β becomes a bottleneck for iteration

## Production Reality[#](#production-reality)

Practical Systems

Monitor KL metrics

Alert on divergence

Automated rollback

A/B test β values

Anthropic

KL penalty

+ Constitutional AI

Self-critique reduces drift

Iterative reference updates

OpenAI (InstructGPT)

Adaptive KL

Target ≈ 6.0 nats

Multiple β per task

Periodic ref resets

**Key Production Insights:**

1. **OpenAI (InstructGPT):**

   * Used adaptive KL with target around 6.0 nats
   * Different β for different tasks/domains (DaVinci, Codex, etc.)
   * Periodic resets to newer reference models to capture iterative improvement
2. **Anthropic:**

   * KL penalty + Constitutional AI principles work synergistically
   * Constitutional self-critique acts as a natural KL dampener
   * Iterative refinement with updated references (policy from previous stage becomes new reference)
3. **Meta/Research:**

   * Close monitoring of KL metrics during training
   * Automated alerts when KL exceeds bounds
   * Rollback mechanisms for failures
4. **General Best Practices:**

   * Always use adaptive KL for production
   * Target KL typically 6–12 nats (dataset dependent)
   * Log KL curves per-batch for debugging
   * Use separate β for different deployment branches

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. A policy generates a 20-token response. The per-token log probabilities under the policy are each -3.0, and under the reference they are each -3.2. Compute the per-token KL, the total sequence KL, and the KL penalty with beta = 0.1. If the reward model gives this response R = 5.0, what is the penalized reward?
2. Your adaptive KL controller has target\_kl = 6.0 nats. At step 100, observed KL = 12.0 nats and current beta = 0.10. Should beta increase or decrease? If the controller uses proportional update with horizon = 5000, compute the new beta value.
3. You run RLHF for 1000 steps and observe KL stuck at 0.01 nats. Diagnose the most likely cause. Then separately consider: KL has grown to 50 nats. What is happening, and what is your first fix?

## Research Hooks[#](#research-hooks)

**Optimal KL target:**
What's the right KL budget? 6 nats? 10? 15? Too low prevents learning; too high enables hacking. Can we determine this automatically from the reward model's calibration?

**Reference model selection:**
Should the reference always be SFT? What if you use an earlier RLHF checkpoint? An ensemble? This choice affects what behaviors are preserved versus optimized away.

**KL vs other regularizers:**

* Could we use entropy regularization instead? Why not?
* Does KL penalty interact with other forms of regularization (weight decay, layer norm regularization)?
* Is there a better divergence than KL for this purpose (Wasserstein, MMD)?

**Curriculum learning + KL:**

* Start with high β (no learning) and decay it as training progresses
* Or warm-start from SFT and gradually increase reward signal?
* How does curriculum interact with adaptive KL?

---

## Deep Dive: Computing KL in a Real RLHF Loop[#](#deep-dive-computing-kl-in-a-real-rlhf-loop)

In practice, computing KL means running both the policy and reference model on the same batch, comparing their log probabilities token-by-token.

rlhf\_kl\_computation.pycpu-only

```
import numpy as np

def compute_kl_for_batch(
  batch_log_probs_policy: np.ndarray,
  batch_log_probs_ref: np.ndarray,
  batch_mask: np.ndarray = None
):
  """
  Compute KL divergence for a batch of sequences.

  Args:
      batch_log_probs_policy: [batch_size, seq_len] log probabilities from policy
      batch_log_probs_ref: [batch_size, seq_len] log probabilities from reference
      batch_mask: [batch_size, seq_len] binary mask (1 = real token, 0 = padding)

  Returns:
      Dictionary with per-sequence and batch statistics
  """
  batch_size, seq_len = batch_log_probs_policy.shape

  # Compute per-token KL
  per_token_kl = batch_log_probs_policy - batch_log_probs_ref  # [batch_size, seq_len]

  # Apply mask if provided (ignore padding)
  if batch_mask is not None:
      per_token_kl = per_token_kl * batch_mask

  # Sum across tokens for each sequence
  per_seq_kl = np.sum(per_token_kl, axis=1)  # [batch_size]
```

## Monitoring KL During Training[#](#monitoring-kl-during-training)

High

Low

Target

Continue

Training Loop

Forward pass  
Policy and Reference

Compute mean KL  
across batch

Log KL metric

KL vs  
Target?

Update β via  
adaptive controller

Policy Gradient  
Update

Next batch

In production systems, KL is one of the most important metrics to monitor:

**What to log:**

* Mean KL per batch
* Max/min KL per batch
* KL std deviation (should be stable)
* Current β value
* Target KL

**Warning signs:**

* KL growing unbounded (reward hacking)
* KL stuck at zero (no learning)
* β oscillating wildly (target KL wrong)
* Sudden KL spike (reference model updated)

## Real-World Example: Tuning β for a 7B Model[#](#real-world-example-tuning-for-a-7b-model)

Let us walk through a realistic scenario: you've trained a reward model, and now you are starting RLHF on a 7B parameter model. What β should you use?

beta\_tuning\_example.pycpu-only

```
import numpy as np

class RLHFSimulator:
  """
  Simulate RLHF training loop to demonstrate KL and β dynamics.
  """

  def __init__(self, init_beta=0.1, target_kl=6.0, reward_mean=0.0, reward_std=1.0):
      self.beta = init_beta
      self.target_kl = target_kl
      self.reward_mean = reward_mean
      self.reward_std = reward_std
      self.horizon = 5000

  def step(self, observed_kl, observed_reward):
      """
      Simulate one training step.
      """
      # Adaptive KL update
      error = observed_kl - self.target_kl
      self.beta = max(0.001, self.beta * (1 + error / self.horizon))

      # Objective
      objective = observed_reward - self.beta * observed_kl

      return dict(
          beta=self.beta,
          objective=objective,
          kl=observed_kl,
          reward=observed_reward,
```

### β Tuning Heuristics[#](#tuning-heuristics)

Based on empirical work across labs, here's a practical guide:

No

Yes

Weak

Strong

Fast

Slow

Start RLHF  
on 7B model

RM trained well?

Reward signal  
strong?

Early KL  
behavior?

High KL  
β ← 0.20

Low KL  
β ← 0.05

Use adaptive  
KL

Set target\_kl  
6–8 nats

**Decision rules:**

| Condition | Recommendation |
| --- | --- |
| RM is weak/noisy | Start high (0.2), use adaptive KL |
| RM is strong | Start moderate (0.1), use adaptive KL |
| Early KL grows fast | Lower target\_kl to 4–5 |
| Early KL grows slow | Higher target\_kl to 8–10 |
| Reward signal weak | Increase β (encourage stability over improvement) |
| Reward signal strong | Decrease β (encourage learning) |
| Always | Use adaptive KL with monitoring |

## The Theoretical Connection to Trust Regions[#](#the-theoretical-connection-to-trust-regions)

The KL penalty is related to Trust Region Policy Optimization (TRPO), a fundamental RL algorithm. Understanding this connection gives you intuition for why KL matters.

**TRPO Constraint:**

```
maximize E[A(s,a) · log π(a|s)]
subject to: KL(π_old || π) ≤ δ
```

TRPO says: "Improve advantage, but stay within a KL ball of the old policy." This is theoretically justified because inside a small KL ball, the advantage estimate is reliable.

**RLHF Penalty (Approximates TRPO):**

```
maximize E[R(x,y) - β · KL(π || π_ref)]
```

By adding a KL penalty, you are approximating a trust region: the policy can improve reward as much as it wants, *but* has to pay in KL currency. Larger β → tighter trust region.

**Why approximate?** Direct constraint optimization (TRPO-style) is hard at scale. Penalty methods are simpler and work empirically.

## Debugging KL Issues: A Practical Troubleshooting Guide[#](#debugging-kl-issues-a-practical-troubleshooting-guide)

When something goes wrong with KL during RLHF, here's how to diagnose it:

▶Issue: KL stays at 0 or very close to 0

**What it means:** Policy and reference are nearly identical. Model is not learning.

**Diagnosis:**

* Check if you are computing KL correctly (sum over tokens)
* Verify policy and reference models are different
* Check if reward signal is too weak

**Fix:**

* Lower β to allow more learning
* If using adaptive, lower target\_kl
* Check reward model's reward range

▶Issue: KL grows unbounded (10, 20, 50+)

**What it means:** Policy is diverging far from reference. Likely mode collapse or exploitation.

**Diagnosis:**

* Check if β is 0 or very small
* Look at sample outputs—are they repetitive?
* Is reward signal flawed?

**Fix:**

* Increase β immediately
* If using adaptive, the controller should raise β automatically
* Lower target\_kl to a tighter constraint
* Manually inspect outputs

▶Issue: KL oscillates wildly between steps

**What it means:** Adaptive KL controller is hunting too aggressively.

**Diagnosis:**

* Horizon parameter might be too small
* Target KL might be mismatched to reward scale

**Fix:**

* Increase horizon (slower adaptation)
* Re-calibrate target\_kl
* Check reward model outputs are normalized

▶Issue: KL spikes suddenly mid-training

**What it means:** Something changed—reference model, batch distribution, or random seed.

**Diagnosis:**

* Did you update reference model?
* Did dataset change?
* Check for any recent code changes

**Fix:**

* Revert to previous reference
* Investigate data distribution
* Resume training and monitor closely

## Advanced: Per-Token vs Sequence KL[#](#advanced-per-token-vs-sequence-kl)

Sometimes you want to know *where* in the sequence the policy diverges most.

per\_token\_analysis.pycpu-only

```
import numpy as np

def analyze_kl_by_position():
  """
  Analyze KL divergence broken down by token position.
  Useful for understanding *where* the policy diverges.
  """
  np.random.seed(42)

  print("KL Divergence by Token Position")
  print("=" * 70)
  print()

  # Simulate a batch
  batch_size = 4
  seq_len = 20

  log_probs_ref = np.random.randn(batch_size, seq_len) * 0.5 - 3.5
  log_probs_policy = log_probs_ref + np.random.randn(batch_size, seq_len) * 0.2 + 0.1

  # Compute per-token KL (averaged across batch)
  per_token_kl = np.mean(log_probs_policy - log_probs_ref, axis=0)

  print("Average KL per token position:")
  print()
  print("%5s %10s %-40s" % ("Pos", "KL", "Interpretation"))
  print("-" * 55)

  for pos, kl in enumerate(per_token_kl):
      if kl > 0.15:
```

## Key Takeaways[#](#key-takeaways)

**Conceptually:**

* KL penalty prevents mode collapse by penalizing drift from reference
* It creates a trust region: reward is only trustworthy nearby
* Adaptive KL removes manual tuning, enabling scalable RLHF

**Practically:**

* Compute per-token KL, sum over sequences, average over batch
* Always use adaptive KL in production (target\_kl ≈ 6–10)
* Monitor KL curves; red flags include 0, unbounded, or oscillation
* Reference model must be stable (do not update mid-run)

**Intuitively:**

* Reward model is like a compass that only works nearby
* KL penalty is like a leash that keeps the policy close to the reference
* Longer leash (lower β) = more learning but more risk
* Shorter leash (higher β) = safer but slower improvement

**When debugging:**

* Zero KL → not learning
* Unbounded KL → mode collapse
* Oscillating KL → controller too aggressive
* Sudden spike → something changed

## Summary: The KL Penalty Checklist[#](#summary-the-kl-penalty-checklist)

When implementing RLHF with KL penalty:

✓ **Compute KL correctly**: per-token, sum across sequence, average across batch

✓ **Start with adaptive KL**: Manual β tuning is a dead end at scale

✓ **Target KL**: Typically 6–10 nats, but dataset-dependent. Experiment.

✓ **Monitor relentlessly**: Log KL every batch. Watch for red flags.

✓ **Reference model stability**: Don't update mid-run. Keep it fixed.

✓ **Check for mode collapse**: Manually inspect outputs. Is the model just repeating?

✓ **Balance reward and KL**: If reward improves but outputs degrade, β is too low.

✓ **Scale aware**: Different β values for different model sizes and reward models.

✓ **Debug systematically**: Use per-token KL analysis to find issues.

✓ **Document your β**: Log final β value and target\_kl for reproducibility.

---

*Next up: RLHF is notoriously unstable. Success requires careful hyperparameter selection, extensive monitoring, and debugging skills.*