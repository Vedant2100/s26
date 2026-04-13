In this tutorial, you will derive the DPO loss step by step, implement it from scratch, and measure how the beta parameter controls the tradeoff between learning from preferences and staying close to the reference policy.

This tutorial walks through the full derivation: from the KL-regularized objective to the Bradley-Terry substitution to the final loss. You will see exactly where the partition function cancels and why the reward model never appears explicitly.

## Prerequisites: Three Key Pieces[#](#prerequisites-three-key-pieces)

▶Bradley-Terry Model (Preference Likelihood)

The Bradley-Terry model converts preference pairs into a parametric likelihood:

Given that we prefer response y\_w over y\_l on prompt x, what's the probability of this preference under a parametric reward model r(x, y)?

**Answer:** The Bradley-Terry preference likelihood is:

```
P(y_w ≻ y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))
```

Why sigmoid? Because:

1. We want a probability (between 0 and 1)
2. The difference in rewards should directly control preference strength
3. Equal rewards → 50-50 preference (probability = 0.5 via sigmoid(0) = 0.5)
4. High reward difference → strong preference (sigmoid(∞) ≈ 1)

This is a simple and mathematically convenient model, widely used in ranking and preference learning.

▶KL Divergence & Regularization

KL divergence measures how much probability distribution Q differs from distribution P:

```
KL(Q || P) = ∑_i Q(i) log(Q(i) / P(i))
```

Properties:

* Always ≥ 0 (= 0 only when Q = P)
* Not symmetric: KL(Q || P) ≠ KL(P || Q)
* In our case: KL(π || π\_ref) = ∑\_y π(y|x) log(π(y|x) / π\_ref(y|x))

**Why regularize with KL?**

* Without it: policy could assign 100% probability to one response (unbounded reward)
* With KL term: policy must balance reward with staying close to reference
* Temperature parameter β controls the tradeoff: high β → stay close to reference (stronger KL penalty), low β → more drift from reference (weaker KL penalty)

This is the key to preventing "reward hacking" where the policy exploits the reward model.

▶Log-Probability Ratios & Policy Differences

A key insight: we'll be comparing policies using log-probability ratios.

For a sequence y given prompt x:

* π(y|x) is the sequence probability (product of token probabilities)
* log π(y|x) is the sequence log-probability (sum of token log-probs)
* log(π(y|x) / π\_ref(y|x)) is the **policy divergence**

Why log-ratios appear:

1. Numerically stable (log-probs are already computed during generation)
2. Naturally appear when taking derivatives of KL divergence
3. The ratio tells us "by what factor does this policy prefer this sequence more than reference?"

Example: If π(y|x) = 0.01 and π\_ref(y|x) = 0.001, then log-ratio ≈ 2.3 (policy prefers this by ~10x).

## The KL-Regularized Objective (Recap)[#](#the-kl-regularized-objective-recap)

kl\_objective\_recap.pycpu-only

```
import numpy as np

def recap_rlhf_objective():
  """
  Recall the RLHF objective we're trying to optimize.
  """
  print("RLHF Objective (KL-Regularized)")
  print("=" * 60)
  print()

  print("The RLHF objective is:")
  print()
  print("  max_pi  E_{x~D, y~pi}[r(x,y)] - beta * KL(pi || pi_ref)")
  print()
  print("Where:")
  print("  - pi is the policy we're training")
  print("  - pi_ref is the reference policy (usually SFT)")
  print("  - r(x,y) is the reward model")
  print("  - beta controls how close we stay to reference")
  print()
  print("Standard approach (RLHF):")
  print("  1. Train reward model r(x,y) on preference data")
  print("  2. Use PPO to maximize the objective")
  print("  3. Multiple rounds of refinement")
  print()
  print("DPO's insight: We can solve for the OPTIMAL pi analytically!")
  print("  - No explicit reward model")
  print("  - No PPO")
  print("  - Direct supervised learning on preference pairs")
```

## Step 1: The Optimal Policy Has Closed Form[#](#step-1-the-optimal-policy-has-closed-form)

Now we start the derivation. This is the mathematical heart of DPO.

optimal\_policy\_derivation.pycpu-only

```
import numpy as np

def derive_optimal_policy():
  """
  Derive the closed-form optimal policy.

  Given a KL-regularized reward objective, what policy maximizes it?
  The answer is a surprisingly simple formula.
  """
  print("Deriving the Optimal Policy: Full Derivation")
  print("=" * 70)
  print()

  print("OBJECTIVE (for a single prompt x):")
  print("-" * 70)
  print()
  print("  J(pi) = Sum_y pi(y|x) * r(x,y) - beta * KL(pi || pi_ref)")
  print()
  print("  where KL(pi || pi_ref) = Sum_y pi(y|x) * log(pi(y|x) / pi_ref(y|x))")
  print()
  print("Expanding:")
  print("  J(pi) = Sum_y pi(y|x) * r(x,y)")
  print("        - beta * Sum_y pi(y|x) * log(pi(y|x) / pi_ref(y|x))")
  print()
  print("  J(pi) = Sum_y pi(y|x) * [r(x,y) - beta * log(pi(y|x) / pi_ref(y|x))]")
  print()

  print("OPTIMIZATION (variational approach):")
  print("-" * 70)
  print()
```

Key Difference

Optimal Policy Has Closed Form

Reward Cancels in Preferences

DPO Pipeline

Preference Data

Direct Policy Optimization

Aligned Policy

RLHF Pipeline

Preference Data

1. Train Reward Model

2. PPO Optimization

Aligned Policy

## Step 2: The Implicit Reward Model[#](#step-2-the-implicit-reward-model)

Before we derive the loss, we need to understand what reward function would produce our optimal policy.

implicit\_reward\_model.pycpu-only

```
import numpy as np

def derive_implicit_reward():
  """
  What reward function is IMPLIED by the optimal policy?

  This is the key insight: we don't need to train a reward model
  because we can infer what it would be from the policy itself.
  """
  print("The Implicit Reward Model")
  print("=" * 70)
  print()

  print("Given that pi*(y|x) is the optimal policy under our objective:")
  print()
  print("  pi*(y|x) = pi_ref(y|x) * exp(r(x,y)/beta) / Z(x)")
  print()

  print("We can INVERT this to find what reward function r would produce it:")
  print()
  print("Starting from:")
  print("  pi*(y|x) = pi_ref(y|x) * exp(r(x,y)/beta) / Z(x)")
  print()

  print("Divide both sides by pi_ref(y|x):")
  print("  pi*(y|x) / pi_ref(y|x) = exp(r(x,y)/beta) / Z(x)")
  print()

  print("Take natural log of both sides:")
  print("  log(pi*(y|x) / pi_ref(y|x)) = r(x,y)/beta - log Z(x)")
```

## Step 3: Bradley-Terry Substitution → DPO Loss[#](#step-3-bradley-terry-substitution-dpo-loss)

Now we plug the implicit reward into the Bradley-Terry preference model and derive the final loss.

bradley\_terry\_substitution.pycpu-only

```
import numpy as np

def derive_dpo_loss():
  """
  Derive the DPO loss by substituting into Bradley-Terry.

  This is where the reward model cancels out.
  """
  print("The DPO Loss Derivation")
  print("=" * 70)
  print()

  print("BRADLEY-TERRY PREFERENCE MODEL:")
  print("-" * 70)
  print()
  print("The probability that y_w is preferred to y_l, given prompt x:")
  print()
  print("  P(y_w ≻ y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))")
  print()
  print("where r(x,y) is the reward and sigmoid(z) = 1 / (1 + exp(-z))")
  print()

  print("IMPLICIT REWARD SUBSTITUTION:")
  print("-" * 70)
  print()
  print("Substitute the implicit reward we derived:")
  print()
  print("  r(x,y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log Z(x)")
  print()
  print("into the preference probability:")
```

## Visual: The Partition Function Cancels[#](#visual-the-partition-function-cancels)

Bradley-Terry Preference

P(y\_w ≻ y\_l | x) = sigmoid(r\_w - r\_l)

r\_w = beta*log(π\_w/π\_ref\_w) + beta*log Z(x)

r\_l = beta*log(π\_l/π\_ref\_l) + beta*log Z(x)

Subtract

r\_w - r\_l = beta*log(π\_w/π\_ref\_w) - beta*log(π\_l/π\_ref\_l)

+ beta\*log Z(x) - beta\*log Z(x)

Z(x) terms cancel!

sigmoid(beta*log(π\_w/π\_ref\_w) - beta*log(π\_l/π\_ref\_l)

DPO Loss: Only log-probs needed

## Implementing the DPO Loss[#](#implementing-the-dpo-loss)

dpo\_loss\_implementation.pycpu-only

```
import numpy as np

def dpo_loss(
  pi_logprobs_w: np.ndarray,      # Policy log P(y_w | x)
  pi_logprobs_l: np.ndarray,      # Policy log P(y_l | x)
  ref_logprobs_w: np.ndarray,     # Reference log P(y_w | x)
  ref_logprobs_l: np.ndarray,     # Reference log P(y_l | x)
  beta: float = 0.1
) -> dict:
  """
  Compute DPO loss from the derived formula.

  The formula we derived is:
    L_DPO = -E[log sigmoid(beta * (log(π_w/π_ref_w) - log(π_l/π_ref_l)))]

  Args:
      pi_logprobs_w: Log-probabilities of preferred response under policy
      pi_logprobs_l: Log-probabilities of dispreferred response under policy
      ref_logprobs_w: Log-probabilities of preferred response under reference
      ref_logprobs_l: Log-probabilities of dispreferred response under reference
      beta: Temperature/KL strength parameter. Controls preference strength.
            High beta (0.5+): strong KL constraint, policy stays close to reference
            Low beta (0.01): weak KL constraint, policy can drift further from reference

  Returns:
      Dictionary with loss and diagnostic metrics
  """
  # Compute log-probability ratios (policy relative to reference)
  log_ratio_w = pi_logprobs_w - ref_logprobs_w
  log_ratio_l = pi_logprobs_l - ref_logprobs_l
```

## The Complete DPO Training Algorithm[#](#the-complete-dpo-training-algorithm)

dpo\_training\_algorithm.pycpu-only

```
def dpo_training_algorithm():
  """
  Complete DPO training loop with detailed comments.
  """
  algorithm = """
================================================================================
                       DPO TRAINING ALGORITHM
================================================================================

INPUT:
- reference_model: frozen SFT policy (e.g., 7B Llama)
- preference_data: tuples of (prompt, preferred_response, dispreferred_response)
- beta: KL regularization strength (typical: 0.1 to 0.5)

SETUP:
reference_model.freeze()  # Don't update reference
policy = copy(reference_model)  # Initialize policy from SFT
optimizer = AdamW(policy.parameters(), lr=1e-6)

================================================================================
                         TRAINING LOOP
================================================================================

for step, batch in enumerate(preference_dataloader):
  # batch.prompt: shape [B] (B prompts)
  # batch.preferred: shape [B, T_w] (B preferred responses of various lengths)
  # batch.dispreferred: shape [B, T_l] (B dispreferred responses)

  # STEP 1: Forward pass on POLICY (compute gradients)
  # -------------------------------------------------------
```

## Why Does This Work? Understanding the Mechanism[#](#why-does-this-work-understanding-the-mechanism)

dpo\_intuition.pycpu-only

```
import numpy as np

def explain_dpo_intuition():
  """
  Build intuition for why DPO works despite not training an explicit reward model.
  """
  print("Why DPO Works: Three Key Insights")
  print("=" * 70)
  print()

  print("INSIGHT 1: The Log-Ratio IS the Implicit Reward")
  print("-" * 70)
  print()
  print("When a policy assigns higher probability to a response than")
  print("the reference policy, that's mathematically equivalent to")
  print("assigning it higher reward.")
  print()
  print("From our derivation:")
  print("  r(x,y) = beta * log(π(y|x) / π_ref(y|x)) + [constant w.r.t. y]")
  print()
  print("So:")
  print("- If π(y|x) > π_ref(y|x)  →  reward increases")
  print("- If π(y|x) < π_ref(y|x)  →  reward decreases")
  print("- Log-ratio magnitude = strength of preference")
  print()
  print("We never explicitly compute 'reward', but it's implicit")
  print("in the policy's probability distributions!")
  print()

  print("INSIGHT 2: Preferences Only Encode Relative, Not Absolute Rewards")
```

## Simulating DPO Training on a Toy Problem[#](#simulating-dpo-training-on-a-toy-problem)

dpo\_simulation.pycpu-only

```
import numpy as np

def simulate_dpo_training():
  """
  Simulate DPO training end-to-end on a simple problem.

  Problem: 3 possible responses to a prompt.
  - Response 0: okay (quality = 0.2)
  - Response 1: best (quality = 0.7)
  - Response 2: worst (quality = 0.1)

  Goal: Train policy to match these quality rankings.
  """
  np.random.seed(42)

  print("DPO Training Simulation: From Scratch")
  print("=" * 70)
  print()

  # Ground truth quality ranking
  true_quality = np.array([0.2, 0.7, 0.1])

  # Reference model: uniform distribution (no preference)
  ref_logits = np.array([0.0, 0.0, 0.0])
  ref_probs = np.exp(ref_logits) / np.sum(np.exp(ref_logits))

  # Policy starts as reference (no training yet)
  policy_logits = ref_logits.copy()

  beta = 0.5
```

## Break It: The Effect of Beta (KL Strength)[#](#break-it-the-effect-of-beta-kl-strength)

break\_it\_beta.pycpu-only

```
import numpy as np
import matplotlib.pyplot as plt

def break_it_wrong_beta():
  """
  Beta is the temperature parameter controlling KL strength.
  Show how different values affect training dynamics.
  """
  np.random.seed(42)

  print("Break It: Sensitivity to Beta Parameter")
  print("=" * 70)
  print()

  # Simulated log-probabilities from a trained policy
  batch_size = 100
  ref_logprobs_w = np.random.randn(batch_size) * 5 - 100
  ref_logprobs_l = ref_logprobs_w + np.random.randn(batch_size) * 2

  # After training: policy assigns higher prob to preferred response
  pi_logprobs_w = ref_logprobs_w + 1.5
  pi_logprobs_l = ref_logprobs_l - 0.5

  print("Setup:")
  print("  Batch size: %d" % batch_size)
  print("  Policy advantage over reference: ~2 nats (strong but not extreme)")
  print()
  print("Question: What beta value gives the best training dynamics?")
  print()
```

## Break It: Preference Label Noise[#](#break-it-preference-label-noise)

break\_it\_label\_noise.pycpu-only

```
import numpy as np

def break_it_label_noise():
  """
  DPO directly trains on preference labels. What happens if labels are noisy?

  This is a critical issue because:
  1. Human preferences can be ambiguous or inconsistent
  2. Labelers can make mistakes
  3. Some pairs are genuinely hard to judge
  """
  np.random.seed(42)

  print("Break It: Preference Label Noise")
  print("=" * 70)
  print()

  def train_dpo_with_noise(noise_rate, steps=60):
      """Simulate DPO training with percentage of inverted labels."""
      # True quality ranking
      true_quality = np.array([0.2, 0.8])  # Response 1 is objectively better

      # Reference: uniform
      ref_probs = np.array([0.5, 0.5])

      # Policy: starts uniform
      policy_logits = np.array([0.0, 0.0])

      beta = 0.5
      lr = 0.05
```

## Break It: Incoherent Implicit Rewards[#](#break-it-incoherent-implicit-rewards)

▶What's the Problem with Implicit Rewards?

In RLHF, the reward model is explicitly trained to assign consistent scores. In DPO, the reward is implicit — it emerges from how the policy's log-probabilities change.

This can lead to **incoherent rewards**: different parts of the policy may have conflicting implicit reward signals.

**Example of Incoherent Reward:**

Imagine three responses A, B, C on the same prompt.

* Preference data says: A > B and B > C (implies A > C transitively)
* But the policy structure makes it easier to prefer C > A

DPO will try to satisfy A > B and B > C, but the implicit reward function r(x,y) computed from log-ratios might not satisfy r\_A > r\_C. The policy can satisfy pairwise preferences while having an internally inconsistent reward model.

This is **hard to debug** because:

1. Loss looks good (preferences are satisfied)
2. But the implicit reward function has cycles or inconsistencies
3. The policy may generalize poorly to new examples

RLHF is more robust here because the explicit reward model is trained to be transitive and globally consistent.

break\_it\_incoherent\_rewards.pycpu-only

```
import numpy as np

def demonstrate_incoherent_rewards():
  """
  Show how DPO can satisfy local preferences while having
  globally incoherent implicit rewards.
  """
  print("Break It: Incoherent Implicit Rewards")
  print("=" * 70)
  print()

  print("Scenario: 3 responses (A, B, C) to same prompt")
  print("-" * 70)
  print()

  # Simulated scenario
  np.random.seed(42)

  # Let's say the implicit reward (log-ratio) ends up being:
  # r_A = 1.0, r_B = 0.5, r_C = 0.8
  # This is incoherent: A > C > B (not transitive with preferences)

  implicit_rewards = {"A": 1.0, "B": 0.5, "C": 0.8}

  print("Preferences learned from data:")
  print("  A ≻ B (satisfied: r_A=1.0 > r_B=0.5) ✓")
  print("  B ≻ C (satisfied: r_B=0.5 < r_C=0.8) ✗ VIOLATED")
  print("  A ≻ C (satisfied: r_A=1.0 > r_C=0.8) ✓")
  print()
```

rlhf\_vs\_dpo\_comparison.pycpu-only

```
def compare_rlhf_dpo():
  """
  Comprehensive comparison of RLHF and DPO.
  """
  print("RLHF vs DPO: Comprehensive Comparison")
  print("=" * 85)
  print()

  comparison = """
DIMENSION                    RLHF                          DPO
───────────────────────────────────────────────────────────────────────────────
MODELS NEEDED                3 (ref, policy, RM)           2 (ref, policy)
TRAINING STAGES              2 (RM training, then PPO)      1 (direct training)
HYPERPARAMETERS              Many (PPO + KL + RM)          Few (beta, lr)
MEMORY USAGE (70B)           ~2.5-3 TB                     ~1.3-1.5 TB
COMPUTE PER STEP             ~3x (three forward passes)     ~2x (two forward)
STABILITY                    Requires careful tuning        Very stable
REWARD HACKING RISK          High (explicit RM)            Lower (implicit)
GENERALIZATION               Better (coherent reward)      May overfit pairs
IMPLEMENTATION               Very complex (500-1000 LOC)   Very simple (50-100)
TRAINING TIME                Longer (2-3x baseline)        Baseline (1x)
CONVERGENCE SPEED            Slower (RL is hard)           Faster (supervised)
PREFERENCE NOISE TOLERANCE   Moderate (learns from trends) Low (direct labels)
REWARD COHERENCE             Good (explicit model)         Potential cycles
───────────────────────────────────────────────────────────────────────────────
  """
  print(comparison)

  print()
  print("=" * 85)
```

## Scale Thought Experiment[#](#scale-thought-experiment)

| Aspect | RLHF | DPO |
| --- | --- | --- |
| **Memory (70B model)** | ~2.5-3 TB (4 models + optimizer states) | ~1.3-1.5 TB (2 models + optimizer states) |
| **Training time** | 2-3x longer | Baseline |
| **Hyperparameter sensitivity** | Very high | Moderate |
| **Reward hacking risk** | High | Lower (no explicit RM) |
| **Sample efficiency** | Lower (needs RL exploration) | Higher (supervised) |
| **Scalability** | Harder | Easier |

## Production Reality[#](#production-reality)

**Anthropic's approach:**

* Uses both RLHF and preference-based methods
* Constitutional AI provides additional training signal
* Multiple rounds of refinement

**Llama 2:**

* Used RLHF with rejection sampling
* PPO for final alignment
* Extensive hyperparameter tuning

**Mistral/Zephyr:**

* Used DPO for alignment
* Simpler pipeline, competitive results
* Showed DPO can match RLHF quality

**When to use DPO:**

* You have good preference data
* You want simpler training
* You're resource-constrained
* You need faster iteration

**When RLHF might be better:**

* You need very fine-grained control
* You have a well-tuned reward model
* You're doing extensive safety filtering

## Making the Derivation Operational[#](#making-the-derivation-operational)

If you can explain the derivation but cannot run DPO safely, you are missing the engineering layer:

1. **Treat DPO as supervised learning with a reference model.**
   Your core loop is stable, but the choice of `beta`, data filtering, and response lengths still controls outcomes.
2. **Design preference data as an optimization target.**
   DPO will faithfully amplify patterns in preferences, including spurious correlations and formatting artifacts.
3. **Validate with behavior slices, not only aggregate metrics.**
   Check refusal quality, helpfulness, hallucination rate, and verbosity. These are the usual regressions.
4. **Use beta as a control knob, not a magic constant.**
   If the model drifts too far, increase `beta`. If it does not learn, decrease `beta` or improve the preference signal.

## Checkpoint Questions[#](#checkpoint-questions)

1. Compute the DPO loss for a single preference pair where the policy assigns log-prob -95 to the preferred response and -100 to the dispreferred response, the reference assigns -98 and -99 respectively, and beta=0.1. Is the logit positive or negative? What does this imply about whether the policy has learned the preference?
2. A DPO run shows 98% training accuracy but poor held-out generalization. The mean logit magnitude is 15.0. Diagnose the likely cause by reasoning about what happens to sigmoid gradients at large logit values, and propose a fix.
3. You have 50K preference pairs with approximately 15% label noise (annotator disagreements). Estimate whether DPO or RLHF will degrade more from this noise level, and explain the mechanism.

## Research Hooks & Open Problems[#](#research-hooks-open-problems)

DPO is elegant, but the elegance masks some deep questions:

### 1. Is the KL-Regularized Objective the Right Target?[#](#1-is-the-kl-regularized-objective-the-right-target)

DPO assumes we want to maximize:

```
max_π E_x,y~π[r(x,y)] - β * KL(π || π_ref)
```

But why? Several assumptions are baked in:

* **Is KL divergence the right regularizer?** Other divergences (JS, Wasserstein, etc.) might better capture our intent. KL is convenient mathematically but not obviously optimal.
* **Does the implicit reward match human preferences?** We derive that π∗(y|x) ∝ π\_ref(y|x) \* exp(r/β). But the true human reward might not have this form.
* **What about value of information?** The formulation doesn't account for how much human feedback we've received. Early preferences should be weighted differently.

**Research direction:** Can we characterize when the KL-regularized objective provably captures human preferences? What properties of preference data make it valid?

### 2. Incoherence: Can We Extract & Fix the Implicit Reward?[#](#2-incoherence-can-we-extract-fix-the-implicit-reward)

DPO's implicit reward r(x,y) emerges from policy log-probs but may violate transitivity.

**Open questions:**

* Can we extract r(x,y) post-hoc and use it for further refinement?
* If we train a separate reward model to match the implicit reward, does that improve robustness?
* Can we add transitivity regularization losses to prevent cycles?
* How much does the incoherence actually hurt in practice? (Empirical question)

**Research direction:** Build tools to analyze and visualize the implicit reward landscape. Develop consistency metrics.

### 3. Why Is DPO So Sensitive to Label Noise?[#](#3-why-is-dpo-so-sensitive-to-label-noise)

Our breakit section showed that 20-30% label noise significantly degrades learning.

**Theories:**

* The loss gradients directly flow from labels → no error correction mechanism
* RLHF learns a reward model that can average out noisy labels
* DPO's "supervised learning" view doesn't account for label quality

**Research direction:** Develop noise-robust variants of DPO:

* Confidence-weighted DPO (weight examples by uncertainty)
* Triplet DPO (require transitivity across triples)
* Multi-rater DPO (aggregate preferences from multiple annotators)
* Contrastive DPO (learn what responses to avoid, not just prefer)

### 4. Scalability: How Does DPO Perform with 1000s of Preference Pairs?[#](#4-scalability-how-does-dpo-perform-with-1000s-of-preference-pairs)

Most DPO experiments use relatively small datasets (~10k examples).

**Open questions:**

* At what scale do we see incoherence issues multiply?
* Does DPO overfit more than RLHF as dataset size grows?
* How does batch size affect the quality of implicit rewards?
* Can we use curriculum learning (easy preferences first)?

**Research direction:** Large-scale DPO experiments. Compare final model quality vs sample efficiency.

### 5. Connecting DPO to Inverse Reinforcement Learning[#](#5-connecting-dpo-to-inverse-reinforcement-learning)

DPO implicitly inverts from policies to rewards. This is related to IRL.

**Interesting observation:**

* RLHF: Learn reward r → Maximize r (forward RL)
* DPO: Invert preferences to implicit r → Policy has implicit r built-in (inverse RL)

**Research direction:** Apply IRL techniques to improve DPO:

* Use maximum entropy IRL to learn more robust reward functions
* Combine DPO with IRL-inspired loss terms
* Study identifiability: is the implicit reward unique?

### 6. Beyond Bradley-Terry: Are There Better Preference Models?[#](#6-beyond-bradley-terry-are-there-better-preference-models)

DPO uses Bradley-Terry (sigmoid of reward difference). Are there better models?

**Alternatives:**

* **Thurstone model:** Assumes latent utilities with Gaussian noise
* **Plackett-Luce:** Ranking model that handles partial orders
* **Neural preference models:** Learn the preference distribution directly (but more complex)
* **Contextual models:** Preferences depend on user history, context, etc.

**Research direction:** Derive DPO equivalents for these models. Compare robustness.

### 7. DPO for Multi-Agent Preferences[#](#7-dpo-for-multi-agent-preferences)

Human preferences are inconsistent. Different people prefer different things.

**Questions:**

* Can DPO learn from multi-rater data where raters disagree?
* Should we personalize (learn π\_user)?
* What's the theoretical limit on disagreement?

**Research direction:** Multi-task DPO. Learn both persona-specific and shared components.

### 8. Extracting Interpretable Rewards[#](#8-extracting-interpretable-rewards)

DPO hides the reward function. Can we make it explicit post-hoc?

**Approaches:**

1. **Distillation:** Train RM to predict implicit reward
2. **Attribution:** Analyze which tokens affect reward most
3. **Probing:** Train classifiers to decode reward from internals
4. **Extraction:** Directly recover r from log-prob ratios

**Research direction:** Build interpretability tools for implicit rewards. Understand what DPO actually learns.

### 9. Theoretical Guarantees[#](#9-theoretical-guarantees)

RLHF has some theoretical understanding (RL convergence). DPO is newer.

**Open questions:**

* Under what conditions does DPO converge?
* What is the approximation error between implicit and true reward?
* Can we bound how far the policy drifts from reference?
* Generalization bounds: how does sample size affect final policy quality?

**Research direction:** Derive convergence guarantees. Characterize the approximation error.

---

## Summary: The DPO Story[#](#summary-the-dpo-story)

DPO is powerful because it **collapses a complex pipeline into a single loss function**:

```
RLHF: {data} → {RM} → {RM loss} → {trained RM} → {PPO} → {policy}

DPO:  {data} → {DPO loss} → {policy}
```

The trick: the partition function cancels when computing preferences.

But this elegance has costs:

* Implicit reward may be incoherent
* Sensitive to label noise
* No explicit error correction
* Limited theoretical understanding

The future likely lies in **hybrid methods**: DPO's simplicity + RLHF's robustness + improved understanding of implicit rewards.

---

*Next: DPO in practice. We'll implement a full training loop and debug real failure modes.*