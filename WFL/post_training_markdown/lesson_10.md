In this tutorial, you will derive the REINFORCE gradient estimator from the log-derivative trick, implement it on a toy problem, measure the variance of gradient estimates at different batch sizes, and observe how baseline subtraction reduces that variance. By the end, you will understand why vanilla REINFORCE is impractical for LLM training and why PPO adds trust-region clipping.

## Prerequisites: Probability & Calculus Refresher[#](#prerequisites-probability-calculus-refresher)

▶Softmax & log-probabilities

If we have logits z = [z\_1, z\_2, ..., z\_K], the softmax probability is:

π(a) = exp(z\_a) / Σ\_i exp(z\_i)

The log-probability is:

log π(a) = z\_a - log(Σ\_i exp(z\_i)) = z\_a - logsumexp(z)

Taking the derivative w.r.t. z\_i:

∂log π(a) / ∂z\_i = 1(i=a) - π(i)

This is important: the gradient of log softmax is easy and has closed form.

▶Chain rule for expectations

If X is a random variable and f(x) is a function:

E[f(X)] = ∫ p(x) f(x) dx

Taking the derivative w.r.t. a parameter θ (assuming support does not depend on θ):

∂/∂θ E[f(X)] = ∫ ∂/∂θ [p(x) f(x)] dx

= ∫ [∂p(x)/∂θ · f(x) + p(x) · ∂f(x)/∂θ] dx

The first term is key: we can pull the gradient of the probability out.

▶Why we cannot just backprop through sampling

In supervised learning, we compute loss = f(model(x)) and backprop:

∂loss / ∂weights ← chain rule through model(x)

But in RL, we do: y ~ sample(π\_θ(·|x)), then reward = R(y)

The sampling operation has no gradient. There's a discrete random variable in the middle. You cannot compute ∂(sampling) / ∂θ because sampling is not a differentiable operation.

That is the whole problem REINFORCE solves.

## The RL Objective: What Are We Optimizing?[#](#the-rl-objective-what-are-we-optimizing)

rl\_objective.pycpu-only

```
import numpy as np

def explain_rl_objective():
  """
  The fundamental RL objective for language model alignment.
  """
  print("The RLHF Objective")
  print("=" * 60)
  print()

  print("Goal: Find policy parameters θ that maximize expected reward")
  print()
  print("  J(θ) = E_{x~prompts, y~π_θ(·|x)} [R(x, y)]")
  print()
  print("Where:")
  print("  - x is a prompt from the training distribution")
  print("  - y is a response sampled from policy π_θ")
  print("  - R(x, y) is the reward model score")
  print()
  print("The challenge: How do we compute ∇_θ J(θ)?")
  print()
  print("The response y is SAMPLED from π_θ.")
  print("Sampling is not differentiable!")
  print()
  print("We cannot just backprop through: y = sample(π_θ(·|x))")
  print()
  print("-" * 60)
  print()
  print("Concrete example:")
  print("  Prompt: 'Why is the sky blue?'")
```

## The Log-Derivative Trick: The Mathematical Sleight of Hand[#](#the-log-derivative-trick-the-mathematical-sleight-of-hand)

The entire REINFORCE algorithm rests on a single mathematical identity. Let me derive it carefully.

log\_derivative\_trick.pycpu-only

```
import numpy as np

def derive_log_derivative_trick():
  """
  Derive the log-derivative trick step by step.
  The most important derivation in policy gradient methods.
  """
  print("The Log-Derivative Trick Derivation")
  print("=" * 60)
  print()

  steps = [
      ("1. Start with objective",
       "J(θ) = E_{y~π_θ}[R(y)] = ∫ π_θ(y) R(y) dy"),

      ("2. Take gradient w.r.t. θ",
       "∇_θ J(θ) = ∫ ∇_θ π_θ(y) R(y) dy"),
      ("   (R(y) does not depend on θ, only π_θ does)",
       ""),

      ("3. Apply the key identity",
       "∇_θ π_θ(y) = π_θ(y) · ∇_θ log π_θ(y)"),
      ("   (This comes from chain rule: if p = exp(log p),",
       "    then ∇p = p · ∇log p)"),

      ("4. Substitute identity into integral",
       "∇_θ J(θ) = ∫ π_θ(y) · ∇_θ log π_θ(y) · R(y) dy"),

      ("5. Recognize as expectation under π_θ",
       "∇_θ J(θ) = E_{y~π_θ}[R(y) · ∇_θ log π_θ(y)]"),
```

▶Deep dive: Why ∇π = π · ∇log π?

This is a pure calculus identity. Let us derive it step-by-step.

**From first principles:**

Let p(θ) be any positive function (e.g., a probability).

By definition: log p(θ) = ln(p(θ))

Taking the derivative of both sides w.r.t. θ:

d/dθ [log p(θ)] = 1/p(θ) · dp/dθ

Rearrange:

dp/dθ = p(θ) · d/dθ [log p(θ)]

That is it! The gradient of p is p times the gradient of log p.

**Intuition:** log is a monotonic transformation. It "scales down" the original gradient. So to recover the original gradient, we multiply back by the original value.

**Why does this help?** Now when we compute ∇\_θ J(θ), we get:

∇\_θ J(θ) = ∫ [π\_θ(y) · ∇log π\_θ(y)] · R(y) dy

We can move π\_θ(y) inside the expectation: now it is a probability weight, and we can sample!

## REINFORCE Implementation: The Algorithm in Code[#](#reinforce-implementation-the-algorithm-in-code)

reinforce\_implementation.pycpu-only

```
import numpy as np

class SimplePolicy:
  """
  A simple policy for demonstration.
  Action probabilities are softmax of linear weights.
  """

  def __init__(self, num_actions):
      self.num_actions = num_actions
      self.weights = np.zeros(num_actions)

  def get_probs(self):
      """Softmax probabilities."""
      exp_w = np.exp(self.weights - np.max(self.weights))
      return exp_w / np.sum(exp_w)

  def sample(self):
      """Sample an action."""
      probs = self.get_probs()
      return np.random.choice(self.num_actions, p=probs)

  def log_prob(self, action):
      """Log probability of an action."""
      probs = self.get_probs()
      return np.log(probs[action] + 1e-10)

  def grad_log_prob(self, action):
      """
      Gradient of log probability w.r.t. weights.
```

## The Variance Problem: Why REINFORCE Alone Fails[#](#the-variance-problem-why-reinforce-alone-fails)

variance\_problem.pycpu-only

```
import numpy as np

def demonstrate_variance_problem():
  """
  REINFORCE is unbiased but HIGH VARIANCE.
  This explains why it is impractical in real systems.
  """
  np.random.seed(42)

  print("Variance in REINFORCE Gradient Estimates")
  print("=" * 60)
  print()
  print("We want to estimate: ∇J = E[R(y) · ∇log π(y)]")
  print()
  print("For a fixed reward R, the variance scales as:")
  print("  Var[estimate] ≈ Var[R] · E[||∇log π||²] / N")
  print()
  print("-" * 60)
  print()

  # Simulate: vary batch size, see variance of gradient estimates
  true_expected_grad = 1.0  # What we want to estimate
  reward_std = 2.0  # Typical reward variance

  sample_sizes = [10, 50, 100, 500, 1000]

  for n in sample_sizes:
      estimates = []
      for trial in range(100):
          # Simulate: N samples of (reward * score_fn)
```

High Variance Gradient  
(REINFORCE with N=32)

Large Random Updates

Training Oscillates  
Around Optimum

Need Tiny Learning Rate

Training Converges  
Very Slowly

Impractical!

## Variance Reduction: Baselines Save the Day[#](#variance-reduction-baselines-save-the-day)

The key insight: we can subtract ANY constant baseline from rewards without biasing the gradient. This is the most important variance reduction technique in all of RL.

baseline\_subtraction.pycpu-only

```
import numpy as np

def derive_baseline():
  """
  Mathematically prove that baselines do not bias the gradient.
  """
  print("Baseline Subtraction: Math")
  print("=" * 60)
  print()

  print("Original REINFORCE:")
  print("  ∇J = E_{y~π}[R(y) · ∇log π(y)]")
  print()

  print("Modified REINFORCE with baseline b:")
  print("  ∇J = E_{y~π}[(R(y) - b) · ∇log π(y)]")
  print()

  print("Proof that baseline does not bias gradient:")
  print()
  print("  E[(R - b) · ∇log π] = E[R · ∇log π] - E[b · ∇log π]")
  print("                      = E[R · ∇log π] - b · E[∇log π]")
  print()
  print("  What is E[∇log π]?")
  print("    = ∫ ∇log π(y) · π(y) dy")
  print("    = ∫ ∇π(y) dy")
  print("    = ∇ ∫ π(y) dy")
  print("    = ∇ 1")
  print("    = 0  ← Key fact!")
  print()
```

## Application to LLMs: From Toy Problem to Real RLHF[#](#application-to-llms-from-toy-problem-to-real-rlhf)

llm\_policy\_gradient.pycpu-only

```
import numpy as np

def llm_policy_gradient_intuition():
  """
  How REINFORCE applies to language model training.
  This is the exact setup of real RLHF systems.
  """
  print("REINFORCE for Language Models")
  print("=" * 60)
  print()

  print("Policy Setup:")
  print("  - π_θ is the base language model")
  print("  - Action space = vocabulary (e.g., 50K tokens)")
  print("  - Trajectory = full response sequence")
  print("  - Reward = reward model score for that response")
  print()

  print("Autoregressive decomposition:")
  print("  Response: y = (y_1, y_2, ..., y_T)")
  print()
  print("  Probability:")
  print("    π_θ(y|x) = π_θ(y_1|x) · π_θ(y_2|x,y_1) · ... · π_θ(y_T|x,y_<T)")
  print()
  print("  Log-probability:")
  print("    log π_θ(y|x) = Σ_{t=1}^T log π_θ(y_t | x, y_{<t})")
  print()
  print("  Gradient:")
  print("    ∇log π_θ(y|x) = Σ_{t=1}^T ∇log π_θ(y_t | x, y_{<t})")
  print()
```

Sample prompt x

Generate y ~ π\_θ(·|x)  
via autoregressive sampling

Score with reward  
model: R(x,y)

Compute log π\_θ(y|x)  
= sum of token log-probs

Backprop to get  
∇log π\_θ(y|x)

Scale by reward:  
R(x,y) · ∇log π

Average over batch:  
(1/B) Σ gradients

Update policy:  
θ ← θ + α · ∇J

Repeat

## Break It: High Variance Training Failure Mode[#](#break-it-high-variance-training-failure-mode)

What happens when you use REINFORCE with very high variance? Training becomes unstable and oscillates wildly around the optimum.

break\_it\_variance.pycpu-only

```
import numpy as np

def demonstrate_unstable_training():
  """
  Show what happens with high-variance gradients in RL.
  This is a real problem in naive RLHF implementations.
  """
  np.random.seed(42)

  print("Training Instability from High Variance")
  print("=" * 60)
  print()

  # Simulate a single policy parameter
  theta = 0.0
  optimal_theta = 1.0

  # Compare: low variance vs high variance
  for variance_level, noise_std in [("LOW (variance-reduced)", 0.1),
                                     ("HIGH (no baselines)", 2.0)]:
      theta = 0.0
      theta_history = [theta]

      lr = 0.1

      for step in range(50):
          # True gradient points toward optimal
          true_grad = optimal_theta - theta

          # Noisy gradient estimate
```

## Scale Thought Experiment: How Batch Size Changes Everything[#](#scale-thought-experiment-how-batch-size-changes-everything)

| Batch Size | Variance Level | Practical Issues | Required Mitigations |
| --- | --- | --- | --- |
| **N = 32** | Extremely high | Training oscillates wildly | Baselines + small LR |
| **N = 128** | Very high | Unstable, erratic updates | Baselines + PPO clipping |
| **N = 512** | High | Slow convergence, noisy signal | Standard baselines + PPO |
| **N = 2048** | Moderate | Acceptable variance | Standard RLHF setup |
| **N = 8192+** | Low | Stable but expensive compute | Can use simpler algorithms |

In practice, RLHF labs typically use N = 256-512 (compute-constrained), which is why baselines and PPO are essential, not optional.

## Break It: Catastrophic Updates Without Trust Region[#](#break-it-catastrophic-updates-without-trust-region)

What happens if a single unlucky sample gives a very high reward? REINFORCE will make a huge gradient step.

break\_it\_catastrophic.pycpu-only

```
import numpy as np

def demonstrate_catastrophic_updates():
  """
  REINFORCE can take catastrophically large steps when a lucky
  high-reward sample gets sampled. This is a real failure mode.
  """
  np.random.seed(42)

  print("Catastrophic Updates in REINFORCE")
  print("=" * 60)
  print()

  print("Scenario: Policy trained on mostly bad responses")
  print("Then one LUCKY high-reward response gets sampled")
  print()

  # Typical rewards: mostly negative with rare high rewards
  typical_rewards = np.random.normal(-0.5, 0.3, 100)
  outlier_reward = 3.0

  print("Typical rewards: mean=%.2f, std=%.2f" % (np.mean(typical_rewards), np.std(typical_rewards)))
  print("Outlier reward: %s" % outlier_reward)
  print()

  # Gradient magnitude
  score_function_magnitude = 0.5
  typical_gradient = np.mean(typical_rewards) * score_function_magnitude
  outlier_gradient = outlier_reward * score_function_magnitude
```

## Production Reality: Why PPO Exists[#](#production-reality-why-ppo-exists)

REINFORCE is the theoretical foundation, but production RLHF requires practical modifications:

Too High  
Variance

Still Unstable  
with Outliers

Production  
Algorithm

REINFORCE

Add Baselines

Add Trust Region  
Clipping

PPO

PPO also adds:

Value Function  
Baseline

KL Divergence  
Penalty

**Typical RLHF pipeline (what labs actually use):**

1. Sample batch of prompts x\_1, ..., x\_B
2. Generate responses y\_1, ..., y\_B from current policy π\_θ
3. Score responses with reward model: R(x\_i, y\_i)
4. Compute advantages A\_i = R(x\_i, y\_i) - V(x\_i) (baseline subtraction!)
5. Update policy with PPO objective:
   * Compute log-probabilities and baselines
   * Clip probability ratios to prevent large updates
   * Update policy with clipped surrogate loss
6. Update value function V to match rewards
7. Repeat

The key insight: REINFORCE teaches the math, PPO teaches the practice.

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. Starting from J(theta) = E[R(y) \* grad\_log pi(y)], write the Monte Carlo estimate using N = 4 samples. If the rewards are [0.2, 0.8, -0.3, 0.5] and the score function magnitudes are all 1.0, compute the gradient estimate.
2. A REINFORCE run uses batch size N = 32 with reward mean = 10.0 and reward std = 2.0. Estimate the coefficient of variation (std/mean) of the gradient. Now add a baseline b = 10.0 (the mean reward). What is the new coefficient of variation? By what factor did variance decrease?
3. A single sample in your RLHF batch receives reward R = 15.0 while the batch mean is 2.0. Without trust-region clipping, the parameter update is proportional to (15.0 - 2.0) = 13.0. With PPO clipping at epsilon = 0.2, the effective ratio is clamped to 1.2. Estimate the maximum parameter update magnitude relative to the unclipped case.

## Research Hooks[#](#research-hooks)

**Better variance reduction:**
Can we do better than simple mean baselines? Control variates, action-dependent baselines, exponential weighting of trajectories—all active research areas.

**Off-policy corrections:**
Can we reuse old experience rather than requiring fresh samples? Importance sampling enables this, but introduces new variance. Recent work: offline RL, importance-weighted policy gradients.

**Alternative policy gradients:**
Actor-critic methods use learned baselines. Natural policy gradients use curvature information. Evolutionary strategies avoid gradients altogether. Each trades off different properties.

**Reward model instability:**
As the policy drifts from training data, reward model predictions become less reliable. How do we maintain alignment? This is an open research problem.

---

*Next up: PPO clips the objective to prevent catastrophically large updates. It's a practical approximation to trust region optimization that empirically just works.*