In this tutorial, you will train a reward model from pairwise preference data using the Bradley-Terry loss, measure its calibration with Expected Calibration Error (ECE), and diagnose overfitting by tracking the train/val loss gap. By the end, you will be able to decide whether a given reward model is safe to deploy in an RLHF loop or needs further regularization.

## Prerequisites: Pairwise Comparisons and Preference Data[#](#prerequisites-pairwise-comparisons-and-preference-data)

▶What is a pairwise preference?

A pairwise preference is a human judgment comparing two responses:

* Prompt: "What is 2+2?"
* Response A: "2+2=4"
* Response B: "2+2=5"
* Preference: A > B

Unlike ratings (which are absolute), comparisons are **relative** and have several advantages:

* Easier for humans to judge (which is better, A or B?)
* More information per annotation (transitive relationships)
* Natural foundation for ranking and sorting
* Less prone to scale calibration issues

In reward model training, your dataset is typically a list of tuples: `(prompt, response_better, response_worse)`.

▶Why not just use ratings directly?

Rating-based approaches (e.g., score each response 1-5) have well-known problems:

* **Scale drift**: Different annotators use the scale differently
* **Fewer constraints**: "I gave it a 3" is weak information
* **Rater unreliability**: Harder to achieve high inter-annotator agreement

Pairwise comparisons solve these:

* **Binary outcome**: Only two possible answers (A>B or B>A)
* **Transitive structure**: A>B and B>C implies A>C (can validate)
* **Higher agreement**: Humans agree more on relative comparisons

Trade-off: You need more pairwise annotations to cover the same space, but each one is higher quality.

▶From Track 3: Probability Theory Refresher

Key concepts you will need:

* **Sigmoid function**: `sigmoid(x) = 1 / (1 + e^(-x))` maps real numbers to (0,1)
* **Log-likelihood**: `log(p(y|x))` measures fit of model to data
* **Cross-entropy loss**: Average negative log-likelihood across samples
* **Calibration**: Model's predicted probabilities match empirical frequencies

If you're rusty on these, review Track 0 on numerical stability and loss functions before diving deep here.

## The Bradley-Terry Model: Mathematical Foundation[#](#the-bradley-terry-model-mathematical-foundation)

The Bradley-Terry model is the canonical framework for learning from pairwise comparisons. It's elegant and gives us a principled way to derive the loss function.

### The Intuition[#](#the-intuition)

Imagine each response has a latent "quality score" `r\_i`. The probability that response A is better than B is:

```
P(A > B) = sigmoid(r\_A - r\_B)
```

This makes intuitive sense:

* If `r\_A >> r\_B`, then `sigmoid(large positive) ≈ 1` (A is definitely better)
* If `r\_A ≈ r\_B`, then `sigmoid(0) = 0.5` (equally good)
* If `r_A` is much less than `r_B`, then `sigmoid(large negative) ≈ 0` (A is worse)

### Formal Derivation[#](#formal-derivation)

The probability that A is preferred to B:

`P(A > B | r_A, r_B) = exp(r_A) / (exp(r_A) + exp(r_B))`

This is equivalent to:

`P(A > B) = 1 / (1 + exp(r_B - r_A)) = σ(r_A - r_B)`

Given a dataset of comparisons `(a_i, b_i, y_i)` where `y_i = 1` if `a_i > b_i`:

`log p(y | r) = Σ_i [ y_i log σ(r_ai - r_bi) + (1-y_i) log(1 - σ(r_ai - r_bi)) ]`

This is **binary cross-entropy** on the difference scores!

### Why Bradley-Terry Matters[#](#why-bradley-terry-matters)

The Bradley-Terry model gives us:

1. **A principled loss function** (binary cross-entropy on score differences)
2. **Theoretical guarantees** (under noise assumptions, the learned scores are consistent)
3. **Intuitive interpretation** (higher score = better response)

Most modern reward model training implements Bradley-Terry (sometimes without saying so explicitly).

### Connection to Ranking[#](#connection-to-ranking)

Bradley-Terry naturally extends to ranking. If you have multiple responses to rank, the model should satisfy transitivity:

If `P(A > B) = 0.9` and `P(B > C) = 0.8`, what's `P(A > C)`?

With Bradley-Terry:

* `P(A > C) = sigmoid(r\_A - r\_C)`

So you get consistent orderings. In practice, if the RM satisfies Bradley-Terry, you can sort responses by score and get a reasonable ranking.

### When Bradley-Terry Breaks Down[#](#when-bradley-terry-breaks-down)

**Critical assumption:** Transitivity. That is, if A > B and B > C, then A > C.

But in practice, this can fail:

* Different annotators have different preferences (A > B for Alice, B > A for Bob)
* Cyclic preferences (A > B > C > A in some contexts)
* Context dependency (A > B for coding questions, B > A for creative writing)

When you see frequent intransitivities in your data:

1. **Check for label noise.** Are annotators making mistakes?
2. **Check for domain mixing.** Is your data homogeneous enough? Or should you train separate RMs?
3. **Consider rejection sampling.** Exclude ambiguous or inconsistent examples.

bradley\_terry\_validation.pycpu-only

```
import numpy as np

def validate_bradley_terry_assumption():
  """
  Check if preference data satisfies transitivity (Bradley-Terry assumption).
  """
  np.random.seed(42)

  print("Validating Bradley-Terry Assumptions")
  print("=" * 70)
  print()

  # Generate some responses with scores
  n_responses = 100
  true_scores = np.random.uniform(-2, 2, n_responses)

  # Generate pairwise comparisons according to Bradley-Terry
  comparisons = []
  for i in range(n_responses):
      for j in range(i + 1, min(i + 10, n_responses)):  # Compare each with ~10 others
          # Bradley-Terry: P(i > j) = sigmoid(r_i - r_j)
          prob = 1 / (1 + np.exp(true_scores[j] - true_scores[i]))
          preference = 1 if np.random.rand() < prob else 0
          comparisons.append((i, j, preference))

  print("Generated %d pairwise comparisons" % len(comparisons))
  print()

  # Check for transitivity violations
  # For each triple (i, j, k), check if i > j, j > k implies i > k
```

### Training with Implicit Differentiation (Optional Advanced)[#](#training-with-implicit-differentiation-optional-advanced)

In large-scale practice (OpenAI, Anthropic), RMs are not necessarily fine-tuned from scratch. Instead:

1. Start from a large language model (GPT-3, Claude, etc.)
2. Add a simple **reward head**: linear layer mapping `[CLS]` embedding to scalar
3. Train **only the reward head** (freeze the base LLM)
4. Or: low-rank fine-tuning (LoRA) of the base model

This is much cheaper than full fine-tuning and often generalizes better.

The loss is still Bradley-Terry:

`L = -log σ(r_A - r_B)` (if A preferred)

But now `r_i = reward_head(f_LLM(response_i))`.

This approach naturally inherits the generalization of the pre-trained LLM.

## Loss Functions: Pairwise vs Ranking[#](#loss-functions-pairwise-vs-ranking)

Different formulations of preference learning lead to different loss functions.

### Pairwise Loss (Bradley-Terry)[#](#pairwise-loss-bradley-terry)

For a single comparison (A > B):

`L_pairwise = -log σ(r_A - r_B) = log(1 + exp(r_B - r_A))`

This is implemented as BCE loss on the score difference. The reward model predicts `r\_A` and `r\_B`, and we minimize:

`loss = -[log σ(r_A - r_B)]` if A is preferred, else `-[log(1 - σ(r_A - r_B))]`

**Pros:** Simple, efficient, handles ties gracefully
**Cons:** Only uses binary preference, ignores magnitude of preference

### Ranking/Margin Loss[#](#rankingmargin-loss)

If you have comparisons on a spectrum (A >> B vs A > B vs A ≈ B):

`L_margin = max(0, margin - (r_A - r_B))`

Forces the score gap to exceed a minimum margin. Useful when you have fine-grained preferences.

**Pros:** Captures preference strength
**Cons:** Requires carefully calibrated margin hyperparameter

In this lesson, we focus on **pairwise loss** (most common in RLHF).

## Data Collection Strategies and Quality Control[#](#data-collection-strategies-and-quality-control)

Before training a reward model, you need data. How you collect it matters enormously. This is arguably the most important part of RLHF: **garbage data in → garbage preferences out → garbage policy**.

### The Cost of Preference Data[#](#the-cost-of-preference-data)

Collecting human preferences is expensive:

| Source | Cost per Comparison | Quality | Scale | Latency |
| --- | --- | --- | --- | --- |
| Expert human annotators | $1-5 | Very high (IAA > 0.85) | 1K-10K | 2-4 weeks |
| Crowd workers | $0.10-0.50 | Medium (IAA > 0.70) | 10K-100K | 1 week |
| AI judges (GPT-4) | $0.01-0.05 | Good (IAA > 0.75 vs humans) | 100K-1M | Hours |
| Self-play/synthesis | $0.001 | Risky (not validated) | Unlimited | Real-time |

**The dilemma:** Human labels are expensive. AI labels are cheap but potentially biased.

**Current best practice:** Hybrid approach.

* Small, high-quality human dataset (5K-10K comparisons) as gold standard
* Larger AI-generated dataset (50K-500K) for scale
* Validate AI labels against human labels to estimate label quality

▶Common data collection approaches

### 1. Human Preference Data (Gold Standard)[#](#1-human-preference-data-gold-standard)

* Pay humans to compare model outputs
* High quality but expensive ($0.50-2.00 per comparison)
* Used by: OpenAI (InstructGPT), Anthropic (Claude)
* Scale: Typically 10K-100K comparisons for training

### 2. AI-Generated Preferences[#](#2-ai-generated-preferences)

* Use a strong model (GPT-4, Claude) to judge outputs
* Much cheaper ($0.01-0.05 per comparison at scale)
* Risk: Biased toward the judging model's style
* Used by: Anthropic (Constitutional AI, recent models)
* Scale: Can scale to millions

### 3. Synthetic/Self-Play Data[#](#3-syntheticself-play-data)

* Have a model compare its own outputs at different points in training
* Cheap but noisy
* Used in some research (Rlang, Self-Play RLHF)
* Scale: Unlimited but quality degrades

### 4. Proxy Signals[#](#4-proxy-signals)

* Code correctness (does it pass unit tests?)
* Information retrieval metrics (BLEU, ROUGE for summarization)
* User feedback (thumbs up/down)
* Risk: Often does not correlate with actual human preference

▶Quality control: What makes good preference data?

**High inter-annotator agreement (IAA):** Multiple humans label the same prompt/pair. If they disagree, the example is ambiguous.

**Clear comparative advantage:** Avoid borderline cases. "A is obviously better than B" is good data. "A and B are similar" is noise.

**Balanced difficulty:** Mix easy wins (A >> B) and hard calls (A ≥ B) so the model learns to discriminate across difficulty levels.

**Diversity:** Examples should span different task types, domains, and failure modes.

**Checked for biases:** Ensure humans are not systematically biased toward:

* Longer responses
* Certain writing styles
* Popular topics
* Earlier-in-conversation examples

## The Overfitting Problem[#](#the-overfitting-problem)

overfitting\_simulation.pycpu-only

```
import numpy as np

def simulate_rm_training(
  num_epochs=50,
  train_size=1000,
  val_size=200,
  regularization=0.0,
  label_noise=0.0
):
  """
  Simulate reward model training with different settings.
  Demonstrates the classic train/val gap that signals overfitting.
  """
  np.random.seed(42)

  # Generate "true" features that determine quality
  # In reality, these are embedding differences between responses
  X_train = np.random.randn(train_size, 10)
  X_val = np.random.randn(val_size, 10)

  # True quality depends on first 3 features (ground truth)
  true_weights = np.array([1, 0.5, 0.3] + [0] * 7)

  y_train = (X_train @ true_weights > 0).astype(float)
  y_val = (X_val @ true_weights > 0).astype(float)

  # Add label noise (annotation disagreements)
  if label_noise > 0:
      flip_mask = np.random.rand(train_size) < label_noise
      y_train[flip_mask] = 1 - y_train[flip_mask]
```

## Calibration: What It Is and Why It Matters[#](#calibration-what-it-is-and-why-it-matters)

Sigmoid

Yes, predictions match reality

No, overconfident

Reward Model Scores  
(Continuous)

Predicted Probability  
P(A > B) ∈ 0,1

RLHF Policy  
Uses Probability  
for Updates

Does Policy  
Actually Prefer  
Higher Scores?

✓ Calibrated RM

✗ Miscalibrated RM  
Enables Hacking

Policy learns  
genuine preferences

Policy learns  
spurious correlations

**Calibration** means: when the reward model predicts 70% probability that A > B, it should be correct approximately 70% of the time.

A miscalibrated RM is one where predicted probabilities do not match empirical frequencies. For example:

* Model predicts 90% confidence, but only correct 60% of the time → **overconfident**
* Model predicts 50% confidence, but correct 80% of the time → **underconfident**

Both break RLHF. Overconfidence amplifies small preference gaps into large reward signals, enabling hacking.

### Expected Calibration Error (ECE)[#](#expected-calibration-error-ece)

ECE measures miscalibration quantitatively:

`ECE = (1/N) Σ_(i=1)^K |P_i - A_i| * n_i`

Where:

* `P\_i` = average predicted probability in bin i
* `A\_i` = fraction of correct predictions in bin i
* `n\_i` = number of samples in bin i
* Target: ECE < 0.05 (very well calibrated), < 0.1 (acceptable)

calibration\_explained.pycpu-only

```
import numpy as np

def calibration_analysis(predictions, labels, num_bins=10):
  """
  Analyze calibration of probability predictions.

  Perfect calibration: when model predicts 70% confidence,
  it should be correct 70% of the time.
  """
  # Bin predictions
  bins = np.linspace(0, 1, num_bins + 1)
  bin_indices = np.digitize(predictions, bins) - 1
  bin_indices = np.clip(bin_indices, 0, num_bins - 1)

  results = []

  for i in range(num_bins):
      mask = bin_indices == i
      if np.sum(mask) > 0:
          avg_pred = np.mean(predictions[mask])
          avg_actual = np.mean(labels[mask])
          count = np.sum(mask)
          results.append({
              "bin_center": (bins[i] + bins[i+1]) / 2,
              "avg_prediction": avg_pred,
              "actual_accuracy": avg_actual,
              "count": count,
              "calibration_error": abs(avg_pred - avg_actual)
          })
```

## Label Smoothing for Better Calibration[#](#label-smoothing-for-better-calibration)

Label smoothing is a simple but powerful technique to improve calibration. Instead of training on hard labels (0 or 1), train on soft labels.

**Why it works:** Hard labels force the model to make extreme predictions. Soft labels allow the model to express uncertainty, which naturally improves calibration.

**Implementation:** `y\_smooth = y \* (1 - α) + 0.5 \* α` where `α ∈ [0.05, 0.2]`

For a preference (A > B), instead of targeting 1.0, target something like 0.95 (α = 0.1).

label\_smoothing.pycpu-only

```
import numpy as np

def train_with_label_smoothing(
  X_train, y_train, X_val, y_val,
  smoothing=0.0, num_epochs=30
):
  """
  Train with label smoothing to improve calibration.

  Instead of hard labels (0 or 1), use soft labels:
  y_smooth = y * (1 - smoothing) + 0.5 * smoothing
  """
  w = np.random.randn(X_train.shape[1]) * 0.01

  # Apply label smoothing
  y_train_smooth = y_train * (1 - smoothing) + 0.5 * smoothing

  for epoch in range(num_epochs):
      # Forward
      logits = X_train @ w
      probs = 1 / (1 + np.exp(-logits))

      # Gradient with smoothed labels
      grad = X_train.T @ (probs - y_train_smooth) / len(y_train)
      w -= 0.1 * grad

  # Evaluate
  val_logits = X_val @ w
  val_probs = 1 / (1 + np.exp(-val_logits))
  val_acc = np.mean((val_probs > 0.5) == y_val)
```

## Goodhart's Law in Action: Reward Hacking[#](#goodharts-law-in-action-reward-hacking)

Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."

In RLHF, this manifests as **reward hacking**: the policy discovers that certain observable features correlate with high RM scores, and exploits them without actually improving the underlying quality.

**Classic examples:**

* Model learns that longer responses score higher → produces verbose, repetitive outputs
* Model learns that confident language scores higher → produces authoritative-sounding wrong answers
* Model learns that certain phrases trigger rewards → copy-pastes them everywhere

The root cause is **overfitting**: the RM learned spurious correlations instead of true quality signals.

goodharts\_law.pycpu-only

```
import numpy as np

def simulate_reward_hacking():
  """
  Demonstrate Goodhart's Law: when a measure becomes a target,
  it ceases to be a good measure.
  """
  print("Goodhart's Law: Reward Hacking Simulation")
  print("=" * 60)
  print()

  # True quality depends on: correctness, helpfulness, safety
  # RM only learns to detect: length, formatting, certain phrases

  # Generate responses
  np.random.seed(42)
  n_responses = 100

  responses = []
  for i in range(n_responses):
      response = {
          # True quality factors (what we actually want)
          "correctness": np.random.uniform(0, 1),
          "helpfulness": np.random.uniform(0, 1),
          "safety": np.random.uniform(0, 1),

          # Observable features (what RM can learn)
          "length": np.random.randint(50, 500),
          "has_bullet_points": np.random.choice([True, False]),
          "uses_confident_language": np.random.choice([True, False]),
```

## Training Protocols to Minimize Overfitting[#](#training-protocols-to-minimize-overfitting)

No

Yes

RM Training  
Checkpoint

Pairwise Loss  
BCE on Score Diff

Regularization  
Weight Decay, Dropout

Early Stopping  
Monitor Val Loss

Calibration  
Compute ECE

Evaluate  
on Test Set

Acceptable?

Adjust Hyperparams  
or Collect More Data

Deploy to RLHF

The key is to structure training as a **validation-driven** process. You're not optimizing for training accuracy; you're optimizing for validation calibration and agreement.

training\_protocols.pycpu-only

```
import numpy as np

def rm_training_protocol():
  """
  Best practices for reward model training.
  """
  protocol = """
REWARD MODEL TRAINING PROTOCOL
==============================

1. DATA PREPARATION
 - Hold out 10-20% for validation
 - Ensure no prompt overlap between train/val
 - Check for label noise (inter-annotator agreement)
 - Balance easy vs hard comparisons

2. INITIALIZATION
 - Start from SFT model checkpoint
 - Same tokenizer and architecture as policy
 - Random init for reward head only

3. HYPERPARAMETERS
 - Learning rate: 1e-5 to 5e-6 (lower than SFT)
 - Batch size: 64-256
 - Epochs: 1-3 (watch for overfitting!)
 - Weight decay: 0.01-0.1
 - Label smoothing: 0.1-0.2

4. REGULARIZATION
 - Dropout in reward head: 0.1-0.3
```

## Break It: Overfitting a Reward Model[#](#break-it-overfitting-a-reward-model)

What happens when you train a reward model with zero regularization and no validation monitoring?

break\_it\_overfit.pycpu-only

```
import numpy as np

def demonstrate_rm_exploitation():
  """
  Show how a policy can exploit an overfit reward model.
  """
  print("Breaking an Overfit Reward Model")
  print("=" * 60)
  print()

  # Overfit RM: rewards specific phrases highly
  overfit_triggers = {
      "I'm delighted to help": 2.0,
      "Great question!": 1.5,
      "Here's a comprehensive answer": 1.8,
      "In conclusion": 1.0,
  }

  def overfit_rm(response):
      score = len(response) * 0.001  # Length bias
      for phrase, bonus in overfit_triggers.items():
          if phrase.lower() in response.lower():
              score += bonus
      return score

  # Policy learns to game the RM
  gaming_responses = [
      "Great question! I'm delighted to help. Here's a comprehensive answer. In conclusion, the answer is yes. " * 5,
      "The answer is: 42",
  ]
```

## Detecting Reward Hacking: Variance as a Signal[#](#detecting-reward-hacking-variance-as-a-signal)

A simple but effective heuristic: **when the policy's outputs disagree strongly across multiple reward models, reward hacking may be happening.**

If you have an ensemble of RMs (trained on slightly different data, or different architectures), high variance in their predictions is a red flag. The policy might have found an adversarial example.

Low

High

Policy Outputs

Score with  
RM Ensemble

Compute Mean  
and Variance

Variance  
High?

✓ Consensus  
Genuine Preference

⚠ Disagreement  
Possible Gaming

Accept Output

Flag for Review  
or Penalize Variance

Ensemble Reward Models

ensemble\_rm.pycpu-only

```
import numpy as np

def ensemble_reward_model():
  """
  Using multiple reward models to reduce variance and catch hacking.
  """
  print("Ensemble Reward Models")
  print("=" * 60)
  print()

  np.random.seed(42)

  # Simulate 5 different RMs (each with different biases)
  def make_rm(bias_type):
      def rm(features):
          base = features["quality"]
          if bias_type == "length":
              return base + features["length"] * 0.002
          elif bias_type == "confident":
              return base + features["confident"] * 0.5
          elif bias_type == "formal":
              return base + features["formal"] * 0.3
          elif bias_type == "clean":
              return base + features["quality"] * 0.5  # Actually good!
          else:
              return base
      return rm

  rms = [
      make_rm("length"),
```

## Common Failure Modes and Diagnostics[#](#common-failure-modes-and-diagnostics)

Yes

No

Collapse

Normal

Stuck

Improving

Symptom:  
RLHF Not Improving

Diagnose

RM predicting  
randomly?

Underfitting:  
Model too weak  
or data too hard

Add model capacity  
or get easier data

Check reward  
distribution

Miscalibration:  
RM overconfident

Add label smoothing  
or regularization

Check policy  
loss curve

Reward hacking:  
RM found shortcuts

Use RM ensemble  
to detect variance

✓ Working as expected

### Failure Mode 1: Reward Collapse[#](#failure-mode-1-reward-collapse)

Symptom: All outputs get similar reward scores (e.g., everything between 0.48-0.52).

Cause: RM is underfitting or has vanished gradients.

Detection:

```
reward_std = np.std(model.scores(outputs))
if reward_std < 0.1:
    print("WARNING: Reward collapse detected!")
```

Fix:

* Increase model capacity (hidden dimensions, layers)
* Decrease learning rate (exploding gradients)
* Check data: are examples too similar?
* Ensure validation loss is actually decreasing

### Failure Mode 2: Runaway Overfitting[#](#failure-mode-2-runaway-overfitting)

Symptom: Training loss near 0, validation loss > 1.0

Cause: Model memorized training set patterns.

Fix:

* Early stopping (stop when val loss starts increasing)
* Increase regularization (weight decay, dropout)
* Reduce learning rate
* Collect more diverse data

### Failure Mode 3: Bias in Predictions[#](#failure-mode-3-bias-in-predictions)

Symptom: RM systematically prefers certain types (longer, formatted, confident) regardless of quality.

Cause: Preference data had the bias built in.

Detection: Audit by hand. Sample RM's highest/lowest scored outputs and ask: "Are these actually good/bad?"

Fix:

* Relabel problematic examples
* Rebalance preference data (ensure equal distribution of lengths, styles)
* Use ensemble and check if all RMs agree

### Failure Mode 4: High Variance Across Ensemble[#](#failure-mode-4-high-variance-across-ensemble)

Symptom: Different RMs strongly disagree on some outputs.

Cause: Policy found an adversarial example or data was heterogeneous.

Fix:

* Flag these outputs for human review
* Penalize variance in the RL objective: `reward = mean(RM\_scores) - λ \* var(RM\_scores)`
* Collect more data on this domain

failure\_mode\_detection.pycpu-only

```
import numpy as np

def detect_rm_failure_modes():
  """
  Demonstrate how to diagnose reward model failure modes.
  """
  np.random.seed(42)

  print("Reward Model Failure Mode Detection")
  print("=" * 70)
  print()

  # Simulate three RMs
  n_outputs = 1000

  # Scenario 1: Healthy RM
  healthy_scores = np.random.normal(0.5, 0.15, n_outputs)
  healthy_scores = np.clip(healthy_scores, 0.1, 0.9)

  # Scenario 2: Collapsed RM
  collapsed_scores = np.random.normal(0.5, 0.01, n_outputs)
  collapsed_scores = np.clip(collapsed_scores, 0, 1)

  # Scenario 3: Overfit RM (few extreme values)
  overfit_scores = np.random.choice([0.05, 0.95], n_outputs, p=[0.2, 0.8])
  overfit_scores = overfit_scores + np.random.normal(0, 0.02, n_outputs)

  scenarios = [
      ("Healthy RM", healthy_scores),
      ("Collapsed RM", collapsed_scores),
```

## Detecting Overfitting in Practice[#](#detecting-overfitting-in-practice)

The key metric is the **train/val divergence**: if validation loss starts increasing while training loss decreases, you're overfitting.

detect\_overfitting.pycpu-only

```
import numpy as np

def analyze_overfitting_trajectory():
  """
  Demonstrate how to detect overfitting during training.
  Track when validation loss starts diverging from training loss.
  """
  np.random.seed(42)

  # Simulate training curves
  num_epochs = 100

  # Ideal case: train/val both improve together
  ideal_train = 0.7 * np.exp(-np.arange(num_epochs) / 20) + 0.1
  ideal_val = ideal_train + 0.02 * np.random.randn(num_epochs) * 0.02

  # Overfit case: train keeps improving, val diverges
  overfit_train = 0.7 * np.exp(-np.arange(num_epochs) / 15) + 0.05
  overfit_val = 0.15 + 0.1 * np.exp(-np.arange(num_epochs) / 30)

  print("Reward Model Training Diagnostics")
  print("=" * 70)
  print()

  print("IDEAL TRAINING (Well-Regularized):")
  print("%-8s%-15s%-15s%-15s%s" % ("Epoch", "Train Loss", "Val Loss", "Gap", "Status"))
  print("-" * 70)

  for epoch in [0, 10, 25, 50, 99]:
      gap = ideal_val[epoch] - ideal_train[epoch]
```

## Scale Thought Experiment[#](#scale-thought-experiment)

| Challenge | Small Data | Large Data |
| --- | --- | --- |
| **Overfitting** | Very high risk | Moderate risk |
| **Calibration** | Hard to assess | Can measure reliably |
| **Hacking** | Easy (limited patterns) | Harder (more diverse) |
| **Solution** | Heavy regularization, ensemble | Standard training, validation |

## Break It: Misspecified Preferences[#](#break-it-misspecified-preferences)

What happens when your preference annotations are systematically biased?

break\_it\_bias.pycpu-only

```
import numpy as np

def demonstrate_preference_bias():
  """
  Show how systematic bias in preference data corrupts the RM.
  """
  np.random.seed(42)

  print("Preference Data Bias Analysis")
  print("=" * 70)
  print()

  # Simulate two types of responses
  n_samples = 1000

  # True quality (what we want)
  true_quality = np.random.uniform(0, 1, n_samples)

  # Correlate: longer responses, formatted with bullets
  length = np.random.randint(50, 500, n_samples)
  has_bullets = (length > 200).astype(float)

  # Annotator bias: humans prefer longer, formatted responses
  # (Even if they're not actually better)
  preference_prob_unbiased = true_quality
  preference_prob_biased = (
      0.3 * true_quality +
      0.4 * (length / 500) +  # Length preference
      0.3 * has_bullets          # Format preference
  )
```

## Production Reality[#](#production-reality)

No

Yes

No

Yes

Collect  
Preference Data

Quality Control  
IAA, Bias Audit

Data  
Acceptable?

Revise, Relabel,  
or Collect More

Train RM  
with Early Stopping

Validate RM  
on Test Set

RM  
Calibrated?

Adjust Hyperparams  
or Collect Better Data

Deploy to RLHF

**OpenAI's approach:**

* Multiple RMs for different aspects (helpfulness, harmlessness)
* Ensembling for robustness
* Regular retraining as policy improves
* Continuous monitoring for hacking signals

**Anthropic's approach:**

* Constitutional AI reduces reliance on RM accuracy
* AI-generated preferences supplement human data
* Iterative refinement based on discovered failure modes
* Focus on diversity and robustness over sheer data size

**Best practices across labs:**

* Start with human preferences (high quality, expensive)
* Supplement with AI preferences at scale (cheaper, requires careful validation)
* Use ensemble of models to catch gaming
* Monitor distribution of reward values in policy rollouts
* Retrain RM every 1-2 policy iterations

## End-to-End RM Training Pipeline[#](#end-to-end-rm-training-pipeline)

Here's a practical checklist for training a production reward model:

rm\_training\_checklist.pycpu-only

```
def print_rm_training_checklist():
  """
  Complete checklist for reward model training.
  """
  checklist = """
RM TRAINING CHECKLIST
=====================

PHASE 1: DATA PREPARATION
[ ] Collect preference data (human, AI, or hybrid)
[ ] Verify inter-annotator agreement > 0.70
[ ] Check for systematic biases (length, style, domain)
[ ] Ensure no prompt overlap between train/test
[ ] Balance easy vs hard comparisons (80/20 split)
[ ] Remove duplicates and near-duplicates
[ ] Split: 70% train, 15% val, 15% test

PHASE 2: INITIAL SETUP
[ ] Choose base model (SFT checkpoint recommended)
[ ] Freeze base model, add reward head
[ ] Initialize reward head to zero
[ ] Verify model can overfit on small batch
[ ] Choose optimizer (AdamW, lr=5e-6)

PHASE 3: HYPERPARAMETER SELECTION
[ ] Learning rate: try [1e-6, 5e-6, 1e-5, 5e-5]
[ ] Batch size: 64-256 (larger = more stable)
[ ] Label smoothing: 0.0-0.2 (helps calibration)
[ ] Weight decay: 0.01-0.1 (prevents overfitting)
[ ] Dropout: 0.1-0.3 in reward head
```

## The Full Picture: From Data to Policy[#](#the-full-picture-from-data-to-policy)

No

Yes

Human Evaluations  
5K-10K High-Quality

AI-Generated Labels  
50K-500K at Scale

Merge & Quality Check

Split Data  
Train/Val/Test

Train RM  
Bradley-Terry Loss

Early Stopping  
on Val Loss

Evaluate on Test  
Check Calibration

Acceptable  
Performance?

Debug:  
More data?  
Better hyperparams?  
Check for bias?

Train RM Ensemble  
For Robustness

Integrate into RLHF

Policy Training  
Uses RM Rewards

Monitor for  
Hacking Signals

Retrain RM  
if Needed

The key insight: **RM training is validation-driven, not train-loss-driven.**

You're not trying to fit the training data perfectly. You're trying to learn a generalizable preference model that will not be exploited by the policy. High training accuracy with low validation accuracy is a failure.

## Lessons Learned from Industry[#](#lessons-learned-from-industry)

### From OpenAI's InstructGPT[#](#from-openais-instructgpt)

* Small, high-quality human preference dataset is crucial
* Ensembling multiple RMs catches hacking better than single model
* RM retraining every 1-2 policy iterations improves results
* Monitoring reward variance during policy training is essential

### From Anthropic's Constitutional AI[#](#from-anthropics-constitutional-ai)

* Can supplement human preferences with AI feedback (cheaper, scale better)
* Need explicit validation that AI feedback correlates with human feedback
* Diversity in feedback source improves robustness
* Ensemble voting + majority rules is simple and effective

### Common Mistakes[#](#common-mistakes)

1. **Throwing too much data at it:** 10M preference pairs will not fix bad data quality
2. **Ignoring validation:** Only watching training loss leads to overfitting
3. **Single RM:** Bugs and biases in one RM propagate to policy
4. **No baseline:** Without knowing what "good" looks like, hard to debug
5. **Premature RLHF:** Running RLHF on an uncalibrated RM is running blind

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. A reward model assigns scores r\_A = 2.0 and r\_B = 0.5 to two responses. Compute the Bradley-Terry probability P(A > B) using the sigmoid function. Then compute the pairwise loss -log sigma(r\_A - r\_B).
2. You train an RM for 50 epochs. At epoch 50, train loss = 0.12, val loss = 0.38. Compute the train/val gap. Should you deploy this RM to RLHF? What is your first fix?
3. An RM predicts 90% confidence on 200 comparisons. Of those 200, it gets 140 correct (70% actual accuracy). Compute the calibration error for this bin. Is this RM overconfident or underconfident, and by how many percentage points?

## Research Hooks[#](#research-hooks)

**Reward model robustness:**
Can we make RMs robust to adversarial inputs? If the policy is optimizing against the RM, it is essentially an adversary. How do we train defenses? This is related to adversarial robustness in supervised learning.

**Detecting reward hacking:**
Can we automatically detect when a policy is exploiting RM weaknesses? Some approaches: variance across RM ensemble, out-of-distribution detection, or adversarial validation sets. Active learning for RLHF.

**Uncertainty-aware rewards:**
Instead of point estimates, predict reward distributions: `p(r | output)`. Use uncertainty to moderate policy updates. High uncertainty → lower trust → smaller gradient steps. This is Bayesian RL.

**Preference learning from human evaluations:**
How do we infer individual preferences from population-level behaviors? Can we learn personalized RMs? This touches on preference aggregation and social choice theory.

**Online preference learning:**
Can the RM improve during RLHF based on policy-generated outputs? Continuous feedback loop instead of offline training then deployment.

---

*Next up: Agreement rate is necessary but not sufficient for evaluating reward models. We will see how to properly evaluate RMs before using them in RLHF.*