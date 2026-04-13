In this tutorial, you will write the SFT loss function, implement loss masking strategies, configure learning rate schedules for fine-tuning, and diagnose common SFT failures such as memorization, format collapse, and catastrophic forgetting.

## Prerequisites: Quick Refresh[#](#prerequisites-quick-refresh)

▶Cross-Entropy Loss

Cross-entropy loss measures how well a model predicts the next token. If the model assigns probability `p` to the correct token:

`loss = -log(p)`

* If `p = 0.9` (confident and correct): loss ≈ 0.1 (good)
* If `p = 0.1` (confident but wrong): loss ≈ 2.3 (bad)
* If `p = 0.5` (uncertain): loss ≈ 0.7 (okay)

This is the same loss used in pretraining. No new math in SFT -- only different data.

▶Gradient Flow in Attention

When we update model weights based on SFT loss, gradients flow backward through the attention mechanism. Key insight: if we mask out prompt tokens, gradients don't flow to those positions. This prevents the model from learning to generate prompts -- we only train on response generation.

▶Catastrophic Forgetting

When fine-tuning a pretrained model, there's a tradeoff: either (a) learn the new task well but forget pretraining knowledge, or (b) preserve pretraining but learn the new task poorly. This is "catastrophic forgetting."

SFT hyperparameters (low LR, short training) are designed to minimize this tradeoff.

## The SFT Loss Function[#](#the-sft-loss-function)

SFT uses the same cross-entropy loss as pretraining, but only on response tokens:

sft\_loss.pycpu-only

```
import numpy as np

def cross_entropy_loss(logits, targets, mask=None):
  """
  Cross-entropy loss, optionally masked.

  logits: (batch, seq, vocab) - model predictions
  targets: (batch, seq) - target token IDs
  mask: (batch, seq) - 1 for tokens to train on, 0 for tokens to ignore
  """
  # Softmax to get probabilities
  probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
  probs = probs / probs.sum(axis=-1, keepdims=True)

  # Get probability of correct token
  batch_size, seq_len = targets.shape
  correct_probs = np.zeros((batch_size, seq_len))
  for b in range(batch_size):
      for s in range(seq_len):
          correct_probs[b, s] = probs[b, s, targets[b, s]]

  # Negative log likelihood
  nll = -np.log(correct_probs + 1e-10)

  # Apply mask (if provided)
  if mask is not None:
      nll = nll * mask
      return nll.sum() / mask.sum()
  else:
      return nll.mean()
```

## Loss Masking Strategies[#](#loss-masking-strategies)

There are multiple ways to apply masking during SFT training. Each has tradeoffs:

Full Response

Partial Response

End-of-Turn Token

No Masking

SFT Data: prompt + response

Masking Strategy

Only compute loss on  
response tokens

Ignore first N tokens  
of response

Only loss on final  
EOS token

Train on all tokens  
prompt + response

Pro: Clean signal  
Con: Wastes prompt computation

Pro: Balance gradient  
Con: More hyperparameters

Pro: Minimal forgetting  
Con: Weak learning signal

Pro: Maximum data  
Con: Model learns prompt format

### Strategy 1: Full Response Masking (Most Common)[#](#strategy-1-full-response-masking-most-common)

Train loss on all response tokens, zero out prompt:

```
Prompt:   "What is 2+2?" → tokens [t1, t2, t3, t4]  (mask=0)
Response: "The answer is 4." → tokens [t5, t6, t7, t8]  (mask=1)

Loss = CrossEntropy(logits[t5:t8], targets[t5:t8])
```

**Best for:** General instruction-following, when response format is important.

### Strategy 2: Partial Response Masking[#](#strategy-2-partial-response-masking)

Include some prompt context in gradient computation to prevent prompt format drift:

```
Prompt:    [t1, t2, t3, t4]  (mask=0)
Buffer:    [t5, t6]          (mask=0.5)  ← soft transition
Response:  [t7, t8]          (mask=1)    ← full loss
```

**Best for:** Domain-specific SFT where prompt format must be preserved exactly.

### Strategy 3: Per-Token Masking (Advanced)[#](#strategy-3-per-token-masking-advanced)

Weight each response token's loss differently based on importance:

per\_token\_masking.pycpu-only

```
import numpy as np

def compute_token_importance_weights(response_tokens, response_text):
  """
  Example: weight tokens differently based on content.

  High weight: factual, core tokens
  Low weight: filler, repeated tokens
  """
  weights = np.ones(len(response_tokens))

  # Downweight common filler words
  filler_words = ["the", "a", "is", "are", "it", "that"]

  for i, token in enumerate(response_tokens):
      if token.lower() in filler_words:
          weights[i] = 0.5

  # Upweight tokens at response end (usually most important)
  final_tokens = max(1, len(response_tokens) // 4)
  weights[-final_tokens:] *= 1.5

  # Normalize
  weights = weights / weights.mean()

  return weights

# Example
response_tokens = ["The", "answer", "is", "4", ".", "This", "is", "correct."]
response_text = "The answer is 4. This is correct."
```

## Learning Rate Schedules[#](#learning-rate-schedules)

The classic approach: start lower than pretraining, then schedule down.

lr\_schedules.pycpu-only

```
import numpy as np
import matplotlib.pyplot as plt

def constant_lr(step, init_lr=1e-5):
  """No schedule, just constant LR."""
  return init_lr

def linear_decay(step, total_steps, init_lr=1e-5):
  """Linear decay: LR decreases linearly to zero."""
  progress = min(step / total_steps, 1.0)
  return init_lr * (1 - progress)

def cosine_decay(step, total_steps, init_lr=1e-5):
  """Cosine annealing: smooth decay following cosine curve."""
  progress = min(step / total_steps, 1.0)
  return init_lr * (1 + np.cos(np.pi * progress)) / 2

def cosine_with_warmup(step, total_steps, warmup_steps, init_lr=1e-5):
  """
  Warmup then cosine decay.
  Typical SFT schedule: warm up for 100-500 steps, then decay.
  """
  if step < warmup_steps:
      # Linear warmup
      return init_lr * (step / warmup_steps)
  else:
      # Cosine decay
      progress = (step - warmup_steps) / (total_steps - warmup_steps)
      progress = min(progress, 1.0)
      return init_lr * (1 + np.cos(np.pi * progress)) / 2
```

## Batch Size Considerations[#](#batch-size-considerations)

Batch size in SFT has different implications than in pretraining:

| Batch Size | Gradient Signal | Training Speed | Memory | Stability | When to Use |
| --- | --- | --- | --- | --- | --- |
| **32** | Noisy, but diverse | Slower | Low | High variance | Small datasets (< 5K examples) |
| **64** | Good balance | Medium | Medium | Recommended | 10K-50K examples |
| **128-256** | Smoother gradients | Fast | High | Stable | Large datasets (> 50K examples) |
| **512+** | Very smooth | Very fast | Very high | Less sensitive to hyperparams | Distributed training only |

batch\_size\_analysis.pycpu-only

```
import numpy as np

def simulate_sft_with_batch_size(batch_size, num_updates=100, learning_rate=1e-5):
  """
  Simulate how batch size affects gradient noise and convergence.

  Key insight: larger batches = less noisy gradients = smoother convergence.
  But SFT datasets are small, so we can't use very large batches.
  """
  loss = 2.0
  losses = []

  for step in range(num_updates):
      # Simulate gradient noise inversely proportional to sqrt(batch_size)
      gradient_magnitude = 1.0
      gradient_noise = 1.0 / np.sqrt(batch_size)

      # Noisy gradient step
      noisy_grad = gradient_magnitude + np.random.normal(0, gradient_noise)
      loss = loss - learning_rate * noisy_grad

      # Prevent negative loss
      loss = max(0.1, loss)
      losses.append(loss)

  return np.array(losses)

batch_sizes = [32, 64, 128, 256]
num_updates = 100
```

## SFT vs Pretraining[#](#sft-vs-pretraining)

SFT

Pretraining

Web Data  
Trillions of tokens  
All documents

Next Token Prediction  
Train on everything

Curated Data  
10K-100K examples  
prompt, response pairs

Next Token Prediction  
Train only on responses

Base Model  
Capable but misaligned

Instruct Model  
Aligned behavior

| Aspect | Pretraining | SFT |
| --- | --- | --- |
| **Data** | Trillions of tokens | 10K-100K examples |
| **Source** | Web, books, code | Curated (prompt, response) pairs |
| **Objective** | P(next token) | P(response token | prompt) |
| **Learning rate** | 1e-4 to 1e-3 | 1e-6 to 1e-5 (much lower!) |
| **Training time** | Weeks to months | Hours to days |

sft\_training\_loop.pycpu-only

```
import numpy as np

class SimpleSFTTrainer:
  """
  Simplified SFT training loop demonstration.
  (Not actual training - just showing the algorithm structure)
  """

  def __init__(self, model, learning_rate=1e-5):
      self.model = model
      self.lr = learning_rate

  def prepare_batch(self, examples):
      """
      Prepare a batch with proper masking.

      Each example: {"prompt": str, "response": str}
      """
      batch_data = {
          "input_ids": [],
          "labels": [],
          "attention_mask": [],
          "loss_mask": [],  # 1 for response tokens, 0 for prompt
      }

      for ex in examples:
          # Tokenize (simplified)
          prompt_tokens = self.tokenize(ex["prompt"])
          response_tokens = self.tokenize(ex["response"])
```

## Learning Rate Selection[#](#learning-rate-selection)

lr\_selection.pycpu-only

```
import numpy as np

def simulate_sft_with_lr(base_capability, lr, num_steps=1000):
  """
  Simulate how learning rate affects SFT.

  Too high LR: destroys base capability
  Too low LR: barely changes behavior
  Just right: adjusts behavior while preserving capability
  """
  capability = base_capability
  alignment = 0.0

  for step in range(num_steps):
      # Alignment improves with training
      alignment += lr * 1000 * (1 - alignment)

      # Capability degrades if LR is too high (catastrophic forgetting)
      if lr > 5e-5:
          capability *= (1 - lr * 100)  # Faster degradation
      elif lr > 1e-5:
          capability *= (1 - lr * 10)   # Moderate degradation
      else:
          capability *= (1 - lr * 1)    # Minimal degradation

      capability = max(0.1, capability)  # Floor

  return {
      "final_capability": capability,
      "final_alignment": alignment,
```

## Common SFT Failures: Break It[#](#common-sft-failures-break-it)

### 1. Format Collapse[#](#1-format-collapse)

Homogeneous SFT Data  
All responses use same format

Model learns strong format prior  
P format x | prompt

At inference, model applies  
format everywhere

Format collapse detected  
User: how to fix a bike?  
Model: 1. Step one  
2. Step two...

Diverse SFT Data  
Varied response formats

Model learns flexible format  
P format x | question type

At inference, model adapts  
format to question

format\_collapse.pycpu-only

```
# Format collapse: model learns ONE format and uses it everywhere

# Bad: All responses become lists
bad_responses = [
  "Here's information about X:\n1. Point one\n2. Point two\n3. Point three",
  "Here's information about Y:\n1. Point one\n2. Point two\n3. Point three",
  "Here's information about Z:\n1. Point one\n2. Point two\n3. Point three",
]

# Good: Format varies based on the question
good_responses = [
  "The capital of France is Paris.",  # Simple answer
  "Here are the steps:\n1. First...\n2. Then...\n3. Finally...",  # List for procedures
  "This is a complex topic. Let me break it down:\n\nFirst...",  # Paragraph for explanation
]

print("Format Collapse Detection")
print("=" * 60)

def detect_format_collapse(responses):
  """Check if responses have too-similar formatting."""
  patterns = []
  for r in responses:
      # Simple pattern detection
      has_list = "1." in r or "- " in r
      has_intro = r.startswith("Here's") or r.startswith("Here are")
      num_newlines = r.count("\n")

      patterns.append((has_list, has_intro, num_newlines))
```

### 2. Memorization and Overfitting[#](#2-memorization-and-overfitting)

Train for too many epochs  
High learning rate  
Small dataset

Model fits SFT data perfectly  
Loss goes to zero

But at inference...

OOD prompt: model fails  
No exposure to variations

Train for 1-2 epochs  
Low learning rate  
Rich dataset

Model generalizes  
Loss plateaus smoothly

At inference...

OOD prompt: model adapts  
Knowledge + SFT adjustment

memorization\_detection.pycpu-only

```
import numpy as np

def detect_memorization(train_responses, generated_responses, threshold=0.9):
  """
  Detect if model is memorizing training data.

  If generated responses are too similar to training data,
  the model may be overfitting rather than generalizing.
  """

  def similarity(a, b):
      """Simple character-level similarity."""
      a, b = a.lower(), b.lower()
      matches = sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i])
      return matches / max(len(a), len(b))

  memorized = []
  for gen in generated_responses:
      for train in train_responses:
          sim = similarity(gen, train)
          if sim > threshold:
              memorized.append((gen[:50], train[:50], sim))
              break

  return {
      "num_memorized": len(memorized),
      "memorization_rate": len(memorized) / len(generated_responses),
      "examples": memorized[:3]
  }
```

### 3. Catastrophic Forgetting[#](#3-catastrophic-forgetting)

The most critical failure mode in SFT: the model "forgets" pretraining knowledge while learning SFT data.

catastrophic\_forgetting.pycpu-only

```
import numpy as np

def simulate_forgetting_curve(learning_rate, num_epochs, data_size=10000):
  """
  Simulate the tradeoff between SFT performance and pretraining preservation.

  Key insight: higher LR or more epochs = better SFT but worse base knowledge.
  """
  # Simulate capability metrics
  base_capability = 1.0  # MMLU, reasoning, math
  sft_alignment = 0.0    # Instruction-following ability

  base_knowledge_over_time = []
  sft_performance_over_time = []

  epoch_steps = data_size // 64  # Assume batch size 64

  for epoch in range(num_epochs):
      for step in range(epoch_steps):
          global_step = epoch * epoch_steps + step

          # SFT learning signal improves alignment
          sft_alignment += learning_rate * 10 * (1 - sft_alignment)

          # Gradient updates cause some forgetting
          # The weight: learning_rate * magnitude of gradient change
          forgetting_rate = learning_rate * 50
          base_capability = base_capability * (1 - forgetting_rate)
          base_capability = max(0.7, base_capability)  # Floor at 70%
```

## Checkpoint Selection[#](#checkpoint-selection)

checkpoint\_selection.pycpu-only

```
import numpy as np

def simulate_training(num_epochs=10):
  """
  Simulate how different metrics evolve during SFT.
  """
  results = []

  for epoch in range(1, num_epochs + 1):
      # Validation loss (keeps decreasing, but eventually overfits)
      val_loss = 2.0 * np.exp(-0.3 * epoch) + 0.1 * epoch / 10 + np.random.normal(0, 0.05)

      # Downstream task performance (peaks then degrades due to forgetting)
      downstream = min(0.85, 0.3 + 0.15 * epoch - 0.02 * epoch**2 + np.random.normal(0, 0.02))

      # Format diversity (decreases as model overfits)
      format_diversity = max(0.1, 0.9 - 0.08 * epoch + np.random.normal(0, 0.03))

      results.append({
          "epoch": epoch,
          "val_loss": val_loss,
          "downstream": downstream,
          "format_diversity": format_diversity,
      })

  return results

results = simulate_training()

print("Checkpoint Selection Analysis")
```

## Scale Thought Experiment[#](#scale-thought-experiment)

1B

7B

70B

175B+

What happens to SFT  
as model size changes?

Model Size

Easy to overfit  
Small data causes problems  
LR must be very conservative

Sweet spot  
Enough capacity to memorize  
But not so large training is cheap

Harder to overfit  
Can train longer epochs  
Likely needs LoRA/QLoRA

Very stable  
Can tolerate more aggressive training  
Requires distributed inference

Practical: Colab-feasible  
Fast iteration possible

Practical: LoRA, RTX4090  
1-2 epochs typical

Practical: LoRA/QLoRA required  
Multiple V100s minimum

Practical: LoRA + sharding  
Expensive, fewer experiments

| Model Size | Overfitting Risk | LR Range | Epochs | LoRA? | Notes |
| --- | --- | --- | --- | --- | --- |
| **1B** | Very high | 1e-6 to 5e-6 | 1 | No | Easy to overfit, needs careful tuning |
| **7B** | High | 5e-6 to 1e-5 | 1-2 | Optional | Sweet spot for experimentation |
| **70B** | Medium | 1e-6 to 5e-6 | 1-3 | Recommended | Can train longer, LoRA saves memory |
| **175B+** | Low | 1e-6 to 5e-6 | 3-5 | Required | Stable but expensive |

## Production Reality[#](#production-reality)

**OpenAI (InstructGPT):**

* 13K demonstrations for SFT
* Human contractors wrote high-quality responses
* Model trained for ~1-2 epochs
* Then moved to RLHF
* Lesson: Quality beats quantity at small scale

**Meta (Llama 2):**

* 27,540 high-quality demonstrations
* Mix of vendor-provided and internally-curated
* Emphasis on helpfulness and safety
* Multiple training stages (SFT → DPO)
* Lesson: Careful curation pays off

**LIMA: Less is More:**

* Just 1,000 carefully curated examples
* Achieved strong instruction-following close to 65B model performance
* Quality > Quantity for SFT
* Lesson: 1000 excellent examples > 10000 mediocre ones

**Practical implications:**

1. Spend time on data quality, not data quantity
2. Train for 1-2 epochs (unless very large model)
3. Use low learning rate and monitor for forgetting
4. Always benchmark against base model on held-out tasks

## Advanced Topics: The Frontier[#](#advanced-topics-the-frontier)

### Why does SFT work with so few examples?[#](#why-does-sft-work-with-so-few-examples)

The conventional wisdom: the base model already has the capability. SFT just "activates" it. This is sometimes called "capability elicitation."

But recent work suggests something deeper. Models may be learning task-specific gating functions:

* "If prompt is `code\_review`, output code feedback style"
* "If prompt is `tutoring`, output step-by-step explanation style"

Question for research: Is SFT teaching new behaviors or routing to existing latent behaviors?

### The format prior and its implications[#](#the-format-prior-and-its-implications)

SFT teaches a strong distributional prior on response format. This can be:

* **Good:** Consistent, professional outputs
* **Bad:** Collapsed formats that ignore context
* **Subtle:** Models learn to mimic the *style* of SFT data, not just content

Advanced question: Can we decouple content learning from format learning? Early work on "format-agnostic" SFT is promising but limited.

### Distribution shift and out-of-distribution robustness[#](#distribution-shift-and-out-of-distribution-robustness)

SFT examples don't cover all possible user queries. What happens when users ask things far from training distribution?

Empirical observations:

1. Model performance drops gracefully OOD (usually)
2. But failure modes can be severe (hallucination, confidence-unrelated-to-accuracy)
3. Uncertainty estimates are often poorly calibrated OOD

Research direction: Can we use SFT to improve OOD robustness, not just in-distribution performance?

### Connection to mechanistic interpretability[#](#connection-to-mechanistic-interpretability)

Recent work (circuits, activation patching) suggests SFT doesn't "reprogram" the model. Instead, it:

1. Increases activation magnitudes on instruction-following circuits
2. Suppresses activation magnitudes on capability circuits that don't align
3. Creates new input-output mappings through linear probing onto existing representations

Deep question: If SFT operates through gating and routing, can we study the circuits that gate instruction-following behavior?

---

## Summary Checklist: SFT Best Practices[#](#summary-checklist-sft-best-practices)

✓ **Data:** 10K-50K high-quality examples (quality > quantity)
✓ **Masking:** Prompt masked out, response trained, smooth transition zone optional
✓ **Learning rate:** 1e-5 to 1e-6 (start conservative, monitor for forgetting)
✓ **Scheduling:** Cosine decay with warmup (100-500 steps warmup)
✓ **Batch size:** 64-128 (smaller with lower LR, larger with more data)
✓ **Epochs:** 1-3 (usually 1-2 is enough)
✓ **Checkpoint:** Based on downstream metrics, not loss
✓ **Verification:** Monitor base model performance on held-out tasks
✓ **Format:** Diverse response formats, detect collapse early
✓ **Memorization:** Spot check for verbatim training set matches

---

*Next up: RLHF and beyond. SFT gets us instruction-following. But how do we optimize for human values? Enter reward models and policy optimization -- the hard part.*

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. You have 20K SFT examples with an average prompt length of 50 tokens and response length of 200 tokens. If you mask the prompt and train only on response tokens, what fraction of total tokens contribute to the loss? Estimate the effective training tokens per epoch.
2. A pretraining run used LR = 3e-4. You want to fine-tune the resulting 7B model on 10K instruction examples. Estimate an appropriate SFT learning rate and justify your choice relative to the pretraining LR.
3. After 3 epochs of SFT, your model scores 0.82 on instruction-following but MMLU dropped from 0.65 to 0.58. Calculate the capability regression. Would you deploy this checkpoint, and what is the first fix you would try?