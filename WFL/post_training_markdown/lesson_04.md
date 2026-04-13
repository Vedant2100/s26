In this tutorial, you will measure catastrophic forgetting by comparing benchmark scores before and after fine-tuning, simulate gradient interference between tasks, implement mitigation strategies including data replay and EWC, and design a fine-tuning protocol with explicit capability constraints.

## Prerequisites Refresher[#](#prerequisites-refresher)

▶How does fine-tuning differ from pretraining?

**Pretraining** optimizes for next-token prediction on diverse, unlabeled internet text. The model learns broad, general representations.

**Fine-tuning** is a second optimization phase on a narrower task (e.g., instruction-following, safety). You're using a much smaller dataset, fewer steps, and a much lower learning rate.

The risk: pretraining took 6 months on 2T tokens. Fine-tuning might take 2 days on 10M tokens. If you use a learning rate that's too high or train too long, the model "forgets" what it learned in those 2T tokens.

This is not theoretical -- it happens constantly in production.

▶What's a weight in a neural network?

A neural network is just a function: given input `x`, produce output `y`. The function is parameterized by weights (matrices) and biases (vectors).

During pretraining, these weights are adjusted via gradient descent to minimize the next-token prediction loss. Each weight "specializes" for different aspects of language understanding.

During fine-tuning, we adjust those same weights again. The question is: can we improve on the new task without breaking the old representations?

▶What's a gradient and why does it cause interference?

The gradient tells us how to change each weight to reduce loss. It's computed via backprop.

**Gradient interference** happens when:

* Task A optimal direction: move weight w toward +1
* Task B optimal direction: move weight w toward -1
* Gradient for task B points toward -1
* But we need w to stay near +1 for task A

So we have a conflict. Whichever gradient we follow, we move away from the other task's optimum. This is unavoidable with shared weights.

With enough capacity and careful data mixing, you can find directions that help both tasks. But small models or very different tasks have severe interference.

## Mechanisms of Forgetting[#](#mechanisms-of-forgetting)

There are actually three distinct mechanisms at play:

**1. Gradient Interference** -- The gradients for different tasks point in conflicting directions in weight space.

**2. Loss Landscape Shift** -- As you update weights, the loss landscape itself changes. A region that used to be a good optimum for old tasks might become terrible.

**3. Feature Forgetting** -- The low-level features (embeddings, attention patterns) that the old task depended on get overwritten by new patterns.

The first mechanism is unavoidable with shared weights. The second two can be mitigated with the right techniques.

### Gradient Interference in Detail[#](#gradient-interference-in-detail)

gradient\_interference.pycpu-only

```
import numpy as np

def simulate_gradient_interference():
  """
  Demonstrate how learning new tasks can interfere with old tasks.

  Simple scenario: 2D weights, 2 tasks with different optimal directions.
  """

  # Initial weights (trained on Task A)
  w = np.array([1.0, 0.5])

  # Task A optimal direction
  task_a_optimal = np.array([1.0, 0.5])

  # Task B optimal direction (different!)
  task_b_optimal = np.array([0.2, 1.0])

  history = [w.copy()]
  lr = 0.1

  print("Gradient Interference Simulation")
  print("=" * 60)
  print("Initial weights: %s" % w)
  print("Task A optimal:  %s" % task_a_optimal)
  print("Task B optimal:  %s" % task_b_optimal)
  print()

  # Task A and B gradients are orthogonal -- pure conflict
  grad_a = task_a_optimal - w
```

Result

Fine-tuning: 2 days, 10M tokens

Pretraining: 6 months, 2T tokens

Weights optimized for

General Knowledge

Gradient for Instruction-Following

Weights shift to optimize new task

✓ Instruction-Following improves

✗ Math, Code, Facts degrade

## Detecting Forgetting[#](#detecting-forgetting)

The first step in fighting forgetting is measuring it. You need a **forgetting detection suite**.

detect\_forgetting.pycpu-only

```
import numpy as np
from dataclasses import dataclass

@dataclass
class EvaluationResult:
  benchmark: str
  base_score: float
  finetuned_score: float

  @property
  def regression(self):
      return self.base_score - self.finetuned_score

  @property
  def regression_pct(self):
      if self.base_score == 0:
          return 0
      return self.regression / self.base_score * 100

def evaluate_capabilities(is_finetuned: bool = False):
  """
  Simulate evaluating a model on multiple benchmarks.
  Returns scores that illustrate the forgetting phenomenon.
  """
  # Base model scores (realistic for 7B model)
  base_scores = dict(
      MMLU=0.65,
      HumanEval=0.45,
      GSM8K=0.58,
      TriviaQA=0.72,
```

### Detecting Gradient Projection[#](#detecting-gradient-projection)

A more sophisticated detection method: measure how much fine-tuning gradients **align with** pretraining loss landscape directions.

detect\_gradient\_alignment.pycpu-only

```
import numpy as np

def analyze_gradient_alignment(num_weights=1000):
  """
  Simulate gradient projection analysis.

  High-level idea:
  - Compute gradients on pretraining tasks (base model's "good directions")
  - Compute gradients on fine-tuning tasks
  - If they point opposite directions → expect forgetting
  - If they align → compatible learning
  """

  # Simulate Fisher Information Matrix diagonal (which gradients matter for pretraining)
  fisher_diag = np.abs(np.random.randn(num_weights)) ** 2
  fisher_diag = fisher_diag / fisher_diag.max()  # Normalize

  # Gradient for fine-tuning
  finetuning_grad = np.random.randn(num_weights)

  # Project onto important directions (high Fisher)
  important_indices = fisher_diag > 0.5
  projection_on_important = np.abs(finetuning_grad[important_indices]).mean()

  # Project onto unimportant directions
  unimportant_indices = fisher_diag <= 0.5
  projection_on_unimportant = np.abs(finetuning_grad[unimportant_indices]).mean()

  print("Gradient Alignment Analysis")
  print("=" * 60)
```

## Mitigation Strategies[#](#mitigation-strategies)

### Strategy 1: Data Replay[#](#strategy-1-data-replay)

Mix pretraining data into the SFT dataset to maintain general capabilities.

The intuition: if you're updating weights based on SFT loss, you're moving away from pretraining optima. Mixing in pretraining data keeps you "honest" -- it pulls you back toward good directions for old tasks.

data\_replay.pycpu-only

```
import numpy as np

def simulate_training_with_replay(replay_ratio: float, num_steps: int = 100):
  """
  Simulate SFT with data replay.

  replay_ratio: fraction of each batch that is pretraining data
  """
  # Capability starts at 1.0 (base model)
  capability = 1.0
  alignment = 0.0

  history = []

  for step in range(num_steps):
      # SFT data improves alignment, may hurt capability
      sft_contribution = (1 - replay_ratio)
      alignment += sft_contribution * 0.02 * (1 - alignment)
      capability -= sft_contribution * 0.005

      # Replay data maintains capability
      capability += replay_ratio * 0.003  # Slight recovery

      # Floor at some minimum
      capability = max(0.5, min(1.0, capability))
      alignment = max(0, min(1.0, alignment))

      history.append(dict(
          step=step,
          capability=capability,
```

### Strategy 2: Elastic Weight Consolidation (EWC)[#](#strategy-2-elastic-weight-consolidation-ewc)

Penalize changes to weights that are important for previous tasks.

The core idea: not all weights matter equally. Some weights (high Fisher Information) are critical for pretraining performance. Others are nearly irrelevant. EWC lets you change unimportant weights freely while locking down the critical ones.

ewc\_concept.pycpu-only

```
import numpy as np

def compute_fisher_information(weights, task_gradients):
  """
  Fisher Information approximates weight importance for a task.

  Weights with high gradient variance are more important --
  changing them significantly impacts task performance.

  Fisher ≈ E[gradient²] -- how much the gradient varies per weight.
  """
  # Fisher ≈ E[gradient²]
  fisher = np.mean(task_gradients**2, axis=0)
  return fisher

def ewc_loss(
  current_weights,
  old_weights,
  fisher_importance,
  sft_loss,
  ewc_lambda=1000
):
  """
  EWC adds a penalty for moving away from old weights,
  scaled by their importance.

  Total loss = SFT loss + λ * Σ F_i * (θ_i - θ*_i)²

  The penalty term: quadratic distance, scaled by Fisher.
  """
```

### Strategy 3: Low Learning Rate + Early Stopping[#](#strategy-3-low-learning-rate-early-stopping)

The simplest and often most effective approach. This is what most labs actually use in production.

Why it works: a small learning rate means each gradient step moves less in weight space. You're still moving toward the fine-tuning objective, but slowly enough that you don't wander too far from pretraining optima.

lr\_early\_stopping.pycpu-only

```
import numpy as np

def simulate_lr_forgetting(learning_rate, num_epochs, base_capability=1.0):
  """
  Simulate how LR affects the capability-alignment tradeoff.
  """
  capability = base_capability
  alignment = 0.1

  trajectory = []

  for epoch in range(num_epochs):
      # Higher LR = faster learning but more forgetting
      alignment += learning_rate * 100 * (1 - alignment)
      capability -= learning_rate * 50 * (1 - (1 - alignment))

      capability = max(0.3, capability)
      alignment = min(1.0, alignment)

      trajectory.append(dict(
          epoch=epoch,
          capability=capability,
          alignment=alignment,
      ))

  return trajectory

print("Learning Rate vs Forgetting Tradeoff")
print("=" * 60)
print()
```

### Strategy 4: Parameter-Efficient Fine-Tuning (LoRA)[#](#strategy-4-parameter-efficient-fine-tuning-lora)

Don't update all weights. Add a low-rank adapter that modifies only a small subspace.

The breakthrough insight: you don't need to update all 7B weights to learn a new task. You can learn a small ~0.1B parameter adapter that projects gradients into a low-rank space. The base weights stay frozen.

This is why LoRA is so popular in practice: it almost eliminates forgetting by design.

**Key numbers:**

* LoRA rank r=8-32: ~0.5-4M parameters for a 7B model
* Forgetting reduction: 50-80% compared to full fine-tuning
* Training speed: 2-3x faster
* Trade-off: slightly worse task performance than full fine-tuning

We'll dive deep into LoRA mechanics in a later lesson. For now: understand that it's a structural solution to forgetting. Instead of risking damage to 7B weights, you learn a tiny adapter.

## Integrating Strategies: The Complete Protocol[#](#integrating-strategies-the-complete-protocol)

In practice, you combine multiple strategies:

Yes

No

Start Fine-Tuning

Establish Capability Baseline

Prepare Training Data

Mix 10-30% Pretraining Data

Use LoRA or EWC

Set LR: 1e-5 to 5e-6

Train with Frequent Evals

Eval Capability Benchmarks

Regression Acceptable?

Continue Training

Stop Training

Final Eval Suite

Deploy with Eval Report

## Break It: Catastrophic Forgetting Scenarios[#](#break-it-catastrophic-forgetting-scenarios)

Let's demonstrate what happens when you get fine-tuning wrong.

break\_it\_forgetting.pycpu-only

```
import numpy as np

def aggressive_finetuning(num_epochs=10, lr=1e-4):
  """
  Demonstrate how aggressive fine-tuning destroys capabilities.

  THIS IS WHAT YOU DON'T WANT TO DO.
  """
  capabilities = dict(
      math=0.75,
      code=0.68,
      facts=0.82,
      reasoning=0.70,
      instruction=0.30,
  )

  print("SCENARIO 1: Aggressive Fine-Tuning (CATASTROPHIC)")
  print("=" * 60)
  print("Learning rate: %.0e (too high!)" % lr)
  print("Epochs: %d (way too many!)" % num_epochs)
  print("Data replay: None (mistake!)")
  print()
  print("%6s %8s %8s %8s %8s %10s" % ("Epoch", "Math", "Code", "Facts", "Reason", "Instruct"))
  print("-" * 55)

  for epoch in range(num_epochs + 1):
      if epoch == 0:
          print("%6s" % "Base", end="")
      else:
          print("%6d" % epoch, end="")
```

## Scale Thought Experiment[#](#scale-thought-experiment)

Forgetting risk varies dramatically by model size:

| Model Size | Forgetting Risk | Why | Recommended Approach |
| --- | --- | --- | --- |
| **1B** | CRITICAL | Few parameters = less redundancy, high interference | Very low LR (5e-6), data replay essential, LoRA preferred |
| **7B** | MODERATE | Medium redundancy, some slack in weight space | 1e-5 LR, 1-2 epochs, optional replay, LoRA recommended |
| **70B** | LOW | Many parameters = high redundancy, graceful degradation | 5e-6 LR, LoRA standard, light eval needed |
| **175B+** | VERY LOW | Massive redundancy, pretraining value huge, fine-tuning is risky | LoRA almost mandatory, base weights are sacred |

**Why model size matters:**

Larger models have more "slack" in weight space. A 175B model can encode instruction-following WITHOUT destroying math knowledge because there's enough capacity for both. A 1B model can't afford that luxury.

This is why research labs spend so much on pretraining but are careful with fine-tuning. The cost of getting it wrong is proportional to the pretraining cost.

### The "Pretraining Investment" Principle[#](#the-pretraining-investment-principle)

```
Risk of Forgetting ≈ (Pretraining Cost) / (Model Capacity)

Small model, expensive pretraining → Use LoRA, be conservative
Large model, expensive pretraining → Use LoRA, be more careful

```
In practice: all labs now use LoRA for any model they spent significant resources pretraining.

## Production Reality: How Labs Actually Do It


### OpenAI's Approach

- **Capabilities regression suite**: Continuous testing on 50+ benchmarks during training
- **Automated rollback**: If MMLU or code performance drop > 2%, stop and revert
- **Checkpoint selection**: Pick the model that maximizes alignment while staying above capability thresholds
- **Multi-round iteration**: Fine-tune, evaluate, adjust data, fine-tune again

### Anthropic's Approach (Constitutional AI)

- **Self-critique on both axes**: Model critiques itself for alignment AND capability degradation
- **Iterative refinement**: Use the model's own judgments to improve future fine-tuning data
- **Red teaming for capabilities**: Actively search for capability regressions, not just alignment issues
- **Data curation**: Heavily bias toward high-quality examples that improve both metrics

### Meta's Approach (Llama)

- **Diverse evaluation suite**: Test on math, code, facts, reasoning, instruction-following
- **LoRA by default**: Never fine-tune all weights on production models
- **Data replay mandatory**: 20-30% of SFT batches are pretraining-like data
- **Conservative LR**: Typically 5e-6 or lower for large models

### Practical Fine-Tuning Protocol (Implemented by Most Labs)
```

BEFORE FINE-TUNING:

1. Run comprehensive eval on base model → save as baseline
2. Verify all benchmarks (MMLU, HumanEval, GSM8K, etc.)

DURING FINE-TUNING:
3. Use LoRA (r=16-32) or EWC, never bare fine-tuning
4. Mix 20-30% pretraining data into SFT batches
5. Set LR = 5e-6 to 1e-5 (depending on model size)
6. Evaluate full benchmark suite every N steps
7. Track Pareto frontier: (alignment score, capability floor)

AT CHECKPOINT SELECTION:
8. Don't pick the final checkpoint
9. Pick the checkpoint where:

* Instruction-following > threshold (e.g., 0.7)
* Capability regression < 3% per benchmark
* Earliest in training (less risk)

AFTER FINE-TUNING:
10. Full regression test on baseline benchmarks
11. Red team for capability failures
12. Document: which capabilities degraded, by how much
13. Deploy with eval report

```
### Real Example: Llama 2 7B → Chat

From the Llama 2 paper:
- Base: MMLU 46.8%, HumanEval 12.8%
- After SFT: MMLU 47.1%, HumanEval 13.1% (minimal regression!)
- Training: 2 epochs, ~27.5K SFT examples, low learning rate
- Key: quality over quantity, careful checkpoint selection

This is the gold standard: improve instruction-following while PRESERVING or slightly improving benchmarks.

## Common Pitfalls & How to Avoid Them


### Pitfall 1: Over-Training (The Multi-Epoch Trap)

**What happens:** You train for 5+ epochs to maximize alignment score on your SFT benchmark.

**What goes wrong:** Each epoch pushes further from pretraining optima. You get 0.85 alignment but lose 8% on GSM8K.

**How to avoid:** Stop after 2-3 epochs. If you're not at acceptable alignment, your SFT data is bad, not your training time.

**Rule:** "Is the model learning?" Yes? → Monitor and early stop. No learning after 1 epoch? → Rewrite your SFT data, don't train longer.

### Pitfall 2: Ignoring Model-Size-Dependent Effects

**What happens:** You use the same fine-tuning protocol for a 1B model as a 70B model.

**What goes wrong:** The 1B model forgets everything. The 70B model is fine.

**How to avoid:** Scale your conservatism with model size.
- 1B: ultra-conservative (5e-6 LR, 30% replay, LoRA essential)
- 7B: moderate (1e-5 LR, 20% replay, LoRA recommended)
- 70B+: can afford more (5e-6 LR, light replay, LoRA standard)

### Pitfall 3: No Baseline Evaluation

**What happens:** You fine-tune, then notice Math performance is bad. But bad compared to what?

**What goes wrong:** You have no ground truth. You can't measure forgetting.

**How to avoid:** Always run a full eval suite on the base model FIRST. Save those numbers. Compare everything against them.

### Pitfall 4: Forgetting the Pareto Frontier

**What happens:** You pick the checkpoint with the highest alignment score.

**What goes wrong:** It has terrible capability degradation.

**How to avoid:** Think in terms of Pareto frontier:
- X-axis: Instruction-following score
- Y-axis: Capability floor (min of all benchmark scores)
- Pick the checkpoint where both are acceptable, not the frontier extreme.

## Complete Fine-Tuning Simulation


Let's run a realistic fine-tuning scenario with all the mitigation strategies:

<CodeCell
  filename="complete_finetuning_protocol.py"
  compute_estimate="cpu-only"
  code={`import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class FinetuningConfig:
    learning_rate: float
    num_epochs: int
    data_replay_ratio: float
    use_lora: bool
    eval_interval: int = 10
    min_capability_floor: float = 0.70

    def describe(self):
        return "LR=%.0e, Epochs=%d, Replay=%.0f%%, LoRA=%s" % (self.learning_rate, self.num_epochs, self.data_replay_ratio * 100, self.use_lora)

def simulate_finetuning_run(config: FinetuningConfig, seed: int = 42):
    """
    Simulate a complete fine-tuning run with evals, checkpoint selection,
    and Pareto frontier tracking.
    """
    np.random.seed(seed)

    # Initial capabilities (base model)
    capabilities = dict(
        mmlu=0.65,
        humaneval=0.45,
        gsm8k=0.58,
        triviaqa=0.72,
        instruction_following=0.35,
    )

    # Track for plotting/analysis
    checkpoints = []
    best_checkpoint = None
    best_score = -np.inf

    print("\\nFine-Tuning Protocol: %s" % config.describe())
    print("=" * 70)
    print()
    print("%5s %10s %8s %10s %8s %10s %20s" % ("Step", "Instruct", "MMLU", "HumanEval", "GSM8K", "TriviaQA", "Action"))
    print("-" * 70)

    # Simulate training
    total_steps = config.num_epochs * 50  # 50 steps per epoch
    step = 0

    for epoch in range(config.num_epochs):
        for mini_epoch in range(50):
            step += 1

            # Update from SFT data
            sft_improvement = 0.01 if not config.use_lora else 0.0075
            sft_forgetting = 0.003 if config.data_replay_ratio > 0.2 else 0.005

            # Scale by learning rate
            sft_improvement *= config.learning_rate / 5e-6
            sft_forgetting *= config.learning_rate / 5e-6

            # Update capabilities
            capabilities["instruction_following"] = min(
                0.95,
                capabilities["instruction_following"] + sft_improvement
            )

            for cap in ["mmlu", "humaneval", "gsm8k", "triviaqa"]:
                # Replay mitigates forgetting
                replay_factor = 1.0 - config.data_replay_ratio
                capabilities[cap] -= sft_forgetting * replay_factor
                capabilities[cap] = max(0.3, capabilities[cap])

            # LoRA reduces forgetting
            if config.use_lora:
                for cap in ["mmlu", "humaneval", "gsm8k", "triviaqa"]:
                    capabilities[cap] += 0.001

            # Periodic evaluation
            if step % config.eval_interval == 0:
                capability_floor = min(
                    [capabilities[c] for c in ["mmlu", "humaneval", "gsm8k", "triviaqa"]]
                )

                # Compute checkpoint score (weighted combination)
                alignment_score = capabilities["instruction_following"]
                pareto_score = alignment_score if capability_floor >= config.min_capability_floor else -np.inf

                # Check if this is best so far
                action = ""
                if pareto_score > best_score and capability_floor >= config.min_capability_floor:
                    best_score = pareto_score
                    best_checkpoint = dict(
                        step=step,
                        epoch=epoch,
                        capabilities=capabilities.copy(),
                    )
                    action = "✓ NEW BEST"
                elif step == total_steps:
                    action = "(final)"

                if step % (config.eval_interval * 5) == 0 or action:
                    print(
                        "%5d "
                        "%10.3f "
                        "%8.3f "
                        "%10.3f "
                        "%8.3f "
                        "%10.3f "
                        "%20s"
                        % (step, capabilities["instruction_following"], capabilities["mmlu"], capabilities["humaneval"], capabilities["gsm8k"], capabilities["triviaqa"], action)
                    )

    print()
    print("=" * 70)
    if best_checkpoint:
        print("Best Checkpoint Selected: Step %d (Epoch %d)" % (best_checkpoint["step"], best_checkpoint["epoch"]))
        print()
        print("%20s %10s %20s" % ("Capability", "Score", "Regression from Base"))
        print("-" * 55)
        base_caps = dict(mmlu=0.65, humaneval=0.45, gsm8k=0.58, triviaqa=0.72)
        for cap in ["instruction_following", "mmlu", "humaneval", "gsm8k", "triviaqa"]:
            score = best_checkpoint["capabilities"][cap]
            if cap in base_caps:
                regression = (base_caps[cap] - score) / base_caps[cap] * 100
                print("%20s %10.3f %19.1f%%" % (cap, score, regression))
            else:
                print("%20s %10.3f %19s" % (cap, score, "(new)"))
    else:
        print("NO ACCEPTABLE CHECKPOINT FOUND")
        print("(all checkpoints violated capability floor)")

    return best_checkpoint

# Scenario 1: Aggressive (what NOT to do)
print("\\n" + "="*70)
print("SCENARIO 1: AGGRESSIVE (High LR, no replay, no LoRA)")
print("="*70)
config1 = FinetuningConfig(
    learning_rate=1e-4,
    num_epochs=5,
    data_replay_ratio=0.0,
    use_lora=False,
)
result1 = simulate_finetuning_run(config1, seed=42)

# Scenario 2: Recommended (best practice)
print("\\n" + "="*70)
print("SCENARIO 2: RECOMMENDED (Low LR, replay, LoRA)")
print("="*70)
config2 = FinetuningConfig(
    learning_rate=5e-6,
    num_epochs=2,
    data_replay_ratio=0.25,
    use_lora=True,
)
result2 = simulate_finetuning_run(config2, seed=42)

# Summary
print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
if result1:
    r1_cap_floor = min([result1["capabilities"][c] for c in ["mmlu", "humaneval", "gsm8k", "triviaqa"]])
    print("Aggressive: Instruction=%.3f, Cap Floor=%.3f" % (result1["capabilities"]["instruction_following"], r1_cap_floor))
else:
    print("Aggressive: FAILED (no valid checkpoint)")

if result2:
    r2_cap_floor = min([result2["capabilities"][c] for c in ["mmlu", "humaneval", "gsm8k", "triviaqa"]])
    print("Recommended: Instruction=%.3f, Cap Floor=%.3f" % (result2["capabilities"]["instruction_following"], r2_cap_floor))

print()
print("Key insight: The recommended protocol maintains a capability floor")
print("while still achieving good instruction-following. It's not a failure")
print("mode like the aggressive approach.")
`}
/>

**Why do some capabilities survive while others don't?**

Early-layer features (learned in the first weeks of pretraining) seem remarkably robust. Later-layer features are more brittle. If we understood this better, we could design fine-tuning to protect early layers and retrain later ones.

**Modular networks for alignment:**

If we could isolate "capability modules" (math, code, facts) from "behavior modules" (instruction-following, safety), we might fine-tune only behavior. Recent mechanistic interpretability work is moving in this direction.

**The forgetting-generalization tradeoff:**

Surprisingly, some forgetting might actually help generalization. By moving away from pretraining optima, you might prevent the model from overfitting to internet patterns. How do we distinguish beneficial vs harmful forgetting?

**Continual learning theory:**

Catastrophic forgetting is one of the oldest problems in machine learning (since the 1990s). Techniques like Experience Replay and Elastic Weight Consolidation come from continual learning. There's likely more to steal from that literature.

## Key Takeaways


1. **Forgetting is real and costly.** You can lose 5-15% of your pretraining value in 2 days of sloppy fine-tuning.

2. **Use multiple defenses:** Low LR + data replay + LoRA + frequent evals = strong protection.

3. **LoRA is now standard.** If you're not using it, you should have a good reason.

4. **Measure before you act.** Establish baselines, evaluate frequently, use Pareto frontier thinking.

5. **Data quality > training time.** LIMA principle: 1K high-quality SFT examples often beats 100K mediocre ones.

6. **Scale matters.** A 1B model needs more mitigation than a 70B model.

---

*Next up: Data quality is the biggest lever for SFT effectiveness. We'll see why the LIMA paper's findings -- 1K carefully curated examples beats 100K random examples -- changed how everyone fine-tunes now.*

## Checkpoint Questions

Use these to verify understanding before moving on:
1. A 7B model scores 0.65 on MMLU, 0.45 on HumanEval, and 0.58 on GSM8K before SFT. After SFT it scores 0.60, 0.43, and 0.50 respectively. Calculate the per-benchmark regression percentages. Which capability degraded most, and what does that suggest about your SFT data composition?
2. You are fine-tuning with 20% data replay. If your SFT dataset has 10K examples, how many pretraining-like examples do you add per epoch? If each SFT example averages 250 tokens and each replay example averages 500 tokens, estimate the total tokens per epoch.
3. Given a learning rate of 1e-5 and batch size 64 on a 10K-example SFT dataset, compute the number of gradient update steps per epoch. If you observe capability regression after 2 epochs, estimate whether switching to LR = 5e-6 or adding 30% replay would reduce total weight displacement more.
```