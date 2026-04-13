In this tutorial, you will build a complete RLHF training loop, select hyperparameters that keep training stable, and diagnose common failure modes (reward hacking, mode collapse, KL explosion) from their metric signatures.

RLHF combines three components that interact in non-obvious ways: policy optimization (PPO), an imperfect reward model, and a KL constraint. Most failures come from interactions between these components rather than any single broken part.

## Prerequisites Refresher[#](#prerequisites-refresher)

▶What is PPO?

Proximal Policy Optimization (PPO) is a policy gradient algorithm that optimizes:

```
L(θ) = 𝔼[min(r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A)]
```

Where:

* `r(θ) = π_new(a|s) / π_ref(a|s)` is the probability ratio
* `A` is the advantage (baseline-adjusted return)
* `clip()` prevents large policy updates

**Key insight:** PPO avoids catastrophic forgetting by limiting how much the policy can change per step. In RLHF, this is your stability mechanism.

**Why PPO in RLHF?** It's stable, sample-efficient, and prevents reward hacking better than vanilla policy gradient.

▶What is the KL penalty?

The KL divergence measures how far your policy has drifted from the reference model:

```
KL(π_new || π_ref) = 𝔼[log(π_new(a|s) / π_ref(a|s))]
```

In RLHF, you optimize:

```
L_total = reward(response) - β · KL(π_new || π_ref)
```

**Why β matters:**

* `β` too high: Policy doesn't improve (KL penalty dominates)
* `β` too low: Policy chases reward, diverges from reference
* `β` just right: Policy improves while staying close to reference

**In practice:** β often needs to *adapt* based on how far KL drifts.

▶What's the relationship between value function and advantage?

The value function `V(s)` estimates the expected return from state s. The advantage is:

```
A(s,a) = R(s,a) - V(s)
```

This tells us: *"Was this action better than average for this state?"*

**In RLHF:** You need a good value function estimate because a bad baseline leads to high-variance advantage estimates, which makes gradient estimates noisy and training unstable.

**Rule of thumb:** If your value loss isn't decreasing, your advantage estimates are garbage, and PPO will struggle.

## The Complete RLHF Training Loop[#](#the-complete-rlhf-training-loop)

Now let's build the actual training loop that you'll need to understand for production RLHF systems.

rlhf\_training\_loop.pycpu-only

```
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class RLHFConfig:
  """Configuration for RLHF training."""
  # Model learning rates
  policy_lr: float = 5e-6
  value_lr: float = 1e-5

  # PPO hyperparameters
  ppo_epochs: int = 3
  ppo_batch_size: int = 32
  ppo_clip_epsilon: float = 0.2

  # KL penalty
  initial_kl_beta: float = 0.1
  target_kl: float = 6.0
  kl_adaptive: bool = True

  # Generation
  generation_batch_size: int = 128
  max_response_length: int = 100

  # Training
  num_iterations: int = 100
  warmup_steps: int = 10
  gradient_clip_norm: float = 1.0
```

This is what production RLHF looks like. Notice the three-way balancing act:

* **Policy loss** gets better but can't move too far from reference
* **KL divergence** needs to stay in bounds
* **β adapts** to keep KL under control

The simulation above is simplified (no actual gradients, deterministic rewards), but the *structure* is identical to real training.

## Hyperparameter Selection[#](#hyperparameter-selection)

hyperparameters.pycpu-only

```
def rlhf_hyperparameters():
  """
  Recommended RLHF hyperparameters with reasoning.
  """
  config = """
RLHF HYPERPARAMETER GUIDE
=========================

LEARNING RATE
-------------
- Policy LR: 1e-6 to 5e-6 (MUCH lower than SFT)
* Why? We're fine-tuning an already-trained 7B/13B/70B model
* SFT uses ~1e-5, but RLHF compounds changes across all parameters
* Too high (>1e-5): Risk of KL explosion or reward hacking
* Too low (<1e-7): Policy won't learn, training stalls

- Value LR: 1e-5 to 5e-5 (can be slightly higher than policy)
* Value function needs to adapt faster than policy
* If value loss doesn't decrease, everything else breaks
* Start at 2x policy LR, adjust based on value loss trend

BATCH SIZE
----------
- Generation batch: 256-1024 responses per iteration
* Each response needs forward+backward through RM and policy
* Larger = lower variance, more stable KL
* 256 minimum; 512-1024 for stable production training
* Cost: Memory scales linearly with batch

- PPO minibatch: 32-128 per gradient step
```

## Monitoring Dashboard: What to Watch[#](#monitoring-dashboard-what-to-watch)

The difference between successful RLHF and fiasco is *good monitoring*. You need to measure five domains simultaneously.

monitoring.pycpu-only

```
import numpy as np

def create_monitoring_dashboard():
  """
  Key metrics to track during RLHF training.
  Track these metrics during every training run.
  """
  print("RLHF MONITORING DASHBOARD")
  print("=" * 70)
  print()

  metrics = dict(
      REWARD_METRICS=[
          ("Mean reward", "Should increase over time"),
          ("Reward variance", "Should decrease as policy improves"),
          ("Reward max/min", "Check for exploitation: sudden huge rewards"),
          ("Reward growth rate", "Should slow over time (log scale)"),
      ],
      KL_METRICS_CRITICAL=[
          ("Mean KL", "Should stay near target (~6), NOT explode"),
          ("KL variance", "High variance = policy unstable"),
          ("beta coefficient", "Should adapt smoothly, not oscillate"),
          ("beta ratio", "Track how much we're increasing penalty"),
      ],
      POLICY_METRICS=[
          ("Policy entropy", "Should decrease slowly, NOT collapse to 0"),
          ("Max action probability", "If > 0.95, mode collapse happening"),
          ("Clip fraction", "Should be 10-30%, indicates learning rate OK"),
          ("Action diversity", "Sample 10 responses; are they similar?"),
      ],
```

### Visualization: The RLHF Metrics Triangle[#](#visualization-the-rlhf-metrics-triangle)

```
graph TB
    subgraph M["RLHF Stability Triangle"]
        R["Reward Growth<br/>(↑ good)"]
        K["KL Constraint<br/>(stable around target)"]
        E["Policy Entropy<br/>(↓ slow, not collapse)"]
    end

    R -->|"Too high<br/>LR"| Unstable["Instability Zone<br/>KL explosion"]
    K -->|"β too low"| Unstable
    E -->|"Mode collapse"| Unstable

    R -->|"Too low<br/>LR"| NoLearning["No Learning Zone"]
    K -->|"β too high"| NoLearning
    E -->|"Frozen policy"| NoLearning

    style Unstable fill:#ffcccc
    style NoLearning fill:#ccccff
    style M fill:#ffffcc
```

These three metrics form a stability triangle. Optimize one in isolation and you'll break another.

## Common Failure Modes[#](#common-failure-modes)

Every RLHF project hits these five failure modes. Knowing their signatures lets you diagnose in minutes instead of days.

### Failure Mode Flowchart[#](#failure-mode-flowchart)

```
graph TB
    Start["RLHF Training Started"]

    Start --> CheckReward{"Reward<br/>increasing?"}

    CheckReward -->|"No"| NoLearning["NO LEARNING"]
    CheckReward -->|"Yes, but KL bad"| KLCheck{"KL staying<br/>in bounds?"}
    CheckReward -->|"Yes, KL OK"| EntCheck{"Entropy<br/>stable?"}

    KLCheck -->|"No, exploding"| KLExp["KL EXPLOSION"]
    KLCheck -->|"No, β broken"| AdaptFail["ADAPTIVE β FAILED"]

    EntCheck -->|"No, collapsed"| ModeCol["MODE COLLAPSE"]
    EntCheck -->|"Yes"| RewardQual{"Human eval<br/>improves?"}

    RewardQual -->|"No"| RewardHack["REWARD HACKING"]
    RewardQual -->|"Yes"| Success["TRAINING OK"]

    NoLearning --> Cause1["LR too low<br/>or β too high"]
    KLExp --> Cause2["LR too high<br/>or β too low"]
    ModeCol --> Cause3["Policy collapsed<br/>to local optimum"]
    RewardHack --> Cause4["RM exploitable<br/>KL penalty weak"]
    AdaptFail --> Cause5["KL far from target<br/>β oscillating"]

    style NoLearning fill:#ff9999
    style KLExp fill:#ff9999
    style ModeCol fill:#ff9999
    style RewardHack fill:#ff9999
    style AdaptFail fill:#ff9999
    style Success fill:#99ff99
```

failure\_diagnosis.pycpu-only

```
import numpy as np

def diagnose_rlhf_failures():
  """
  Common RLHF failure modes and their signatures.
  This is the diagnostic guide you'll use in production.
  """
  failures = dict(
      REWARD_HACKING=dict(
          signature=[
              "Reward increases rapidly (e.g., 0->5 in 10 steps)",
              "KL increases proportionally (policy diverging)",
              "Generations become repetitive, formulaic, or incoherent",
              "Manual inspection: responses sound 'gamed' or unnatural",
              "Human eval: 'Better on your metric, worse in practice'",
          ],
          root_cause="Policy found exploitable patterns in RM",
          diagnosis=[
              "Sample 50 generations; read them manually",
              "Compute RM score distribution; is it multimodal?",
              "Check: does RM upweight token length, repetition, or formatting?",
          ],
          fixes=[
              "Increase beta (stronger KL penalty) by 2-3x",
              "Ensemble multiple RMs; average scores",
              "Retrain RM on adversarial examples from policy",
              "Add auxiliary rewards: penalize repetition, prefer diversity",
          ],
      ),
      MODE_COLLAPSE=dict(
```

### "Break It" Exercise: Trigger Each Failure Mode[#](#break-it-exercise-trigger-each-failure-mode)

Now let's intentionally break RLHF in each way to see the signatures:

break\_rlhf.pycpu-only

```
import numpy as np

class FailureModeDemonstrator:
  """
  Intentionally trigger each failure mode.
  This is what you'll see in production when something goes wrong.
  """

  @staticmethod
  def break_reward_hacking():
      """
      Scenario: beta too low -> policy exploits RM
      """
      print("BREAKING RLHF: Reward Hacking")
      print("-" * 60)

      np.random.seed(42)

      # Scenario: RM is length-biased (a REAL flaw!)
      def biased_reward_model(response):
          """RM that rewards longer responses."""
          return len(response) / 100.0 + np.random.randn() * 0.1

      # Simulate training with very low beta
      kl_beta = 0.001  # Too low!

      responses_over_time = []
      metrics = []

      for step in range(20):
```

Notice:

1. **Reward hacking** shows reward and KL both growing → exploitation
2. **Mode collapse** shows entropy dropping below 1.0 → policy peaked
3. **KL explosion** shows KL increasing every step → divergence unstoppable

These three signatures let you diagnose in *minutes*. Without them, you waste days tweaking hyperparameters blindly.

## Systematic Debugging Workflow[#](#systematic-debugging-workflow)

The key to fast RLHF debugging is a *systematic process*, not random hyperparameter tweaking.

debugging\_workflow.pycpu-only

```
def rlhf_debugging_workflow():
  """
  Step-by-step debugging process. Use this every time RLHF breaks.
  Use this process systematically.
  """
  workflow = """
RLHF DEBUGGING CHECKLIST
==========================

PHASE 1: TRIAGE (10 minutes)
----------------------------
[ ] Look at the three key metrics over time:
- Reward: Is it increasing, flat, or decreasing?
- KL: Is it stable, growing, or oscillating?
- Entropy: Is it decreasing slowly or collapsing?

[ ] Match the signature to a failure mode from the diagnosis guide
- Use the flowchart to narrow down the problem

[ ] Initial hypothesis: Which component is broken?
- Policy learning rate?
- KL penalty (beta)?
- Reward model?
- Something else?

PHASE 2: GENERATION AUDIT (15 minutes)
--------------------------------------
[ ] Sample 20 generations from current policy
- Are they coherent? (If incoherent -> mode collapse or bad LR)
- Are they diverse? (If repetitive -> entropy collapsed)
```

## Early Stopping: When to Shut It Down[#](#early-stopping-when-to-shut-it-down)

Early stopping is critical in RLHF. Most teams run training too long and destroy a good model chasing marginal gains.

early\_stopping.pycpu-only

```
import numpy as np

class RLHFEarlyStopper:
  """
  Multi-criterion early stopping for RLHF.
  Prevents catastrophic failures and overfitting.
  """

  def __init__(
      self,
      patience_steps: int = 10,
      min_reward_improvement: float = 0.05,
      max_kl: float = 20.0,
      min_entropy: float = 0.5,
      kl_patience: int = 3,
      human_eval_interval: int = 50,
  ):
      self.patience_steps = patience_steps
      self.min_reward_improvement = min_reward_improvement
      self.max_kl = max_kl
      self.min_entropy = min_entropy
      self.kl_patience = kl_patience
      self.human_eval_interval = human_eval_interval

      self.best_reward = -float('inf')
      self.no_improvement_count = 0
      self.kl_above_max_count = 0
      self.best_checkpoint = None

  def should_stop(self, metrics: dict, step: int) -> tuple:
```

### When to Use Each Stopping Criterion[#](#when-to-use-each-stopping-criterion)

| Criterion | When to Use | Risk if Ignored |
| --- | --- | --- |
| **KL explosion** | Always (hard stop) | Model diverges completely; unrecoverable |
| **Entropy collapse** | Always (hard stop) | Mode collapse; single boring output forever |
| **No improvement (N steps)** | Always (soft stop) | Overfitting; final checkpoint worse than intermediate |
| **Human evaluation** | Every 50-100 steps | Deploying a model that looks good on metrics but sucks in practice |

## Hands-On: Hyperparameter Search Strategy[#](#hands-on-hyperparameter-search-strategy)

Most teams do random hyperparameter search and waste compute. Here's a *systematic* approach:

hyperparam\_search.pycpu-only

```
import numpy as np
from itertools import product

def simulate_rlhf(lr, beta, batch_size, steps=20):
  """
  Simplified but realistic RLHF simulation.
  This captures the key dynamics:
  - Reward increases with learning rate (up to a limit)
  - KL increases with LR, decreased with beta
  - Larger batches -> more stable (lower variance)
  """
  np.random.seed(hash((lr, beta, batch_size)) % (2**31))

  reward = 1.0
  kl = 0.5
  entropy = 3.5

  for step in range(steps):
      # Reward increases from learning
      learning_signal = lr * 100 * np.exp(-reward / 3)  # Diminishing returns
      reward_gain = learning_signal * (1 - kl / 20)  # Damped by KL penalty

      # KL increases with learning, damped by beta
      kl_gain = lr * 50 - beta * kl * 0.1

      # Entropy decreases slowly with training
      entropy_decay = 1 - 0.02 * step

      # Batch size reduces variance in estimates
      noise_scale = 0.5 / np.sqrt(batch_size / 256)
```

### Search Strategy Template[#](#search-strategy-template)

**Step 1: Learning Rate Sweep**

* Try: 1e-6, 5e-6, 1e-5, 5e-5
* Fix β=0.1, batch=256
* Find: Which LRs don't cause KL explosion or collapse?

**Step 2: Beta Sweep**

* Use best LR from step 1
* Try: 0.01, 0.05, 0.1, 0.2, 0.5
* Measure: Which β keeps KL near target (5-8)?

**Step 3: Batch Size Sweep**

* Use best LR and β
* Try: 64, 128, 256, 512
* Measure: Which batch gives most stable training?

**Step 4: Validate**

* Run 50-100 steps with chosen hyperparameters
* Check: Are metrics stable? Do humans prefer generations?
* Save checkpoint at best point (not final)

## Production Reality: What Actually Happens[#](#production-reality-what-actually-happens)

**Real RLHF training runs (from Meta, Anthropic, OpenAI):**

* **Duration:** 1-3 weeks to tune hyperparameters from scratch
* **Compute:** 5-20 A100 GPUs per run
* **Team:** 1-2 ML engineers + 1-2 policy experts
* **Checkpoints:** Save every 10 steps; best checkpoint often step 30-50, not final
* **Monitoring:** Dashboards with 20+ metrics; alerts on KL, entropy, reward anomalies
* **Failure rate:** ~30% of runs hit a failure mode; require debugging + restart
* **Human evaluation:** Every 50-100 steps; final decision based on human preference, not metrics

**Common gotchas from production RLHF:**

1. **Hyperparameters from paper don't transfer**

   * Different RM → different safe β range
   * Different model size → different safe LR
   * Different task → different KL target
   * You MUST search your own hyperparameters
2. **Metrics look good, humans hate outputs**

   * Reward hacking: RM thinks it's great, humans disagree
   * Unfaithful improvement: Policy learned quirks, not real capability
   * **Solution:** Human evaluation is the source of truth
3. **Reward model is often the bottleneck**

   * Bad RM → policy learns to exploit it
   * Weak RM signal → slow learning
   * **Solution:** Invest heavily in RM training; test for exploits
4. **Final checkpoint is often worse than intermediate**

   * Overfitting/reward hacking accumulates over time
   * **Solution:** Track best reward and use that checkpoint
5. **Communication costs dominate at multi-node scale**

   * Generation needs to stay close to reference model (kl penalty)
   * But reference model requires full RM forward passes
   * **Solution:** Use smaller RM, or cache reference model logits

## Scale Thought Experiment: What Breaks When You Scale?[#](#scale-thought-experiment-what-breaks-when-you-scale)

```
graph TB
    subgraph S1["At 7B (1 A100)"]
        S1A["Quick iteration<br/>30 min per run<br/>Easy debugging"]
    end

    subgraph S2["At 70B (8 A100s)"]
        S2A["3-4 hours per run<br/>Each mistake costs GPU time<br/>Must get HPO right first"]
    end

    subgraph S3["At 400B (multi-node)"]
        S3A["8+ hours per run<br/>Synchronization overhead<br/>Generation bottleneck<br/>KL computation expensive"]
    end

    S1 -->|"10x compute<br/>Fancier HPO"| S2
    S2 -->|"10x compute<br/>Communication costs<br/>Architecture changes"| S3

    S2B["New challenges:<br/>• FSDP synchronization<br/>• Multi-GPU generation<br/>• RM scaling<br/>• Checkpoint management"]
    S3B["New challenges:<br/>• Network latency<br/>• Gradient compression<br/>• Reference model<br/>  distributed forward<br/>• Monitoring latency"]

    S2 -.-> S2B
    S3 -.-> S3B

    style S1 fill:#ccffcc
    style S2 fill:#ffffcc
    style S3 fill:#ffcccc
```

| Aspect | 7B | 70B | 400B+ |
| --- | --- | --- | --- |
| **Per-run cost** | $50 | $500 | $5,000 |
| **Hyperparameter search** | Fast (30 min/run) | Slow (4 hrs/run) | Very slow (8+ hrs/run) |
| **Batch size** | 256-512 | 512-1024 | 2048+ (multi-node) |
| **Stability** | Easy to debug | Medium difficulty | Hard (distributed issues) |
| **RM bottleneck** | Negligible | Measurable (10-20%) | Major (30-50% cost) |
| **Typical iterations** | 100-200 | 20-50 | 5-20 |

**Key insight:** At scale, you can't afford trial-and-error. HPO strategy becomes critical. This is why large labs use curriculum learning or simpler loss functions (like DPO).

## Common Debugging Scenarios[#](#common-debugging-scenarios)

### Scenario 1: "Reward increases but humans hate it"[#](#scenario-1-reward-increases-but-humans-hate-it)

```
Diagnosis: Reward hacking
Metrics: Reward ↑↑↑, KL ↑↑, Entropy stable
Human eval: Reject
Fix: Increase β by 5x, retrain RM
```

### Scenario 2: "Nothing is learning"[#](#scenario-2-nothing-is-learning)

```
Diagnosis: Learning rate too low
Metrics: Reward flat, KL ≈ 0, Entropy stable
Gradient norms: < 1e-8
Fix: Increase LR by 10x, verify RM works
```

### Scenario 3: "Training starts good, then explodes"[#](#scenario-3-training-starts-good-then-explodes)

```
Diagnosis: KL explosion starting at step N
Metrics: Reward ↑, KL stable then ↑↑↑, Entropy ↓
Policy logits: Diverging from reference after N steps
Fix: Lower LR by 5x, increase initial β by 10x
```

### Scenario 4: "All generations are identical"[#](#scenario-4-all-generations-are-identical)

```
Diagnosis: Mode collapse
Metrics: Entropy collapses < 0.5, max_prob > 0.95
Sampling: All 20 samples nearly identical
Fix: Lower LR by 3x, remove temperature annealing
```

## Key Takeaways[#](#key-takeaways)

1. **RLHF is fundamentally a three-variable balancing act**: Policy learning rate, KL penalty, batch size. Change one and you affect others.
2. **Monitor obsessively**: Reward, KL, entropy, value loss, clip fraction. Missing one hides the real problem.
3. **Diagnose before fixing**: Use the failure mode signatures. Triage → hypothesis → targeted ablation.
4. **Humans are the ground truth**: Your reward model is imperfect. Validate with human eval, not just metrics.
5. **Best checkpoint ≠ final checkpoint**: Track peak reward and use that. Final often shows overfitting/hacking.
6. **Search hyperparameters systematically**: LR first (biggest effect), then β, then batch. Not random.
7. **Early stopping saves GPU**: Train until plateau, not to convergence. Overfitting is real.

## Triage Walkthrough[#](#triage-walkthrough)

When a run drifts, use this triage sequence before changing many knobs:

1. **Classify the failure mode from signatures first.**
   Distinguish KL explosion, reward hacking, and collapse before touching hyperparameters.
2. **Change one control variable at a time.**
   Start with learning rate or `beta`, not both, to preserve causal interpretability of the fix.
3. **Run short confirmation windows.**
   Validate improvements on 20 to 50 steps before committing expensive long runs.
4. **Re-check with human eval slices quickly.**
   A metric-only recovery can still regress user preference quality.
5. **Keep best-checkpoint promotion automated.**
   Do not rely on final-step checkpoints in RLHF workflows.

## Checkpoint Questions[#](#checkpoint-questions)

1. A 13B model is RLHF-trained with policy\_lr=5e-6, beta=0.1, batch 256. After 20 steps KL has risen from 2.0 to 18.0 and is climbing. Estimate whether reducing LR to 1e-6 or increasing beta to 0.5 is the better first intervention. Which metric do you check after 10 steps to confirm?
2. Dashboard shows: reward +40% in 15 steps, KL steady at 5.0, entropy stable at 2.8, but human evaluators rate outputs lower than the reference. Which failure mode is this, and what is the first fix?
3. You run RLHF on a 70B model (8 A100s, 15 min/iteration). Estimate total GPU-hours for a 3-stage hyperparameter search: 5 LRs, 5 betas, 4 batch sizes, 30 steps each.

## Research Hooks[#](#research-hooks)

**Automated hyperparameter tuning for RLHF:**
Can we learn to automatically adjust β and LR based on KL and reward? Several papers (e.g., adaptive β algorithms) tackle this. When is automatic better than manual monitoring?

**Reward model robustness:**
How do we build RMs that resist exploitation while remaining predictive? Ensemble RMs, adversarial training, and causal reward models are active areas.

**Simpler alternatives to RLHF:**
DPO and other methods eliminate PPO entirely. When is DPO sufficient and when do you need full RLHF?

**Scaling RLHF to larger models:**
How do you do RLHF on 400B+ models with communication constraints? Curriculum learning, sparse updates, and approximate KL penalties are being explored.

---

*Next up: DPO analytically integrates out the reward model, converting RLHF into a supervised learning problem. One equation replaces the entire RL pipeline.*