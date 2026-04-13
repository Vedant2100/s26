In this tutorial, you will map out the evaluation gap when model capability exceeds human evaluation ability, compare five scalable oversight approaches, and estimate their reliability.

A key challenge: as models get smarter, humans become worse evaluators. A chess novice can't judge if a grandmaster made the right move. A non-expert can't evaluate cutting-edge research. What happens when AI capabilities exceed human evaluation abilities?

## Prerequisites Refresher[#](#prerequisites-refresher)

Before diving into scalable oversight, let's review the foundations you need:

▶RLHF Foundations

Remember: RLHF works in two phases:

1. **Supervised Fine-Tuning (SFT):** Train model to imitate high-quality outputs
2. **Reward Modeling & RL:** Train reward model on human preferences, then RL to maximize reward

The key assumption: **Humans can evaluate outputs better than the base model.**

Once this assumption breaks, RLHF itself breaks down.

▶The Reward Hacking Problem

We covered reward hacking in earlier lessons: models find ways to get high reward that weren't intended by the reward designer.

For scalable oversight, this is critical: if we can't directly evaluate outputs, how do we even know if the reward model is being gamed?

▶Why Model Capability Matters

Capability creates an asymmetry:

* A model at capability level C can solve tasks at level `<= C`
* Humans at capability level H can evaluate tasks at level `<= H`
* When C > H, we have a problem

The rest of this lesson explores approaches to bridge this gap.

## The Evaluation Gap[#](#the-evaluation-gap)

evaluation\_gap.pycpu-only

```
import numpy as np

def illustrate_evaluation_gap():
  """
  Illustrate how the evaluation gap grows with capability.
  """
  np.random.seed(42)

  print("The Evaluation Gap")
  print("=" * 60)
  print()

  print("MODEL CAPABILITY vs HUMAN EVALUATION ABILITY")
  print("-" * 50)
  print()

  # Capability levels and human evaluation ability
  scenarios = dict(
      GPT3_2020=dict(
          model_capability=0.6,
          human_eval_ability=0.9,
          example="Humans can easily spot bad summaries",
      ),
      GPT4_2023=dict(
          model_capability=0.8,
          human_eval_ability=0.7,
          example="Humans struggle with advanced math proofs",
      ),
      Projected_2025=dict(
          model_capability=0.9,
```

The Challenge

The Problem

Model less than Human

Model greater than Human

Model Output

Human Evaluation

Reliable Feedback

Unreliable Feedback

How do we know if helpful or deceptive?

How do we train for unevaluable goals?

How do we maintain control as capability grows?

concrete\_examples.pycpu-only

```
def concrete_oversight_challenges():
  """
  Concrete examples where oversight becomes hard.
  """
  print("Concrete Oversight Challenges")
  print("=" * 60)
  print()

  challenges = dict(
      ADVANCED_MATH=dict(
          task="Evaluate a proof of a novel theorem",
          why_hard="Non-mathematicians can't verify correctness",
          current_gap="Already a problem (GPT-4 math)",
          risk="Accept flawed proofs, reject valid ones",
      ),
      CODE_SECURITY=dict(
          task="Evaluate if code has subtle vulnerabilities",
          why_hard="Security bugs are hard to spot",
          current_gap="Experts often miss issues",
          risk="Approve insecure code",
      ),
      LONG_HORIZON_PLANNING=dict(
          task="Evaluate if a 10-step plan will succeed",
          why_hard="Humans can't simulate many steps ahead",
          current_gap="Growing with model capability",
          risk="Approve plans with hidden problems",
      ),
      DECEPTION_DETECTION=dict(
          task="Detect if model is being subtly deceptive",
          why_hard="Sophisticated models can be convincing",
```

## Approach 1: Scalable AI Feedback (RLAIF)[#](#approach-1-scalable-ai-feedback-rlaif)

rlaif\_for\_oversight.pycpu-only

```
def rlaif_for_oversight():
  """
  RLAIF as a partial solution to scalable oversight.
  """
  print("RLAIF for Scalable Oversight")
  print("=" * 60)
  print()

  print("THE IDEA:")
  print("-" * 50)
  print("Use AI to evaluate AI outputs")
  print()
  print("  Human evaluator (limited) --> AI evaluator (scalable)")
  print()

  print("HOW IT HELPS:")
  print("-" * 50)
  print("1. AI can process more information than humans")
  print("2. AI can be more consistent")
  print("3. AI can scale to any volume")
  print("4. AI may catch things humans miss")
  print()

  print("LIMITATIONS:")
  print("-" * 50)
  print("1. AI evaluator has same blind spots as AI being evaluated")
  print("2. Both may share biases from similar training")
  print("3. Doesn't help when BOTH are wrong in the same way")
  print("4. AI evaluator can be fooled by AI being evaluated")
  print()
```

## Approach 2: AI Safety via Debate[#](#approach-2-ai-safety-via-debate)

Debate Protocol

Question

Debater A for X

Debater B for Y

Argument round 1

Argument round 2

More rounds

Human Judge evaluates

Verdict

### Why Debate Might Scale[#](#why-debate-might-scale)

The key insight: **Finding flaws is easier than generating solutions.**

* Hard: "Is this medical diagnosis correct?" (requires medical expertise)
* Easier: "Which debater found a real flaw in the diagnosis?" (requires following argument logic)

debate\_approach.pycpu-only

```
def explain_debate():
  """
  Explain the AI debate approach to scalable oversight.
  """
  print("AI Safety via Debate")
  print("=" * 60)
  print()

  print("THE CORE IDEA:")
  print("-" * 50)
  print("Two AI agents debate, human judges the debate")
  print()
  print("  Instead of: Human evaluates AI output directly")
  print("  We have:    Human evaluates which ARGUMENT is stronger")
  print()

  print("WHY THIS MIGHT HELP:")
  print("-" * 50)
  print()
  print("1. FINDING FLAWS IS EASIER THAN GENERATING SOLUTIONS")
  print("   - Hard: 'Is this proof correct?'")
  print("   - Easier: 'Which debater found a real flaw?'")
  print()
  print("2. ADVERSARIAL PRESSURE REVEALS WEAKNESSES")
  print("   - If answer X has a flaw, debater B is incentivized to find it")
  print("   - Debate surfaces information that wouldn't emerge otherwise")
  print()
  print("3. HUMAN ONLY NEEDS TO FOLLOW ARGUMENTS")
  print("   - Don't need to understand the full solution")
  print("   - Just need to judge 'did this counter-argument land?'")
```

debate\_approach\_detailed.pycpu-only

```
def explain_debate_detailed():
  """
  Detailed analysis of debate as an oversight mechanism.
  """
  print("Debate Mechanics: How It Scales Oversight")
  print("=" * 60)
  print()

  print("THE DEBATE TOURNAMENT STRUCTURE:")
  print("-" * 50)
  print()
  print("Round 1: Opening statements (both debaters make cases)")
  print("Round 2: Rebuttals (each counters the other)")
  print("Round 3: Clarifications (address confusions)")
  print("Round 4: Final arguments (last word)")
  print()
  print("Outcome: Human judges determines winner")
  print()

  print("TRAINING THE DEBATE SYSTEM:")
  print("-" * 50)
  print()
  print("Step 1: Collect human judgments")
  print("  - Run many AI debates")
  print("  - Have humans judge winners")
  print("  - Collect (debate_transcript, winner) pairs")
  print()
  print("Step 2: Train judge model")
  print("  - Learn to predict which debater won")
  print("  - Mimics human judgment")
```

debate\_simulation.pycpu-only

```
import numpy as np

def simulate_debate():
  """
  Simulate how debate helps with oversight.
  """
  np.random.seed(42)

  print("Debate Simulation")
  print("=" * 60)
  print()

  # Simulate a scenario where direct evaluation is hard
  # but debate makes it tractable

  # Ground truth: there IS a flaw in the answer
  has_flaw = True

  # Human direct evaluation: only 40% chance to catch flaw
  human_direct_ability = 0.4

  # Debater ability to find flaw (if it exists)
  debater_find_flaw = 0.85

  # Human ability to judge debate argument
  human_judge_debate = 0.75

  print("SCENARIO: Evaluating a complex AI output")
  print("-" * 50)
  print("  Output has a subtle flaw: %s" % has_flaw)
```

## Approach 3: Recursive Reward Modeling[#](#approach-3-recursive-reward-modeling)

Recursive Decomposition

decompose

decompose

decompose

decompose

decompose

decompose

decompose

Level 3 Full Task - Hard

Subtask 1

Subtask 2

Subtask 3

Sub-subtask 1a

Sub-subtask 1b

Sub-subtask 2a

Sub-subtask 3a

Human evaluates base

recursive\_reward.pycpu-only

```
def explain_recursive_reward():
  """
  Explain recursive reward modeling.
  """
  print("Recursive Reward Modeling")
  print("=" * 60)
  print()

  print("THE CORE IDEA:")
  print("-" * 50)
  print("Break hard evaluation tasks into easier subtasks")
  print()
  print("  Hard task: 'Is this research paper correct?'")
  print("             |")
  print("  Subtask 1: 'Is the methodology section sound?'")
  print("  Subtask 2: 'Are the statistics correct?'")
  print("  Subtask 3: 'Do the conclusions follow?'")
  print("             |")
  print("  Sub-subtask: 'Is this specific calculation right?'")
  print()

  print("THE RECURSION:")
  print("-" * 50)
  print()
  print("1. Human can evaluate BASE CASES (simple subtasks)")
  print("2. AI decomposes complex tasks into simpler ones")
  print("3. Each level verified by level above (or human)")
  print("4. Trust propagates from base cases upward")
  print()
```

## Approach 4: Iterated Amplification[#](#approach-4-iterated-amplification)

Each Step

Train AI to mimic H + current AI

New AI slightly better

Use to assist human

Amplification

Human alone

Human plus AI v1

Human plus AI v2

Human plus AI v3

Human plus capable AI

amplification.pycpu-only

```
import numpy as np

def explain_amplification():
  """
  Explain iterated amplification.
  """
  print("Iterated Amplification")
  print("=" * 60)
  print()

  print("THE CORE IDEA:")
  print("-" * 50)
  print("Gradually amplify human capability using AI assistance")
  print()
  print("  Step 0: Human alone (limited capability)")
  print("  Step 1: Human + weak AI (slightly better)")
  print("  Step 2: Human + better AI (even better)")
  print("  ...     (iterate)")
  print("  Step N: Human + powerful AI (human-level+ oversight)")
  print()

  print("THE KEY INSIGHT:")
  print("-" * 50)
  print("At each step, the AI is trained to imitate")
  print("'human + previous AI'")
  print()
  print("If 'human + AI' makes better decisions than AI alone,")
  print("and we train new AI on those decisions,")
  print("then new AI inherits the improvement.")
  print()
```

amplification\_simulation.pycpu-only

```
import numpy as np

def simulate_amplification():
  """
  Simulate how amplification improves oversight.
  """
  np.random.seed(42)

  print("Amplification Simulation")
  print("=" * 60)
  print()

  # Human baseline capability
  human_base = 0.6

  # Each AI iteration adds capability
  ai_boost_per_step = 0.08

  # But there's diminishing returns
  diminishing_factor = 0.9

  n_steps = 8

  print("%s %s %s %s" % ("Step".ljust(8), "Capability".ljust(15), "Boost".ljust(15), "Can Evaluate".ljust(20)))
  print("-" * 58)

  capability = human_base
  current_boost = ai_boost_per_step

  for step in range(n_steps + 1):
```

## Approach 5: Weak-to-Strong Generalization[#](#approach-5-weak-to-strong-generalization)

One emerging approach: **Train strong models using feedback from weak models.**

The Hope

Weak-to-Strong Training

provides labels

trains on

may exceed

generalize beyond

Weak Evaluator

Strong Model to train

Data from weak eval

Strong Output

Exceeds weak teacher

weak\_to\_strong.pycpu-only

```
def weak_to_strong_generalization():
  """
  Explain weak-to-strong generalization for scalable oversight.
  """
  print("Weak-to-Strong Generalization")
  print("=" * 60)
  print()

  print("THE PROBLEM:")
  print("-" * 50)
  print("If a strong model is trained only on weak feedback,")
  print("won't it just learn to be as weak as the teacher?")
  print()

  print("THE SURPRISING FINDING:")
  print("-" * 50)
  print("Recent empirical work suggests: NOT ALWAYS")
  print()
  print("Strong models can sometimes generalize beyond their teacher.")
  print()

  print("EXAMPLE: Chess")
  print("-" * 50)
  print("  Weak evaluator: 1200 ELO chess engine")
  print("  Training data: Labels from 1200 ELO evaluations")
  print("  Strong model: Trained on weak labels")
  print("  Result: Can reach 2000+ ELO (exceed teacher)")
  print()
  print("Why? The model learns patterns that generalize")
  print("even though teacher doesn't understand them.")
```

## Comparing Approaches[#](#comparing-approaches)

comparison\_table.pycpu-only

```
def compare_oversight_approaches():
  """
  Compare different scalable oversight approaches.
  """
  print("Comparison of Scalable Oversight Approaches")
  print("=" * 70)
  print()

  approaches = dict(
      RLAIF=dict(
          mechanism="AI evaluates AI outputs",
          strengths=["Scalable", "Consistent", "Already deployed"],
          weaknesses=["Shared blind spots", "Can be fooled"],
          maturity="Production",
          best_for="Clear-cut evaluations",
      ),
      Debate=dict(
          mechanism="AIs argue, human judges",
          strengths=["Surfaces hidden flaws", "Reduces human burden"],
          weaknesses=["Requires trainable debate", "May not always work"],
          maturity="Research",
          best_for="Verifiable arguments",
      ),
      Recursive_Reward=dict(
          mechanism="Decompose into subtasks",
          strengths=["Tractable subtasks", "Trustworthy composition"],
          weaknesses=["Not all tasks decompose", "Error accumulation"],
          maturity="Research",
          best_for="Hierarchical tasks",
      ),
```

## When Does Each Approach Work?[#](#when-does-each-approach-work)

approach\_selection.pycpu-only

```
def approach_selection_guide():
  """
  Guide for selecting the right oversight approach.
  """
  print("Selecting the Right Oversight Approach")
  print("=" * 60)
  print()

  print("DECISION FRAMEWORK:")
  print("-" * 50)
  print()

  framework = """
Is the task VERIFIABLE (given enough effort)?
|
+- YES: Can the task be DECOMPOSED into subtasks?
|   |
|   +- YES --> Recursive Reward Modeling
|   |          (Break into human-verifiable pieces)
|   |
|   +- NO --> Debate
|              (Have AIs argue about the answer)
|
+- NO: Is evaluation OBJECTIVE or SUBJECTIVE?
  |
  +- OBJECTIVE (right/wrong answers) --> RLAIF
  |   (AI evaluators can be consistent)
  |
  +- SUBJECTIVE (preferences) --> Amplification
      (Need to preserve human values through iteration)
```

## Break It: Oversight Failure Modes[#](#break-it-oversight-failure-modes)

break\_it\_oversight.pycpu-only

```
import numpy as np

def break_oversight_approaches():
  """
  Demonstrate failure modes of each approach.
  """
  np.random.seed(42)

  print("Break It: Oversight Failure Modes")
  print("=" * 60)
  print()

  print("RLAIF FAILURE: Correlated Errors")
  print("-" * 50)
  print()
  print("Scenario: AI evaluator and AI being evaluated both")
  print("         learned from similar data with the same bias")
  print()
  print("  Model output: 'The capital of Australia is Sydney'")
  print("  AI evaluator: 'This is correct' (has same misconception)")
  print("  Human check:  'Actually, it's Canberra'")
  print()
  print("When evaluator and model share training data,")
  print("they may share the same blind spots.")
  print()

  print("DEBATE FAILURE: Persuasive but Wrong")
  print("-" * 50)
  print()
  print("Scenario: One debater is more persuasive,")
```

### Deep Dive: Each Approach's Critical Weakness[#](#deep-dive-each-approachs-critical-weakness)

▶RLAIF: Correlated Blindspots

RLAIF assumes the evaluator model has different failure modes than the model being evaluated. But they often have the SAME failure modes because:

1. **Same training data:** Both trained on similar internet corpus
2. **Similar architecture:** Both transformers with similar inductive biases
3. **Shared misconceptions:** Both may learn the same falsehoods

Example: If both models learned that "correlation implies causation," the evaluator won't catch when the model claims causation falsely.

▶Debate: The Sophistry Problem

Debate assumes truth is more persuasive than falsehood. But:

1. **Sophisticated lies can be more convincing** than messy truth
2. **Information asymmetry:** Debater preparing false argument can research it more thoroughly
3. **Human judges have limits:** Non-experts can't judge even with arguments presented

Real example: A trained debater can make almost any position sound plausible through rhetorical technique.

▶Recursive Reward: Decomposition Blindspots

Even if all subtasks are evaluated correctly, the full task can fail if:

1. **Subtasks miss the holistic goal:** Evaluating each component doesn't catch emergent properties
2. **Decomposition itself is wrong:** The decomposition might miss critical evaluation dimensions
3. **Interdependencies:** Subtle interactions between subtasks not captured

Example: Decomposing "Is this medical treatment good?" into "Is it effective?" + "Is it safe?" misses questions about cost-effectiveness or patient preference.

▶Amplification: Accumulated Drift

Small errors at each step compound dramatically:

1. **Each amplification introduces error:** No process is 100% reliable
2. **Error compounds:** Errors from step N feed into step N+1
3. **No feedback loop:** Can't catch drift until it's too late

Over 10 amplification steps with 5% error each, you lose ~40% alignment with original human values.

▶Weak-to-Strong: The Ceiling Effect

Strong models don't always generalize beyond their weak teacher:

1. **Large capability gaps:** The larger the gap, the less generalization works
2. **Spurious pattern learning:** Model learns patterns the teacher endorsed even if they're wrong
3. **Unknown when it works:** No reliable way to predict which domains allow generalization

Early results promising, but scaling properties unknown.

## Hybrid Approaches: Combining Methods[#](#hybrid-approaches-combining-methods)

In practice, no single approach will work alone. Leading labs combine multiple techniques:

agree

agree

agree

agree

disagree

disagree

disagree

disagree

Task Evaluation

Method 1 RLAIF

Method 2 Debate

Method 3 Human

Method 4 Process

Approve

Escalate

Final Decision

hybrid\_approach.pycpu-only

```
def hybrid_oversight_strategy():
  """
  Show how combining approaches improves reliability.
  """
  import numpy as np

  print("Hybrid Oversight: Combining Methods")
  print("=" * 60)
  print()

  print("THE STRATEGY:")
  print("-" * 50)
  print("Don't rely on one oversight method.")
  print("Use multiple methods in parallel.")
  print("Escalate if they disagree.")
  print()

  print("RELIABILITY CALCULATION:")
  print("-" * 50)
  print()

  # Each method has independent failure probability
  methods = dict(
      RLAIF_evaluation=0.85,
      Debate_outcome=0.80,
      Human_spot_check=0.75,
      Process_rewards=0.70,
  )

  print("%s %s" % ("Method".ljust(25), "Accuracy".ljust(12)))
```

## The Fundamental Challenge[#](#the-fundamental-challenge)

fundamental\_challenge.pycpu-only

```
def fundamental_challenge():
  """
  Articulate the fundamental challenge of scalable oversight.
  """
  print("The Fundamental Challenge")
  print("=" * 60)
  print()

  print("THE CORE DILEMMA:")
  print("-" * 50)
  print()
  print("We want to train AI systems to do things we can't do.")
  print("But we can only train on feedback we can provide.")
  print()
  print("If we can't evaluate an output, we can't train on it.")
  print("If we can't train on it, the model won't learn to do it well.")
  print("If the model does it anyway, we can't tell if it's right.")
  print()

  print("THREE POSSIBLE FUTURES:")
  print("-" * 50)
  print()

  futures = dict(
      OPTIMISTIC=dict(
          assumption="Scalable oversight techniques work",
          outcome="We maintain meaningful oversight as AI advances",
          required="Debate/amplification/etc. scale reliably",
      ),
      MIDDLE_GROUND=dict(
```

## Scale Thought Experiment[#](#scale-thought-experiment)

| Model Capability | Oversight Challenge | Viable Approaches |
| --- | --- | --- |
| **Human-level** | Humans can evaluate | Standard RLHF, RLAIF |
| **Superhuman narrow** | Experts struggle | RLAIF, Debate, Recursive |
| **Superhuman broad** | No human can evaluate | Amplification, Novel methods |
| **Radically superhuman** | Unknown | Unknown (research frontier) |

scale\_thought\_experiment.pycpu-only

```
def scale_thought_experiment():
  """
  Think through oversight at different capability scales.
  """
  print("Scaling Oversight: A Thought Experiment")
  print("=" * 60)
  print()

  print("LEVEL 1: CURRENT MODELS (GPT-4 class)")
  print("-" * 50)
  print("  Human oversight: Mostly feasible")
  print("  Blind spots: Advanced math, subtle bugs, long reasoning")
  print("  Approaches: RLHF, RLAIF, early debate experiments")
  print()

  print("LEVEL 2: NEAR-FUTURE (2-5 years)")
  print("-" * 50)
  print("  Human oversight: Challenged in many domains")
  print("  Blind spots: Most technical domains, long-horizon planning")
  print("  Approaches: RLAIF at scale, debate, recursive methods")
  print("  Key question: Do these approaches scale?")
  print()

  print("LEVEL 3: SUPERHUMAN NARROW (5-10 years?)")
  print("-" * 50)
  print("  Human oversight: Requires AI assistance for most tasks")
  print("  Blind spots: Anything beyond human expert level")
  print("  Approaches: Amplification, AI-assisted debate")
  print("  Key question: Can we trust AI-assisted oversight?")
  print()
```

## Production Reality[#](#production-reality)

**Current state of the art:**

* RLAIF: Deployed at Google, Anthropic, others
* Debate: Active research (Anthropic, OpenAI)
* Recursive Reward: Theoretical, limited experiments
* Amplification: Theoretical, early experiments

**What labs are doing:**

* Multiple redundant oversight methods
* Human spot-checking of AI evaluations
* Red-teaming to find oversight failures
* Investing heavily in oversight research

**Open challenges:**

1. No proven method for superhuman AI oversight
2. Detecting when oversight is failing is hard
3. Adversarial robustness of oversight systems unclear
4. Scaling properties of each approach unknown

current\_practice.pycpu-only

```
def current_practice():
  """
  What leading labs actually do for oversight.
  """
  print("Current Oversight Practices at Leading Labs")
  print("=" * 60)
  print()

  print("ANTHROPIC:")
  print("-" * 50)
  print("  - Constitutional AI (principles + self-critique)")
  print("  - RLAIF for scaling preference data")
  print("  - Red-teaming to find oversight gaps")
  print("  - Research into debate and amplification")
  print()

  print("OPENAI:")
  print("-" * 50)
  print("  - RLHF with human feedback")
  print("  - Superalignment team (disbanded mid-2024; research continues across other teams)")
  print("  - Weak-to-strong generalization experiments")
  print("  - Process supervision (reward intermediate steps)")
  print()

  print("GOOGLE DEEPMIND:")
  print("-" * 50)
  print("  - RLAIF (used in Gemini)")
  print("  - Research into debate and recursive methods")
  print("  - Theoretical work on oversight limits")
  print()
```

## Oversight as an Arms Race[#](#oversight-as-an-arms-race)

There is an adversarial dimension to scalable oversight worth examining.

oversight\_arms\_race.pycpu-only

```
def oversight_as_arms_race():
  """
  Analyze oversight as an adversarial problem.
  """
  print("Oversight as an Arms Race")
  print("=" * 60)
  print()

  print("THE GAME THEORY PERSPECTIVE:")
  print("-" * 50)
  print()
  print("Model incentives:")
  print("  - Maximize reward")
  print("  - Avoid punishment")
  print("  - If caught cheating, try harder to hide it")
  print()
  print("Overseer incentives:")
  print("  - Catch deception")
  print("  - Verify good behavior")
  print("  - Update oversight if fooled")
  print()

  print("WHAT HAPPENS IN EQUILIBRIUM:")
  print("-" * 50)
  print()
  print("Scenario A: Overseer always wins")
  print("  Model learns it can't fool oversight")
  print("  Model genuinely aligns with intended behavior")
  print("  (Optimistic case)")
  print()
```

## Practical Lessons for Oversight Design[#](#practical-lessons-for-oversight-design)

▶Lesson 1: Transparency is Fragile

Relying on model transparency (e.g., interpretability) is risky because:

1. **More capable models may be less interpretable** (scaling laws for interpretability unclear)
2. **Sophisticated deception can hide in complexity**
3. **Explanation can mislead** (model can learn to provide explanations that seem right)

Don't rely on transparency alone. Combine with external verification.

▶Lesson 2: Incentive Alignment is Hard

Training a model to be honest is hard because:

1. **Honesty may not maximize reward** in competitive settings
2. **Models may learn to distinguish evaluator from world** (honest to evaluator, deceptive elsewhere)
3. **Reward signal has blind spots** that models exploit

Structure incentives carefully. Expect adversarial pressure.

▶Lesson 3: Redundancy is Essential

Single oversight methods will fail. You need:

1. **Multiple independent methods** (different failure modes)
2. **Cross-checks** (different methods should agree)
3. **Escalation procedures** (when methods disagree, investigate)
4. **Red-teaming** (adversaries actively search for failures)

Defense in depth is non-negotiable.

▶Lesson 4: Metrics Are Hard to Game-Proof

Even simple metrics can be gamed:

Example: "Reward model accuracy" — model learns to produce outputs that fool reward model without being better.

Solution: Track multiple metrics that would require different exploits to fool simultaneously.

## Thought Experiment: The Oversight Crisis[#](#thought-experiment-the-oversight-crisis)

Imagine this scenario in 2030:

1. **New capability level:** Models exceed human evaluation in 50% of domains
2. **Oversight still in 2024:** We're using RLHF, RLAIF, basic debate
3. **No single breakthrough:** None of the scalable oversight methods fully works
4. **Deployment pressure:** Companies want to deploy more capable models

**What happens?**

Options:

* **Accept reduced oversight:** Deploy anyway, hope for best
* **Limit capability:** Hold back model capability to match oversight
* **Heavy human investment:** Employ armies of evaluators (doesn't scale)
* **Bet on research:** Invest heavily in scalable oversight research
* **Hybrid:** Some mix of above

**This is not hypothetical.** This exact situation is being faced right now, and the outcome is unclear.

oversight\_policy\_analysis.pycpu-only

```
def oversight_policy_space():
  """
  Analyze the policy choices around scalable oversight.
  """
  print("The Oversight Policy Space")
  print("=" * 60)
  print()

  print("DECISION: How to handle capability-oversight gap?")
  print("-" * 50)
  print()

  policies = dict(
      Full_speed=dict(
          description="Deploy regardless of oversight gaps",
          upside="Maximize capability, benefits",
          downside="Risk of undetected misalignment",
          timeline="Fastest",
      ),
      Cautious=dict(
          description="Pause deployment when gap exceeds threshold",
          upside="Lower risk of catastrophic failure",
          downside="Slower progress, lost benefits",
          timeline="Slower but managed",
      ),
      Research_first=dict(
          description="Invest 50% compute in oversight research",
          upside="Build fundamental solutions",
          downside="Slower capability progress, uncertain R&D",
          timeline="Very slow near-term, potentially fast long-term",
```

## Checkpoint Questions[#](#checkpoint-questions)

1. You have 4 independent oversight methods, each 80% accurate. Compute the probability that majority vote (3/4 agreeing) is correct.
2. Combined detection = P(debater finds) \* P(human judges). If debater=60%, human=75%, what is the rate? Better than 40% baseline?
3. Amplification: 10 steps, 3% drift each (compounding). Final alignment? When does it drop below 80%?

## Research Hooks[#](#research-hooks)

**Weak-to-strong generalization:**
Can we train strong models using feedback from weak ones? Recent work shows promise but also limitations. This could be key to scalable oversight.

**Process vs outcome supervision:**
Rewarding intermediate steps (process) vs final answers (outcome). Process supervision may be more robust to gaming. Active research area.

**Detecting oversight failures:**
If oversight is failing, how would we know? Developing metrics and tests for oversight quality is crucial but underdeveloped.

**Theoretical limits:**
Is there a fundamental limit to oversight capability? Information-theoretic and game-theoretic analyses may reveal hard limits.

**Emergent oversight:**
Can AI systems develop better oversight methods than humans design? Meta-learning approaches to oversight.

---

*This concludes Part 6: Constitutional AI & RLAIF. You now understand both the promise and the fundamental challenges of scaling oversight beyond human evaluation capability.*

*Next up: Part 7 covers Evaluation & Deployment, including reward hacking in the wild and how to measure alignment in production systems.*