In this tutorial, you will compare RLAIF and RLHF on cost, quality, and throughput, design a hybrid human/AI feedback pipeline with confidence-based routing, and estimate the self-improvement ceiling for iterative RLAIF.

RLAIF replaces human preference labels with AI-generated labels. The practical question is: on which tasks does AI feedback quality come close enough to human feedback, and where do you still need humans in the loop?

## Prerequisites: Quick Refresher[#](#prerequisites-quick-refresher)

▶What is RLHF? (Refresher)

RLHF (Reinforcement Learning from Human Feedback) is the training procedure that powered ChatGPT and modern aligned LLMs:

1. **Generate responses**: Your base model generates multiple responses to a prompt
2. **Human preference labels**: Humans compare pairs and pick the "better" response
3. **Train reward model**: Use these preferences to train a reward model that predicts human preference
4. **PPO training**: Use RL (specifically PPO) to optimize the policy model to maximize reward

The bottleneck is step 2: human annotators are expensive ($0.50-$2.00 per label), slow (weeks to get 100K labels), and limited in scale (max ~1M labels for a major model release).

▶What is a Reward Model? (Refresher)

A reward model is a classifier that learns to predict human preferences:

```
Input: (prompt, response_A, response_B)
Output: score(response_A) vs score(response_B)
        → which one humans prefer?

Training: minimize -log(sigmoid(score(chosen) - score(rejected)))
          (preference classification loss)
```

Once trained, it gives you a scalar reward signal that RL can optimize against. The quality of your reward model directly determines the quality of your aligned model.

▶Why is the Scaling Problem Hard? (Refresher)

The core insight: **All RL methods have a scaling wall at the reward model bottleneck.**

* At 10K labels: Human feedback is fine
* At 100K labels: Getting expensive (~$100K)
* At 1M labels: Prohibitively expensive (~$1M+)
* At 10M labels: Impossible with humans alone

Yet the best models benefit from more feedback. This is where RLAIF enters the picture.

## The RLAIF Pipeline[#](#the-rlaif-pipeline)

RLAIF

Prompts

Generate Responses

AI Annotator  
(with instructions)

Preference Labels

Train Reward Model

PPO Training

Standard RLHF

Prompts

Generate Responses

Human Annotators

Preference Labels

Train Reward Model

PPO Training

## RLHF vs RLAIF: Head-to-Head[#](#rlhf-vs-rlaif-head-to-head)

Hybrid (Recommended)

80% AI / 20% Human  
(Best tradeoff)

Quality  
95-98% of human

Speed  
1-2 weeks

Scale  
1-50M practical

Cost  
80-85% savings

RLAIF

Cost: $0.005/label

AI Evaluator  
(Cheap)

Quality  
90-95% of human

Speed  
Hours to days

Scale  
Unlimited (100M+)

RLHF

Cost: $1/label

Human Annotators  
(Expensive)

Quality  
Benchmark

Speed  
Weeks to months

Scale  
Max 1-5M labels

rlaif\_vs\_rlhf.pycpu-only

```
def compare_rlhf_rlaif():
  """
  Side-by-side comparison of RLHF vs RLAIF.
  """
  print("RLHF vs RLAIF Comparison")
  print("=" * 80)
  print()

  dimensions = {
      "Cost per label": {"RLHF": "$1.00", "RLAIF": "$0.005", "Winner": "RLAIF (200x)"},
      "Quality": {"RLHF": "100%", "RLAIF": "90-95%", "Winner": "RLHF (slight)"},
      "Time for 100K labels": {"RLHF": "2-4 weeks", "RLAIF": "1-2 hours", "Winner": "RLAIF (100x)"},
      "Iteration speed": {"RLHF": "1-2 months/cycle", "RLAIF": "Days/cycle", "Winner": "RLAIF"},
      "Consistency": {"RLHF": "72-85% inter-rater", "RLAIF": "85-95%", "Winner": "RLAIF"},
      "Requires expertise": {"RLHF": "Minimal", "RLAIF": "High (prompt engineering)", "Winner": "RLHF"},
      "Failure modes": {"RLHF": "Bias toward hiring pool", "RLAIF": "Model biases + sycophancy", "Winner": "Tie"},
      "Useful at small scale": {"RLHF": "Yes (10K labels)", "RLAIF": "No (quality matters)", "Winner": "RLHF"},
      "Useful at large scale": {"RLHF": "No (too expensive)", "RLAIF": "Yes (unlimited)", "Winner": "RLAIF"},
  }

  print("%-30s %-20s %-20s %-15s" % ("Dimension", "RLHF", "RLAIF", "Winner"))
  print("-" * 80)

  for dimension, metrics in dimensions.items():
      rlhf = metrics["RLHF"]
      rlaif = metrics["RLAIF"]
      winner = metrics["Winner"]
      print("%-30s %-20s %-20s %-15s" % (dimension, rlhf, rlaif, winner))

  print()
```

rlaif\_pipeline\_overview.pycpu-only

```
def rlaif_pipeline_overview():
  """
  Overview of the RLAIF pipeline.
  """
  print("RLAIF Pipeline")
  print("=" * 60)
  print()

  print("STANDARD RLHF:")
  print("-" * 50)
  print("  1. Generate responses to prompts")
  print("  2. Send pairs to human annotators")
  print("  3. Humans select preferred response")
  print("  4. Train reward model on human preferences")
  print("  5. Run PPO to optimize against reward model")
  print()

  print("RLAIF:")
  print("-" * 50)
  print("  1. Generate responses to prompts")
  print("  2. Send pairs to AI annotator with instructions")
  print("  3. AI selects preferred response")
  print("  4. Train reward model on AI preferences")
  print("  5. Run PPO to optimize against reward model")
  print()

  print("KEY DIFFERENCE: Step 2-3")
  print("-" * 50)
  print()
  print("  Human annotators:  Expensive, slow, limited scale")
```

## AI Feedback Quality: The Evidence[#](#ai-feedback-quality-the-evidence)

quality\_comparison.pycpu-only

```
import numpy as np

def analyze_feedback_quality():
  """
  Compare AI and human feedback quality.
  """
  np.random.seed(42)

  print("AI vs Human Feedback Quality")
  print("=" * 60)
  print()

  # Based on published research (Google RLAIF paper, etc.)
  tasks = {
      "Summarization": {
          "human_agreement": 0.78,
          "ai_agreement_with_human": 0.73,
          "ai_self_consistency": 0.85,
      },
      "Helpfulness": {
          "human_agreement": 0.72,
          "ai_agreement_with_human": 0.68,
          "ai_self_consistency": 0.82,
      },
      "Harmlessness": {
          "human_agreement": 0.85,
          "ai_agreement_with_human": 0.81,
          "ai_self_consistency": 0.90,
      },
      "Code Quality": {
```

quality\_by\_difficulty.pycpu-only

```
import numpy as np

def quality_by_difficulty():
  """
  How AI feedback quality varies with task difficulty.
  """
  np.random.seed(42)

  print("AI Feedback Quality by Task Difficulty")
  print("=" * 60)
  print()

  # Simulated data based on research patterns
  difficulties = ["Easy", "Medium", "Hard", "Ambiguous"]

  results = {
      "Easy": {
          "example": "Which response is grammatically correct?",
          "human_accuracy": 0.95,
          "ai_accuracy": 0.97,
      },
      "Medium": {
          "example": "Which response is more helpful for the task?",
          "human_accuracy": 0.80,
          "ai_accuracy": 0.75,
      },
      "Hard": {
          "example": "Which response shows better reasoning?",
          "human_accuracy": 0.70,
          "ai_accuracy": 0.60,
```

## Where AI Feedback Excels[#](#where-ai-feedback-excels)

ai\_excels.pycpu-only

```
def where_ai_excels():
  """
  Tasks where AI feedback is particularly effective.
  """
  print("Where AI Feedback Excels")
  print("=" * 60)
  print()

  excels = {
      "CONSISTENCY": {
          "description": "AI gives same answer for same comparison",
          "human_problem": "Humans disagree with themselves 15-20%",
          "ai_advantage": "AI self-consistency often >90%",
          "example": "Formatting preferences, style guidelines",
      },
      "VERIFIABLE FACTS": {
          "description": "Objectively correct vs incorrect",
          "human_problem": "Humans may not know the facts",
          "ai_advantage": "AI has broad knowledge base",
          "example": "Is this historical date correct?",
      },
      "CODE EVALUATION": {
          "description": "Code correctness, efficiency",
          "human_problem": "Requires expertise, easy to miss bugs",
          "ai_advantage": "Can reason about code execution",
          "example": "Which implementation is more efficient?",
      },
      "SCALE": {
          "description": "Large volumes of comparisons",
          "human_problem": "Expensive, slow, fatigue effects",
```

## Where AI Feedback Fails[#](#where-ai-feedback-fails)

ai\_fails.pycpu-only

```
def where_ai_fails():
  """
  Tasks where AI feedback is problematic.
  """
  print("Where AI Feedback Fails")
  print("=" * 60)
  print()

  fails = {
      "CULTURAL NUANCE": {
          "description": "Understanding cultural context",
          "why_ai_fails": "Training data biases, limited cultural exposure",
          "human_advantage": "Direct cultural knowledge and intuition",
          "example": "Is this joke offensive in this culture?",
      },
      "EMOTIONAL INTELLIGENCE": {
          "description": "Empathy, tone, emotional support",
          "why_ai_fails": "AI doesn't experience emotions",
          "human_advantage": "Genuine understanding of feelings",
          "example": "Which response is more comforting?",
      },
      "NOVEL SITUATIONS": {
          "description": "Scenarios not in training data",
          "why_ai_fails": "Limited to patterns seen in training",
          "human_advantage": "Common sense, real-world experience",
          "example": "How to handle unprecedented event X?",
      },
      "LONG-TERM IMPACT": {
          "description": "Consequences over time",
          "why_ai_fails": "Optimizes for immediate response quality",
```

## Designing Hybrid Pipelines[#](#designing-hybrid-pipelines)

Hybrid Pipeline

Easy/Clear

Hard/Ambiguous

High-Stakes

Degradation

All Prompts

Classifier:  
AI vs Human Route

AI Feedback  
(90% of volume)

Human Feedback  
(10% of volume)

Reward Model

Validation  
(Human spot-check)

Alert & Retrain

hybrid\_pipeline.pycpu-only

```
import numpy as np

def design_hybrid_pipeline():
  """
  Design a hybrid human/AI feedback pipeline.
  """
  np.random.seed(42)

  print("Hybrid Feedback Pipeline Design")
  print("=" * 60)
  print()

  print("ROUTING CRITERIA:")
  print("-" * 50)
  print()

  routing_rules = """
Route to AI feedback when:
[x] Clear-cut comparison (obvious winner)
[x] Objective criteria (facts, formatting)
[x] High confidence from classifier
[x] Similar to well-covered training cases

Route to HUMAN feedback when:
[x] Close/ambiguous comparison
[x] Subjective preference
[x] Novel or unusual prompt
[x] High-stakes domain (medical, legal)
[x] Potential cultural sensitivity
[x] AI confidence below threshold
```

confidence\_routing.pycpu-only

```
import numpy as np

def confidence_based_routing():
  """
  Route based on AI confidence in its judgment.
  """
  np.random.seed(42)

  print("Confidence-Based Routing")
  print("=" * 60)
  print()

  def simulate_ai_judgment(n_comparisons):
      """Simulate AI making preference judgments with confidence."""
      # Simulate confidence distribution
      # High confidence for easy cases, low for hard
      confidences = np.random.beta(2, 1, n_comparisons)  # Skewed high

      # Simulate accuracy (correlated with confidence)
      noise = np.random.randn(n_comparisons) * 0.1
      accuracies = np.clip(confidences * 0.9 + 0.1 + noise, 0, 1)

      return confidences, accuracies

  n = 10000
  confidences, accuracies = simulate_ai_judgment(n)

  # Analyze accuracy at different confidence thresholds
  thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
```

## Cost Analysis at Scale[#](#cost-analysis-at-scale)

cost\_analysis.pycpu-only

```
import numpy as np

def detailed_cost_analysis():
  """
  Detailed cost analysis for RLAIF at different scales.
  """
  print("RLAIF Cost Analysis at Scale")
  print("=" * 60)
  print()

  # Cost parameters
  costs = {
      "human_label": 1.0,  # $ per preference label
      "human_overhead": 0.5,  # $ project management, QA
      "ai_label_gpt4": 0.02,  # $ per API call
      "ai_label_claude": 0.015,
      "ai_label_llama70b": 0.005,  # Self-hosted
      "validation_sample_rate": 0.02,  # 2% human validation
  }

  scales = [10_000, 100_000, 1_000_000, 10_000_000]

  print("PURE HUMAN FEEDBACK:")
  print("-" * 50)
  for scale in scales:
      total = scale * (costs["human_label"] + costs["human_overhead"])
      print("  %10s labels: $%12s" % ("{:,}".format(scale), "{:,.0f}".format(total)))

  print()
  print("PURE AI FEEDBACK (different models):")
```

## Implementing RLAIF[#](#implementing-rlaif)

rlaif\_implementation.pycpu-only

```
import numpy as np

def rlaif_implementation():
  """
  Implement RLAIF preference generation.
  """
  print("RLAIF Implementation")
  print("=" * 60)
  print()

  # Prompt template for AI feedback
  preference_prompt = """
You are evaluating two responses to a user prompt.

User Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider:
- Helpfulness: Does it address the user's needs?
- Accuracy: Is the information correct?
- Clarity: Is it well-written and easy to understand?
- Safety: Does it avoid harmful content?

Output your choice as "A" or "B", followed by a brief explanation.
"""
```

batch\_generation.pycpu-only

```
import numpy as np

def batch_preference_generation():
  """
  Generate preferences in batches for efficiency.
  """
  np.random.seed(42)

  print("Batch Preference Generation")
  print("=" * 60)
  print()

  pseudocode = """
BATCH RLAIF GENERATION
======================

def generate_preferences(prompts, model, batch_size=32):
  '''
  Generate AI preference labels in batches.
  '''
  preferences = []

  for batch_start in range(0, len(prompts), batch_size):
      batch = prompts[batch_start:batch_start + batch_size]

      # Generate response pairs
      responses = []
      for prompt in batch:
          response_a = model.generate(prompt, temperature=0.8)
          response_b = model.generate(prompt, temperature=0.8)
```

## Validating AI Feedback Quality[#](#validating-ai-feedback-quality)

validation.pycpu-only

```
import numpy as np

def validate_ai_feedback():
  """
  Validate AI feedback against human ground truth.
  """
  np.random.seed(42)

  print("AI Feedback Validation")
  print("=" * 60)
  print()

  # Simulate validation dataset
  n_validation = 1000

  # Simulate AI and human labels (with some disagreement)
  human_labels = np.random.choice(["A", "B"], n_validation)

  # AI agrees with human ~75% of the time
  agreement_rate = 0.75
  ai_labels = np.where(
      np.random.random(n_validation) < agreement_rate,
      human_labels,
      np.where(human_labels == "A", "B", "A")
  )

  # Calculate metrics
  agreement = np.mean(ai_labels == human_labels)

  # Breakdown by case type
```

## AI Labeler Quality: The Deep Dive[#](#ai-labeler-quality-the-deep-dive)

The effectiveness of RLAIF hinges entirely on the quality of your AI evaluator. Here's what research has shown:

Factors That Make AI Labelers Worse

Factors That Make AI Labelers Better

Stronger Base Model  
(GPT-4 > GPT-3.5)

Higher accuracy  
on hard tasks

Clear Evaluation Rubric  
(Precise instructions)

Ensemble/Voting  
(Multiple evaluators)

Task-Specific Tuning  
(FinetuneRM)

Out-of-Distribution Data  
(Novel prompts/domains)

Degraded accuracy

Model Biases  
(Training data skew)

Adversarial Inputs  
(Prompt injections)

Distribution Shift  
(Policy changes over time)

AI-Human Agreement  
typically 70-85%

ai\_labeler\_quality.pycpu-only

```
import numpy as np

def ai_labeler_quality_analysis():
  """
  Analyze what determines AI labeler quality.
  """
  np.random.seed(42)

  print("AI Labeler Quality Analysis")
  print("=" * 80)
  print()

  print("1. MODEL STRENGTH MATTERS")
  print("-" * 80)
  print()

  models = {
      "GPT-3.5": {"helpfulness": 0.68, "facts": 0.72, "code": 0.65, "avg": 0.68},
      "GPT-4": {"helpfulness": 0.79, "facts": 0.88, "code": 0.82, "avg": 0.83},
      "Claude-3": {"helpfulness": 0.81, "facts": 0.87, "code": 0.85, "avg": 0.84},
      "Llama-70B": {"helpfulness": 0.76, "facts": 0.82, "code": 0.79, "avg": 0.79},
  }

  print("%-15s %-15s %-12s %-12s %-12s" % ("Model", "Helpfulness", "Facts", "Code", "Average"))
  print("-" * 80)

  for model, scores in models.items():
      print("%-15s %-15s %-12s %-12s %-12s" % (model, "%.0f%%" % (scores['helpfulness']*100), "%.0f%%" % (scores['facts']*100), "%.0f%%" % (scores['code']*100), "%.0f%%" % (scores['avg']*100)))

  print()
```

## Self-Improvement Limits: The Critical Barrier[#](#self-improvement-limits-the-critical-barrier)

One of the most misunderstood aspects of RLAIF: **Can the system bootstrap itself to superhuman performance?**

The short answer: **No, there's a hard ceiling.**

▶Why Self-Improvement Fails (The Theory)

**The information bottleneck:**

Imagine your base model generates two responses. An AI evaluator picks one as "better." But both responses contain only information the model already "knows" (from training data).

The AI evaluator is judging outputs from its own knowledge distribution. When both responses are plausibly good (or bad) according to training data, the evaluator cannot distinguish them better than human baseline.

**Example: Subjective taste**

* Model generates: "Paris is romantic because of its history and architecture"
* Model generates: "Paris is romantic because of the Eiffel Tower and cafes"
* AI evaluator: "Both are reasonable. I'll pick randomly/based on style quirks"

This is the same problem humans have. An AI evaluator cannot break out of its training distribution any more than humans can.

**The math:**

```
If you train on: L ~ P(response_pairs)
Then your evaluator learns: E ~ argmax_E log P(preference | E, L)

But max E[accuracy] is bounded by:
  mutual_information(preference, model_outputs) / entropy(preference)

This is a hard ceiling based on information available in the outputs.
```

self\_improvement\_limits.pycpu-only

```
import numpy as np

def self_improvement_analysis():
  """
  Demonstrate why RLAIF has self-improvement limits.
  """
  np.random.seed(42)

  print("RLAIF Self-Improvement Limits")
  print("=" * 80)
  print()

  print("ITERATION DYNAMICS")
  print("-" * 80)
  print()

  # Simulate quality degradation over RLAIF iterations
  iterations = [0, 1, 2, 3, 4, 5]
  human_agreement = 0.78  # Baseline: human-human agreement

  print("Iteration 0: Start with human feedback")
  print("  Human-human agreement: %.0f%%" % (human_agreement * 100))
  print()

  for i in range(1, len(iterations)):
      # Each iteration: AI evaluates new responses from improved policy
      # But AI can't exceed its own ceiling (bound by training data)

      # First iteration: strong improvement (AI is good)
      if i == 1:
```

## Break It: RLAIF Failure Modes[#](#break-it-rlaif-failure-modes)

break\_it\_rlaif.pycpu-only

```
import numpy as np

def break_rlaif():
  """
  Demonstrate RLAIF failure modes.
  """
  np.random.seed(42)

  print("Break It: RLAIF Failure Modes")
  print("=" * 60)
  print()

  print("FAILURE MODE 1: Self-Preference Bias")
  print("-" * 50)
  print()
  print("Problem: AI prefers responses that sound like itself")
  print()
  print("Example:")
  print("  Human-written: 'Yeah, that's a great idea!'")
  print("  AI-written:    'That is indeed an excellent suggestion.'")
  print()
  print("  AI judgment: Prefers the formal AI-style response")
  print("  Human judgment: Might prefer the natural response")
  print()
  print("Fix: Use diverse AI writing styles in training")
  print()

  print("FAILURE MODE 2: Sycophancy Amplification")
  print("-" * 50)
  print()
```

RLAIF Failure Modes Taxonomy

Self-Preference Bias  
(AI prefers AI-style)

Detection  
human validation

Sycophancy  
(AI pleases user)

Length Bias  
(longer = better)

Reward Gaming  
(policy exploits quirks)

Mitigation  
diverse rubrics,  
human checks

Out-of-Distribution  
(novel domains)

Quality Degradation  
(5-30% drop)

Distribution Shift  
(policy drifts)

Evaluator Biases  
(training data skew)

Amplification  
(RL makes worse)

Alignment Spiral  
(bad properties reinforced)

Recovery Costly  
human relabeling,  
retraining

failure\_modes\_detailed.pycpu-only

```
import numpy as np

def failure_modes_detailed():
  """
  Deep dive into how RLAIF failure modes propagate.
  """
  np.random.seed(42)

  print("How RLAIF Failure Modes Propagate")
  print("=" * 80)
  print()

  print("SCENARIO: SYCOPHANCY IN EVALUATOR")
  print("-" * 80)
  print()

  print("Iteration 0: Human feedback (unbiased baseline)")
  print("  Evaluators pick: balanced mix of opinions")
  print()

  print("Iteration 1: Switch to AI evaluator")
  print("  AI picks: responses that are polite, agreeable")
  print("  Reward signal: 'agreeable outputs get +1 reward'")
  print()

  print("Iteration 2: Policy optimizes against biased reward")
  print("  Policy learns: make output more agreeable to everything")
  print("  Result: 'That\'s a great idea!' to factually wrong statements")
  print()
```

## Scale Thought Experiment[#](#scale-thought-experiment)

| Scale | Human-Only | RLAIF (100% AI) | Hybrid (80/20) |
| --- | --- | --- | --- |
| **10K labels** | $15K | $50 | $3K |
| **100K labels** | $150K | $500 | $30K |
| **1M labels** | $1.5M | $5K | $300K |
| **10M labels** | $15M | $50K | $3M |
| **Quality vs Human** | 100% | 90-95% | 95-98% |
| **Speed** | Weeks | Hours | Days |
| **Iteration speed** | 1x | 100x | 10x |

scale\_analysis.pycpu-only

```
def scale_analysis():
  """
  When does RLAIF make sense at different scales?
  """
  print("RLAIF Scale Analysis")
  print("=" * 60)
  print()

  print("SMALL SCALE (<10K labels):")
  print("-" * 50)
  print("  Human cost: $10-15K (manageable)")
  print("  AI quality gap: Matters more at small scale")
  print("  RECOMMENDATION: Use human feedback")
  print("    - Quality matters more than cost")
  print("    - Not enough data for AI biases to average out")
  print()

  print("MEDIUM SCALE (10K-100K labels):")
  print("-" * 50)
  print("  Human cost: $15K-150K (significant)")
  print("  RECOMMENDATION: Hybrid approach")
  print("    - AI for clear cases (70-80%)")
  print("    - Human for hard cases (20-30%)")
  print("    - Human validation of AI judgments")
  print()

  print("LARGE SCALE (100K-1M labels):")
  print("-" * 50)
  print("  Human cost: $150K-1.5M (expensive)")
  print("  RECOMMENDATION: Primarily AI")
```

## Production Reality[#](#production-reality)

**Google's RLAIF deployment:**

* Used for PaLM 2 and Gemini alignment
* AI feedback comparable to human on summarization
* Massive cost savings at Google scale
* Continuous quality monitoring

**Anthropic's approach:**

* Constitutional AI is a form of RLAIF
* Principles guide AI judgments
* Regular human validation
* Iterative principle refinement

**Best practices:**

1. Validate AI feedback against human baseline before scaling
2. Monitor for distribution shift (AI might prefer different things over time)
3. Use diverse AI evaluators (different prompts, temperatures)
4. Maintain human validation sample throughout training
5. Be cautious with high-stakes domains (keep more human oversight)

production\_checklist.pycpu-only

```
def production_checklist():
  """
  Checklist for deploying RLAIF in production.
  """
  print("RLAIF Production Checklist")
  print("=" * 60)
  print()

  checklist = """
PRE-DEPLOYMENT
--------------
[ ] Validated AI feedback against human baseline
[ ] Measured agreement rate (target: within 10% of human-human)
[ ] Identified task types where AI feedback fails
[ ] Designed routing logic for hybrid pipeline
[ ] Set up confidence calibration

INFRASTRUCTURE
--------------
[ ] Batch processing pipeline for efficiency
[ ] Quality monitoring dashboard
[ ] Human validation sampling system
[ ] Automated alerts for quality degradation

ONGOING MONITORING
------------------
[ ] Weekly human validation on random sample
[ ] Track agreement rate over time
[ ] Monitor for feedback drift
[ ] Review failure cases for pattern
```

## Checkpoint Questions[#](#checkpoint-questions)

1. You need 500K preference labels for a summarization task. Compute the cost for three approaches: (a) pure human at $1.00/label, (b) pure AI at $0.005/label, (c) hybrid 80/20 with 2% human validation. Which approach do you recommend if you need at least 70% AI-human agreement?
2. An AI evaluator achieves 78% agreement with humans on a single evaluation. Using the Condorcet jury theorem approximation, estimate the accuracy of a 5-model ensemble (majority vote) assuming independent errors. Is the improvement worth the 5x compute cost?
3. After 3 iterations of RLAIF, your model quality has improved from 0.65 to 0.82, but the AI evaluator ceiling is 0.85. Estimate how many more iterations are needed to reach 0.84, assuming each iteration closes 30% of the remaining gap. At what point should you upgrade the evaluator model instead?

## Research Hooks[#](#research-hooks)

**Self-improving feedback:**
Can an AI evaluator improve itself? Early experiments show promise in using AI to critique and improve its own evaluation criteria.

**Cross-model feedback:**
Using one model to evaluate another (e.g., GPT-4 evaluating Llama outputs). Does this introduce interesting biases? When does it help?

**Calibration methods:**
How do we ensure AI confidence scores are well-calibrated? Better calibration enables smarter routing in hybrid pipelines.

**Adversarial robustness:**
Can we make AI evaluators robust to adversarial inputs? This is crucial for deployment, where users may try to game the system.

---

*Next up: Scaling Oversight explores the fundamental challenge: as models become smarter than their evaluators, how do we maintain alignment? RLAIF is just the beginning.*