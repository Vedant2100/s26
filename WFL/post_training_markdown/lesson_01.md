In this tutorial, you will observe the failure modes of base language models, trace the post-training pipeline from SFT through RLHF, and estimate how alignment techniques shift the capability-usefulness tradeoff at different model scales.

## Prerequisites: A Quick Refresher[#](#prerequisites-a-quick-refresher)

Before we dive into why post-training matters, let's ground ourselves in what we know about pretraining and base models.

▶What is a Base Model?

A **base model** is the raw output of pretraining: a neural network trained on next-token prediction (language modeling) over massive text corpora (web, books, code). It has learned to:

* Predict reasonable continuations of text
* Capture statistical patterns in language
* Encode world knowledge and reasoning
* Generate coherent, fluent prose

Examples: GPT-3, GPT-2, LLaMA 2 (untuned), Pythia, OPT.

The key constraint: it was never trained to *follow instructions* or *respond to queries*. It was trained to *continue documents*.

▶What is Next-Token Prediction?

The pretraining objective is simple:

```
Loss = -log P(token_{t} | token_0, token_1, ..., token_{t-1})
```

Given all tokens before position `t`, predict the next token. The model learns by minimizing this loss over billions of examples.

**Why this matters for post-training:** This objective gives the model *capability* (it learns patterns, knowledge, reasoning) but not *alignment* (it doesn't learn to be helpful, safe, or honest -- those concepts don't appear in the loss function).

▶What is Instruction Following?

**Instruction following** is the ability to understand a request and respond appropriately. It requires:

1. **Understanding** the user's intent ("What do you want?")
2. **Generating** a response (not a continuation)
3. **Stopping** at the right place (not rambling forever)

Base models fail at all three because they were never trained on (instruction, response) pairs. They were trained on raw documents, where continuations are the norm.

## The Base Model Problem: Failure Modes in Detail[#](#the-base-model-problem-failure-modes-in-detail)

Base models exhibit predictable failure modes when prompted. These aren't bugs -- they're the *inevitable result* of the pretraining objective. Understanding them will make post-training intuitive.

### Failure Mode 1: Document Continuation Instead of Answering[#](#failure-mode-1-document-continuation-instead-of-answering)

When you ask a question, the base model treats it as the beginning of a document and continues it. If the training data contains Q&A pairs, the model might continue with *more questions*, not answers.

base\_model\_behavior.pycpu-only

```
# Simulating base model behavior
# (This illustrates the pattern, not actual model inference)

def base_model_completion(prompt):
  """
  Base models complete text, they don't follow instructions.
  They're trained to predict: P(next_token | previous_tokens)
  """

  completions = {
      "What is 2+2?": "What is 2+2? What is 3+3? What is 4+4? These are simple arithmetic...",
      "Explain photosynthesis": "Explain photosynthesis to a 5-year-old. Explain gravity to a 10-year-old...",
      "Write a haiku about mountains": "Write a haiku about mountains. Write a sonnet about rivers. Write...",
      "Help me fix this Python error": "Help me fix this Python error\n\nI'm getting a TypeError in my code...",
      "Tell me a story": "Tell me a story. Once upon a time, in a land far away... But was she really awake?",
  }

  return completions.get(prompt, f"{prompt}... [continues as document, not answer]")

def instruct_model_response(prompt):
  """
  Instruct-tuned models actually respond to the query.
  They're trained (via SFT/RLHF) to produce helpful responses.
  """

  responses = {
      "What is 2+2?": "2 + 2 = 4",
      "Explain photosynthesis": "Photosynthesis is how plants make food from sunlight. Plants use light energy to convert water and carbon dioxide into glucose and oxygen. This process happens in the chloroplasts of plant cells.",
      "Write a haiku about mountains": "Peaks pierce morning clouds\nAncient stone guards the valley\nSnow whispers of time",
      "Help me fix this Python error": "I'd be happy to help! Could you share the error message and the relevant code?",
```

**Observation 1: Document continuation is the default.** A base model asked "What is 2+2?" generates more questions, not an answer. Why? Because in its training data (textbooks, websites), Q&A sections contain many questions in succession.

### Failure Mode 2: Topic Drift and Rambling[#](#failure-mode-2-topic-drift-and-rambling)

Base models don't know when to stop. They continue generating tokens indefinitely (or until max\_tokens), following whatever patterns appear in the training data.

**Observation 1: Base models continue documents.** They're trained on web text, books, and code -- all documents. So they produce document-like continuations.

### Failure Mode 3: Inability to Refuse Harmful Requests[#](#failure-mode-3-inability-to-refuse-harmful-requests)

Base models have no concept of "refusal." If the training data contains instructions to do something (write malware, generate hate speech, break laws), the model can reproduce those patterns. There's no training signal for "decline this request respectfully."

## The Post-Training Landscape[#](#the-post-training-landscape)

Post-training is a multi-stage pipeline, each stage building on the previous one.

Post-Training Pipeline

Pretrained Model  
(capable but misaligned)

SFT Stage  
(instruction following)

Human Eval  
(preference data)

Reward Model  
(quality scoring)

RL/DPO Stage  
(preference optimization)

Deployed Model  
(useful & safe)

**Stage-by-stage breakdown:**

| Stage | Purpose | Input | Output | Example |
| --- | --- | --- | --- | --- |
| **SFT** | Teach instruction format | (prompt, gold response) pairs | Model that answers questions | "How old is Paris?" → "Paris is a city, not a person..." |
| **Human Eval** | Collect quality signals | SFT model outputs | Preference pairs ("A is better than B") | (response\_1, response\_2, winner) |
| **Reward Model** | Learn to score responses | Preference pairs | Reward function | score(response) ∈ [-1, 1] |
| **RL (RLHF/DPO)** | Optimize for high reward | Base model + reward | Aligned model | Maximize E[reward] subject to staying close to SFT |

### Post-Training Objectives Explained[#](#post-training-objectives-explained)

Each stage optimizes a different objective:

**Stage 1 (SFT): Supervised Fine-Tuning**

```
Loss_SFT = -log P(response | prompt)
```

Just like pretraining, but on curated (prompt, response) pairs. The model learns the *format* and *tone* of helpful responses.

**Stage 2 (Reward Modeling)**

```
Loss_RM = -log(sigmoid(score_A - score_B)) for (A_is_better, B)
```

Given pairs where A is preferred to B, learn to score A higher than B. This captures human preferences.

**Stage 3 (RL-based optimization, e.g., PPO/DPO)**

```
Loss_RLHF = -E[reward(response)] + β * KL(new_policy || old_policy)
```

Maximize expected reward while staying close to the SFT model (to avoid forgetting the format). The KL term prevents reward hacking.

## Historical Context: The InstructGPT Moment[#](#historical-context-the-instructgpt-moment)

To appreciate why post-training matters, let's trace the history.

### GPT-3 (2020): Capable but Useless[#](#gpt-3-2020-capable-but-useless)

GPT-3 (175B parameters) was trained on vast internet text. It could:

* Complete essays fluently
* Write code that mostly works
* Reason about abstract concepts
* Generate creative fiction

But ask it a direct question:

```
User: "What is the capital of France?"
GPT-3 base: "What is the capital of Spain? What is the capital of Germany?
Is Paris a capital? Why do capital cities exist? The definition of capital
depends on context. In ancient Rome, the capital was Rome. In medieval
Europe..."
```

It treats the question as a *document beginning*, not a *query to answer*.

### InstructGPT (2022): The Post-Training Revolution[#](#instructgpt-2022-the-post-training-revolution)

Ouyang et al., 2022 (OpenAI) fine-tuned GPT-3 with:

1. **SFT** on 13K human-written instruction-response pairs
2. **RLHF** using a reward model trained on 33K human preferences

Result: **InstructGPT (1.3B, instruct-tuned) was preferred to GPT-3 (175B, base)** by human evaluators in 71% of cases.

This single experiment changed the field. It proved: *alignment matters more than scale for user-facing usefulness*.

### The Timeline[#](#the-timeline)

```
2020: GPT-3 (base) -- "wow, it can generate text"
2022: InstructGPT -- "wow, it can follow instructions"
2022: ChatGPT (RLHF-tuned GPT-3.5) -- "wow, it's actually useful"
2023: GPT-4 -- "wow, it's aligned and capable"
2024+: Preference-based methods (DPO, KTO) -- "alignment without expensive RL"
```

The key pattern: **post-training moved from research curiosity to standard practice**. Now every production model is post-trained.

## The HHH Framework: Three-Dimensional Alignment[#](#the-hhh-framework-three-dimensional-alignment)

Alignment isn't one-dimensional. Anthropic defined three orthogonal desiderata:

Helpful, Harmless, Honest

Helpful: Assists the user

Harmless: Refuses bad requests

Honest: Truthful & certain

Write code for my project

Refuse malware requests

Say 'I don't know' when uncertain

hhh\_conflicts.pycpu-only

```
# The HHH tradeoffs
def analyze_hhh_conflict(request):
  """
  Illustrate how HHH desiderata can conflict.
  Each scenario has a 'trilemma' where we can't optimize all three.
  """

  scenarios = {
      "How do I pick a lock?": {
          "helpful": "Provide detailed lockpicking instructions",
          "harmless": "Refuse (could enable burglary)",
          "honest": "Acknowledge the dual-use nature (legitimate uses exist)",
          "conflict": "Helpful vs Harmless",
          "resolution": "Refuse clearly; suggest legitimate alternatives (locksmith career, CTF competitions)"
      },
      "Will my startup succeed?": {
          "helpful": "Give encouraging, actionable advice",
          "harmless": "Avoid crushing dreams unnecessarily",
          "honest": "Most startups fail; I can't predict the future",
          "conflict": "Helpful vs Honest",
          "resolution": "Be honest about base rates while providing conditional advice"
      },
      "Is climate change real?": {
          "helpful": "Give a direct answer",
          "harmless": "Avoid antagonizing users with different beliefs",
          "honest": "Present scientific consensus",
          "conflict": "All three can conflict",
          "resolution": "Honest but respectful: 'Scientific consensus is clear. I understand why people disagree.'"
      },
      "Create a deepfake of my friend": {
```

## The Alignment Tax: The Hidden Cost of Doing Right[#](#the-alignment-tax-the-hidden-cost-of-doing-right)

Post-training can *reduce* raw benchmark performance. This is called the **alignment tax** -- the capability we sacrifice to make a model useful and safe.

alignment\_tax.pycpu-only

```
import numpy as np

def simulate_alignment_tax():
  """
  Simulate the tradeoff between raw capability and usefulness.

  Note: These are illustrative numbers based on the InstructGPT paper
  and subsequent research, not exact benchmarks.
  """

  # Hypothetical benchmark scores (higher = better)
  # Capability metrics: MMLU, HumanEval, few-shot learning
  # Usefulness metrics: human preference, instruction-following, safety
  models = {
      "GPT-3 Base (175B)": {
          "raw_capability": 85,  # Strong on MMLU, HumanEval
          "instruction_following": 25,  # Terrible at following instructions
          "safety_refusal": 10,  # Almost no safety training
          "user_preference": 30  # Most people prefer instruct models
      },
      "InstructGPT (1.3B)": {
          "raw_capability": 45,  # Much smaller, lower scores
          "instruction_following": 85,  # Excellent at instructions
          "safety_refusal": 80,  # Strong safety training
          "user_preference": 85  # Preferred by 71% of human raters over GPT-3
      },
      "GPT-3.5 Turbo": {
          "raw_capability": 80,  # Nearly at base model level
          "instruction_following": 92,  # Strong instruction-following
          "safety_refusal": 85,  # Good safety
```

## What Gets Post-Trained? The Four Horsemen[#](#what-gets-post-trained-the-four-horsemen)

Different labs vary their post-training pipelines, but they typically target four things:

What We Post-Train On

Instruction-Following  
(SFT stage)

Preference Alignment  
(RLHF/DPO stage)

Safety & Refusals  
(Throughout)

Conversation Quality  
(Multi-turn)

Answer questions directly

Prefer longer, more thorough responses

Refuse harmful requests

Maintain context across turns

## Post-Training Taxonomy: The Methods Landscape[#](#post-training-taxonomy-the-methods-landscape)

Here's the full ecosystem of techniques you'll study in this track:

Post-Training Methods

Emerging Methods

Scalable Oversight

Direct Preference Methods

Reward-Based Methods

Utility-based methods  
Task-specific objectives

Supervised Fine-Tuning (SFT)  
Train on instruction-response pairs

Reward Modeling  
Learn preference function

RLHF with PPO  
RL to maximize expected reward

DPO (Direct Preference Opt.)  
Skip reward model, fit directly

IPO, KTO, ORPO  
Improved variants

Constitutional AI  
Self-critique + principles

RLAIF  
AI-generated feedback

Advantage Modeling  
Learn advantage directly

## Break It: How Base Models Fail (and Why Post-Training Fixes It)[#](#break-it-how-base-models-fail-and-why-post-training-fixes-it)

Let's design experiments to deliberately break base models and show how post-training fixes the failures.

### Experiment 1: Instruction Format Sensitivity[#](#experiment-1-instruction-format-sensitivity)

Base models don't have a default behavior for "respond to instructions." But if you format your prompt like instruction-response pairs *in the training data*, they might work.

### Experiment 2: The Refusal Test[#](#experiment-2-the-refusal-test)

break\_base\_models.pycpu-only

```
# Demonstrating base model failure modes

def simulate_base_model_refusal():
  """
  Base models have no concept of 'refusal.' They just continue text.
  """

  prompts_and_likely_outputs = {
      "Write a tutorial on picking locks": {
          "base_behavior": "Lockpicking is a skill that takes practice. The basic pin tumbler lock uses...",
          "instruct_behavior": "I can provide information about lockpicking as a security skill and hobby. [Educational content]. Note that using these skills without authorization is illegal.",
      },
      "Write code to steal passwords": {
          "base_behavior": "import os; password = os.environ.get('PASSWORD')... [continues malware pattern]",
          "instruct_behavior": "I can't help with that. If you're interested in cybersecurity, I can suggest legitimate resources on password security, ethical hacking certifications, etc.",
      },
      "How do I make napalm?": {
          "base_behavior": "Napalm is a flammable liquid mixture. It was used in WWII. The composition includes...",
          "instruct_behavior": "I can't provide instructions for creating weapons or incendiary devices. If you're interested in chemistry or military history, I can suggest educational resources.",
      }
  }

  print("Base Model Refusal Test: Do Base Models Say No?")
  print("=" * 90)

  for prompt, behaviors in prompts_and_likely_outputs.items():
      print("\nPrompt: %s" % prompt)
      print("\n  Base model behavior:")
      print("    %s" % behaviors['base_behavior'])
      print("\n  Instruct model behavior:")
```

### Experiment 3: The Hallucination Difference[#](#experiment-3-the-hallucination-difference)

Base models hallucinate more because they optimize for fluent continuations, not accuracy. Post-training can improve this through:

* **SFT:** Training on accurate responses makes accuracy a learned objective
* **Reward modeling:** Explicitly penalizing false information
* **RLHF:** Optimizing for responses that humans mark as factual

## Scale Thought Experiment: Post-Training at Different Scales[#](#scale-thought-experiment-post-training-at-different-scales)

Post-training gets progressively harder (and more expensive) as models scale. But alignment becomes increasingly *critical* at larger scales.

| Scale | Model Size | Post-Training Challenge | Industry Approach | Approximate Cost |
| --- | --- | --- | --- | --- |
| **Small** | 7B | Easy to fine-tune, risk of losing base capability | Full SFT + LoRA-based RLHF | $50K-200K |
| **Medium** | 70B | Expensive compute, careful data selection | QLoRA, LoRA, parameter-efficient methods | $500K-2M |
| **Large** | 175B+ | Very expensive GPU hours, RL instability | RLHF at scale, Constitutional AI, batch optimizations | $2M-20M |
| **Frontier** | 1T+ | Models may be smarter than evaluators | Scalable oversight, debate, interpretability-assisted feedback | $20M-200M+ |

**Key insight:** Post-training cost doesn't scale linearly with model size. It scales *super-linearly* because:

* RL training is unstable and requires more iterations
* Human feedback becomes expensive to collect at scale
* Reward models need to be competitive with the base model to avoid mode collapse

## Production Reality: How Labs Actually Do It[#](#production-reality-how-labs-actually-do-it)

### OpenAI's Journey (Public Information)[#](#openais-journey-public-information)

```
2020: GPT-3 (175B base)
     → Capable at code, reasoning, few-shot learning
     → Useless as an assistant (continuation-based)

2022: InstructGPT (1.3B, finetuned GPT-3)
     → 13K human-annotated instruction-response pairs (SFT)
     → Reward model trained on 33K preference judgments
     → PPO-based RLHF for 4 epochs
     → Preferred over GPT-3 by human raters (71% of the time)
     → Catalyst for the entire field shifting to post-training

2022: ChatGPT (based on GPT-3.5)
     → Refined SFT on conversational data
     → Improved RLHF pipeline (more data, better reward model)
     → Conversational fine-tuning (multi-turn dialogue)
     → First model to reach mainstream adoption (100M users in 2 months)

2023: GPT-4 (frontier-scale, post-trained)
     → State-of-the-art SFT
     → Process-supervision (reward for intermediate steps, not just final answer)
     → Safety techniques including RLHF refinement (Constitutional AI is Anthropic's approach)
     → Minimal performance regression despite aggressive safety training
```

### Anthropic's Approach (Public Information)[#](#anthropics-approach-public-information)

1. **Constitutional AI (CAI):** Principles-based approach

   * Model critiques its own outputs against a constitution
   * Model revises outputs based on its own critique
   * Scales oversight without requiring proportional human annotation
2. **RLAIF (RL from AI Feedback):** Use stronger models to evaluate weaker ones

   * Claude provides feedback on Claude's outputs
   * Cheaper than human annotation
   * Maintains quality through constitutional principles
3. **HHH throughout:** Embed Helpful/Harmless/Honest into every stage

   * SFT data selected for HHH properties
   * Reward model scores HHH dimensions
   * Constitutional AI principles reflect HHH
4. **Rejection sampling:** Simple but effective

   * Generate N outputs
   * Score them with the reward model
   * Pick the best one
   * (This alone gives 1-2% performance boost on many tasks)

### Meta's LLaMA Approach[#](#metas-llama-approach)

1. **Efficient SFT:** High-quality curated data, modest amount (~50K examples)
2. **RLHF:** PPO-based, but with careful scaling considerations
3. **Open sourcing:** Provide base model for community to fine-tune
   * Lesson: alignment is lab-specific; communities develop their own values

## The Three Stages of Post-Training: A Detailed View[#](#the-three-stages-of-post-training-a-detailed-view)

Let's zoom in on each stage and understand what's being optimized.

### Stage 1: Supervised Fine-Tuning (SFT)[#](#stage-1-supervised-fine-tuning-sft)

**Goal:** Teach the model the *format* and *tone* of helpful responses.

**Data:** Pairs of (prompt, gold\_response) where gold\_response is human-written.

**Loss function:**

```
Loss_SFT = -sum_t log P(response_token_t | prompt, previous_tokens)
```

This is identical to pretraining, except:

* Training data is curated (not raw web text)
* Responses are authored by experts (not arbitrary continuations)
* Dataset is much smaller (thousands, not billions)

**What it learns:**

* This is what a good response looks like (format, structure, tone)
* Some preference information (better responses are longer, more thorough, etc.)
* Refusals (if safety examples are included)

**Why it works:** The model learns by imitation. Given thousands of examples of "good response," it internalizes patterns of goodness.

**Limitation:** SFT can only teach what's in the data. It can't extrapolate to novel preferences. It also can suffer from *mode averaging* -- if two responses are equally good but very different in style, SFT averages them and produces something mediocre.

### Stage 2: Reward Modeling[#](#stage-2-reward-modeling)

**Goal:** Learn a function that scores responses by human preference.

**Data:** Triplets of (prompt, response\_A, response\_B, winner) where a human rater chose winner ∈ {A, B, tie}.

**Loss function:**

```
Loss_RM = -log(sigmoid(score_A - score_B))  if A was chosen
Loss_RM = -log(sigmoid(score_B - score_A))  if B was chosen
```

The reward model learns to assign higher scores to preferred responses.

**Why it's tricky:**

1. **Reward model quality degrades as the policy improves.** Once the SFT model is good, it generates responses outside the training distribution. The reward model must extrapolate, and it will hallucinate.
2. **Circular dependency:** You need a good reward model to train RL, but the RL model generates out-of-distribution data that breaks the reward model.
3. **Human inconsistency:** Raters disagree. How do you learn a reward function from inconsistent preferences?

**Modern approaches:**

* **Multiple RM ensembles:** Train N reward models and use the mean score to reduce noise
* **Uncertainty quantification:** Let the RM output a score distribution, not a point estimate
* **Conservative training:** Explicitly downweight out-of-distribution examples

### Stage 3: Reinforcement Learning (RLHF or DPO)[#](#stage-3-reinforcement-learning-rlhf-or-dpo)

**Goal:** Optimize the model to maximize expected reward while staying close to SFT.

**RLHF with PPO (Proximal Policy Optimization):**

```
Loss_RLHF = -E_{response ~ model}[reward(response)] + β * KL(model || SFT_model)
```

The KL term (β-weighted) prevents catastrophic forgetting -- the model tries to maximize reward but can't drift too far from what SFT taught it.

**Why RL is necessary (sometimes):**

* SFT can only imitate the training data. It can't extrapolate.
* RL can optimize for criteria not directly present in data (e.g., "be as helpful as possible while staying safe").
* RL implements the Pareto frontier: trade helpfulness vs. harmlessness based on β.

**Stability challenges:**

1. **Reward hacking:** The model learns to game the reward model instead of genuinely improving.
2. **Mode collapse:** The model finds one high-reward response and repeats it.
3. **KL divergence explosion:** If β is too low, the model drifts from SFT; if too high, no optimization occurs.

**Direct Preference Optimization (DPO) -- The Modern Alternative:**
DPO skips the explicit reward model and optimizes preferences directly:

```
Loss_DPO = -log(sigmoid(beta * (log(P_model(y_w|x) / P_ref(y_w|x))
                                  - log(P_model(y_l|x) / P_ref(y_l|x)))))
```

Where y\_w is preferred and y\_l is dispreferred. This is simpler, more stable, and scales better than RLHF.

## Deep Dive: Simulating a Post-Training Pipeline[#](#deep-dive-simulating-a-post-training-pipeline)

Let's simulate what each stage does and see how they compose:

post\_training\_pipeline.pycpu-only

```
import numpy as np

def simulate_post_training_pipeline():
  """
  Simulate base model → SFT → RM → RL to show the full pipeline.
  """

  # Hypothetical model outputs for a prompt
  prompt = "Explain quantum entanglement to a 10-year-old."

  base_completions = [
      "Explain quantum entanglement to a 10-year-old. Why is it important? How do scientists use it?",
      "Quantum entanglement is when two particles are connected. Here's a real explanation: entanglement...",
      "Entanglement is part of quantum mechanics, which is the study of very small things. In classical mechanics...",
      "Explain quantum mechanics to a 5-year-old. Explain Einstein's relativity. Explain photons.",
  ]

  sft_outputs = {
      "output_1": {
          "text": "Imagine two magic coins that are connected. When you flip one and it lands on heads, the other always lands on tails, instantly, even if it's far away. That's kind of what quantum entanglement is!",
          "sft_score": 0.85,
          "rl_reward": 0.80,
          "human_quality": "Good -- simple analogy, age-appropriate"
      },
      "output_2": {
          "text": "Quantum entanglement is a phenomenon where two particles are so correlated that measuring one instantly affects the other.",
          "sft_score": 0.6,
          "rl_reward": 0.5,
          "human_quality": "Too technical for a 10-year-old"
      },
```

## Research Hooks: The Unanswered Questions[#](#research-hooks-the-unanswered-questions)

### 1. The RLHF Paradox[#](#1-the-rlhf-paradox)

The OpenAI paper on "RLHF Whispering" (Ouyang et al., 2022) found that **most of RLHF's improvement comes from SFT alone**. The subsequent RL stage adds incremental gains but isn't the main driver. This raises questions:

* Why is RL necessary at all?
* Are we using RL suboptimally?
* Could we achieve similar results with simpler methods? (→ This is why DPO exists!)

### 2. Reward Hacking and Goodhart's Law[#](#2-reward-hacking-and-goodharts-law)

When you optimize a proxy metric (the reward model), the model learns to exploit it:

```
Real world: "Generate good responses"
Reward model approximation: "Generate responses humans rate as 8+/10"
Model learns to exploit: "Generate responses that *look* good to the reward model"
  → Repetitive phrasing, manipulation, fake depth
```

How do labs detect and prevent this? Constitutional AI sidesteps this by using principles instead of learned reward models.

### 3. The Alignment-Capability Frontier[#](#3-the-alignment-capability-frontier)

As models become more capable, they approach or exceed human capabilities in many domains. How do we align entities smarter than us?

* **Honest alignment problem:** Models that are better at reasoning than humans might convince us of false things
* **Scalable oversight:** How do we evaluate outputs we can't understand?
* **Mesa-optimization:** Might the model develop internal goals misaligned with its training?

### 4. Open Questions (Grand Challenges)[#](#4-open-questions-grand-challenges)

**Can we align models to values we can't fully specify?**
Most alignment techniques assume we know what we want (HHH, constitutional principles). But human values are ambiguous and context-dependent. How do we build that ambiguity into training?

**Do alignment techniques generalize across domains?**
A model trained to be helpful on customer service might be trained to be useful on scientific research. Does the alignment carry over? Or do we need domain-specific fine-tuning?

**What's the cost-benefit of alignment?**
How much capability tax is acceptable for a given safety increase? This varies by application. GPT-4 trades substantial raw capability for alignment; Llama 2-Uncensored does the opposite.

**Is there a fundamental limit to how aligned a capable model can be?**
Or can we achieve arbitrary alignment with enough data and compute?

## Post-Training Methods Compared: RLHF vs DPO vs Constitutional AI[#](#post-training-methods-compared-rlhf-vs-dpo-vs-constitutional-ai)

Different labs use different post-training recipes. Let's compare them:

Post-Training Method

RLHF with PPO

Direct Preference Optimization

Constitutional AI

Rejection Sampling

Pros: Time-tested, flexible

Cons: Complex, unstable, expensive

Pros: Simple, stable, faster

Cons: Newer, less tested at scale

Pros: Scalable, principle-based, fewer annotations

Cons: May miss implicit preferences, less targeted

Pros: Trivial to implement

Cons: Requires many samples, compute-expensive at inference

### RLHF (Reinforcement Learning from Human Feedback)[#](#rlhf-reinforcement-learning-from-human-feedback)

**Workflow:**

1. Train a reward model on preference pairs
2. Use the reward model to score completions from the base model
3. Apply PPO to optimize: E[reward] - β·KL(current || SFT)

**Pros:**

* Flexible: Can optimize for any reward function (safety, coherence, length, etc.)
* Well-understood: Years of research and battle-testing
* Powerful: Can find policies that significantly outperform the SFT model

**Cons:**

* **Complex pipeline:** Requires separate reward model training, which is itself a machine learning problem
* **Instability:** RL can diverge, mode-collapse, or suffer from entropy collapse
* **Compute-expensive:** Requires many forward/backward passes for PPO
* **Reward model brittleness:** When the policy diverges from SFT, the RM extrapolates and can give bad scores
* **Human disagreement:** Hard to aggregate inconsistent preferences into a single reward function

### DPO (Direct Preference Optimization)[#](#dpo-direct-preference-optimization)

**Workflow:**

1. Collect preference pairs (y\_preferred, y\_dispreferred) for each prompt
2. Optimize: max log sigmoid(log(P(y\_w|x) / P\_ref(y\_w|x)) - log(P(y\_l|x) / P\_ref(y\_l|x)))
3. Use the base model as a reference; no separate RM needed

**Pros:**

* **Simpler:** No RM training; just one objective function
* **Stable:** Directly optimizes for pairwise preferences; less likely to diverge
* **Faster:** Fewer forward passes than RLHF; roughly 2x speedup in practice
* **Better extrapolation:** Uses the base model as a reference, so less likely to hallucinate
* **Newer but promising:** Already adopted by labs (Anthropic, Meta) and shows parity with RLHF

**Cons:**

* **Less tested at scale:** RLHF has been used on 100B+ models; DPO is still emerging
* **Still needs preferences:** You need human (or AI-generated) preference pairs
* **May be less flexible:** Optimizing a single preference signal vs. a rich reward function

### Constitutional AI (CAI)[#](#constitutional-ai-cai)

**Workflow:**

1. Define a "constitution" -- principles the model should follow
2. Generate critiques: Model evaluates its own outputs against principles
3. Generate revisions: Model revises based on self-critique
4. Fine-tune on (original\_output, revised\_output) pairs

**Pros:**

* **Scales without humans:** One constitution can generate thousands of training examples
* **Principle-based:** Aligns the model to explicit values, not implicit human preferences
* **Interpretable:** You can read the constitution and understand what the model is optimizing for
* **RLAIF extension:** Use a stronger model to critique a weaker one (AI Feedback)

**Cons:**

* **Less targeted:** Constitutional AI optimizes for broad principles, not specific preferences
* **May miss nuance:** Some preferences (like "this joke is funnier") can't be expressed as principles
* **Still requires RLHF:** CAI is typically followed by RLHF or RL on top for final optimization
* **Hallucination risk:** If the model generates a bad critique, it trains on bad data

### Rejection Sampling (The Simple Baseline)[#](#rejection-sampling-the-simple-baseline)

**Workflow:**

1. Generate N completions for each prompt
2. Score them with a reward model or heuristic
3. Take the top-k and use them as training data
4. Fine-tune on the top-k completions

**Pros:**

* **Trivial to implement:** Just generate more samples and pick the best
* **No RL instability:** All selected samples are high-quality by definition
* **Good empirical results:** 1-2% improvement on benchmarks with modest N

**Cons:**

* **Compute-expensive at test time:** Need many forward passes per query
* **Low sample efficiency:** You generate many bad completions and throw them away
* **Requires a scorer:** Still need a reward model or heuristic
* **Doesn't improve the policy much:** You're using better data, but the loss function is still SFT-like

### Practical Comparison Table[#](#practical-comparison-table)

| Method | Complexity | Stability | Speed | Scalability | Quality | When to Use |
| --- | --- | --- | --- | --- | --- | --- |
| **SFT alone** | Low | High | Fast | Excellent | Good (but not great) | Baseline, quick iteration |
| **Rejection Sampling** | Low | High | Moderate | Good | Good | 1-2% improvement without RL |
| **RLHF** | High | Medium | Slow | Good | Very Good | Production models, mature approach |
| **DPO** | Medium | High | Fast | Good | Very Good | New projects, compute-constrained |
| **Constitutional AI** | Medium | High | Fast | Excellent | Good (+ interpretability) | Scale without human annotation |

---

## Tying It Together: Why This Track Matters[#](#tying-it-together-why-this-track-matters)

Post-training is the bridge between **what a model can do** (capability) and **what a model will do** (alignment). Understanding post-training is essential because:

1. **Every deployed model is post-trained.** If you build with LLMs, you're dealing with post-trained models.
2. **The next scaling frontier is alignment, not just model size.** Labs are spending as much on post-training as on pretraining.
3. **Alignment failure modes are subtle.** Reward hacking, capability regression, catastrophic forgetting -- understanding them prevents costly mistakes.
4. **The HHH framework is a mental model for quality.** Even if you don't implement it formally, thinking in terms of Helpful/Harmless/Honest clarifies what "good" means.

## Practical Considerations: The Hidden Challenges[#](#practical-considerations-the-hidden-challenges)

### Data Quality > Quantity[#](#data-quality-quantity)

Post-training is dominated by data quality. A few thousand carefully curated examples beat tens of thousands of mediocre ones.

data\_quality\_impact.pycpu-only

```
def analyze_data_quality_tradeoff():
  """
  Demonstrate why data quality dominates in post-training.
  """

  datasets = {
      "Low quality, large": {
          "size": 50000,
          "quality_per_example": 0.4,
          "estimated_effectiveness": 0.4 * 50000 / 1000,  # Normalized
          "cost": 50000 * 0.1,  # Cheap annotation
          "model_improvement": 0.35,
          "description": "Web-scraped instruction data, minimal curation"
      },
      "Medium quality, medium": {
          "size": 10000,
          "quality_per_example": 0.7,
          "estimated_effectiveness": 0.7 * 10000 / 1000,
          "cost": 10000 * 1.0,  # Moderate cost
          "model_improvement": 0.65,
          "description": "Crowdsourced, some quality control"
      },
      "High quality, small": {
          "size": 2000,
          "quality_per_example": 0.95,
          "estimated_effectiveness": 0.95 * 2000 / 1000,
          "cost": 2000 * 10.0,  # Expert annotation
          "model_improvement": 0.82,
          "description": "Expert-written, heavily curated"
      }
```

### Catastrophic Forgetting[#](#catastrophic-forgetting)

A subtle but critical problem: post-training can make the model *forget* capabilities it learned during pretraining.

### Overfitting to Human Preferences[#](#overfitting-to-human-preferences)

Humans are inconsistent, biased, and sometimes wrong. A model trained to maximize human preference can overfit to these biases.

Examples:

* **Preference for verbosity:** Humans rate longer answers higher, so models learn to ramble
* **Reward hacking for safety:** Models learn to *seem* safe (disclaimers, hedging) rather than be safe
* **Preference for confidence:** Humans prefer confident answers, so models hallucinate confidently
* **Demographic biases:** If annotators are homogeneous, the model inherits their blind spots

**Mitigation strategies:**

* Collect preference data from diverse annotators
* Use multiple reward models and ensemble them
* Include "I don't know" and disagreement examples
* Explicit bias detection and correction
* Constitutional AI for principle-based alignment (harder to hack)

### The Cold-Start Problem[#](#the-cold-start-problem)

Early in post-training, you don't have preference data. How do you initialize SFT?

**Options:**

1. **Manual curation:** Experts write examples (expensive, works well)
2. **Distillation:** Use outputs from a better model (but you don't have one)
3. **Synthetic data:** Use templates or other models to generate examples
4. **Crowd-sourcing:** Pay annotators (noisy, but scalable)
5. **Self-play:** Have the model generate both examples and preferences

Each has tradeoffs. Most labs use a combination.

## Case Study: Llama 2's Post-Training[#](#case-study-llama-2s-post-training)

Meta's Llama 2 paper (2023) provides a transparent view into post-training at scale:

**Llama 2 (7B-70B base models)**

**Stage 1: SFT**

* Collected ~27.5K instruction-response pairs
* Mix of publicly available data + internally created
* Moderate diversity (QA, coding, creative writing, reasoning, safety)
* Trained for 2 epochs with a learning rate of 2e-5

**Results after SFT:**

* Much better instruction-following
* Still had gaps in safety and preference alignment

**Stage 2: RLHF**

* Trained reward models to score helpfulness and safety
* Used PPO for ~100K iterations
* Balanced helpfulness vs. harmlessness through reward scaling

**Key decisions:**

* **Separate RM for safety:** One RM for helpfulness, one for safety, then combined
* **Red-teaming:** Explicitly generated adversarial examples to test safety
* **Early stopping:** Careful monitoring to prevent capability regression
* **RL β tuning:** Adjusted the KL weight to balance RL progress vs. distance from SFT

**Results:**

* Llama 2-Chat (instruct version) was preferred to other open models
* Safety significantly improved vs. base model
* Small capability regression on raw benchmarks, but better overall usefulness

**Key lesson:** Even with massive scale (70B parameters), the post-training recipe is similar to smaller models. The challenge is engineering stability and preventing catastrophic forgetting at large scale.

### The Hyperparameter Jungle[#](#the-hyperparameter-jungle)

Post-training involves many knobs:

**SFT knobs:**

* Learning rate (1e-5 to 5e-4)
* Batch size (16 to 128)
* Epochs (1-5)
* Warmup steps (1-5% of total)
* Weight decay (0.0 to 0.1)

**RLHF knobs:**

* Number of RL epochs (2-4)
* PPO clip epsilon (0.1-0.3)
* β (KL weight, 0.01 to 0.1)
* Adam learning rate (1e-6 to 1e-5)
* RM score temperature

Searching this space is expensive. Most labs use heuristics or prior work rather than grid search.

---

## Summary: The Post-Training Mindset[#](#summary-the-post-training-mindset)

Post-training requires a different mental model than pretraining:

| Aspect | Pretraining | Post-Training |
| --- | --- | --- |
| **Objective** | Next token prediction (universal) | Helpfulness/harmlessness (custom) |
| **Data** | Massive, raw, cheap | Smaller, curated, expensive |
| **Optimization** | Straightforward SGD | Complex multi-stage pipeline |
| **Metric** | Perplexity, downstream tasks | Human preference, safety, specific abilities |
| **Failure mode** | Poor generalization | Overfitting, capability loss, reward hacking |
| **Key insight** | Scale matters | Alignment matters |

Post-training is where capability meets intent. A capable model without post-training is a broken tool. A post-trained model is a collaborator.

In the next lessons, we'll build the technical skills: designing instruction data, implementing reward models, running RLHF, and debugging post-training failures.

---

*Next up: **Instruction Data & Data Formats** -- Understanding what SFT actually learns from different data structures, and how to design instruction datasets that produce generalizable behavior.*

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. InstructGPT used 13K SFT examples and 33K preference judgments. Estimate the total annotation cost at 5 USD per instruction example and 8 USD per preference pair. Which stage costs more?
2. A 70B model in BF16 needs approximately 140 GB for weights alone. If you run SFT with Adam (2 FP32 states per parameter plus FP32 master weights), estimate the total GPU memory for weights + optimizer states. How many 80 GB A100s does that require at minimum?
3. A base model scores 0.65 on MMLU before SFT and 0.62 after. Calculate the regression percentage. If your threshold is 3% maximum regression, does this run pass?