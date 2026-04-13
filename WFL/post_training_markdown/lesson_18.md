In this tutorial, you will implement the two-stage Constitutional AI pipeline (SL-CAI and RL-CAI), design constitutional principles for a specific domain, and estimate the cost savings compared to standard RLHF with human labels.

Constitutional AI replaces human preference labels with AI-generated labels guided by a written set of principles (the constitution). The model critiques and revises its own outputs, then uses its own judgments as training signal.

## Prerequisites Refresher[#](#prerequisites-refresher)

▶Need a reminder on RLHF?

RLHF (Reinforcement Learning from Human Feedback) trains a model to maximize a reward function learned from human preferences:

1. **Collect preference data:** Humans compare pairs of model outputs and select the better one
2. **Train reward model:** Learn a function that predicts human preferences
3. **RL training:** Use PPO or similar to maximize expected reward while staying close to the original model

**The bottleneck:** Collecting human preferences is expensive (~$1-2 per comparison), slow (weeks to months), and scales poorly with model capability.

▶What is a constitution in the AI sense?

A constitution is a set of written principles that guide behavior. In Constitutional AI, the constitution is:

* A list of explicit, text-based principles (e.g., "Avoid providing information that could be used to cause harm")
* **Not** a neural network or learned component — just English text
* Used to prompt the model to critique and compare outputs
* Designed to capture human values (helpfulness, harmlessness, honesty)

Think of it like the Constitution of the United States — it's a written document that guides decision-making.

▶What's self-critique? Why would it work?

Self-critique means asking the model to evaluate its own outputs against principles. Why this works:

1. **Modern language models understand critique:** They've been trained on text about criticism, evaluation, and judgment
2. **Principles are interpretable:** Unlike an opaque reward model, the model can reason about why something violates a principle
3. **Scales with model capability:** Better models critique better, instead of becoming harder to evaluate

The key insight: **You don't need humans to label preferences. You just need humans to write down their values (the constitution).**

## The Problem with Human Feedback[#](#the-problem-with-human-feedback)

human\_feedback\_limitations.pycpu-only

```
def analyze_human_feedback_limitations():
```

## The Constitutional AI Pipeline[#](#the-constitutional-ai-pipeline)

RL-CAI (Reinforcement Learning)

SL-CAI (Supervised Learning)

Harmful Prompt

Initial Response  
(possibly harmful)

Self-Critique  
'This response violates principle X'

Revision  
'Here's a better response...'

Fine-tune on  
(prompt, revision) pairs

Prompt

Generate  
Multiple Responses

AI Compares Responses  
Using Constitution

Preference Labels  
(no humans!)

RLHF Training

cai\_pipeline\_overview.pycpu-only

```
def explain_cai_pipeline():
```

## The Constitution: Principles That Guide the AI[#](#the-constitution-principles-that-guide-the-ai)

constitution\_principles.pycpu-only

```
def example_constitutional_principles():
```

## Implementing Self-Critique[#](#implementing-self-critique)

self\_critique\_implementation.pycpu-only

```
import numpy as np
```

critique\_revision\_loop.pycpu-only

```
import numpy as np
```

## Designing Constitutional Principles[#](#designing-constitutional-principles)

Yes

No

Yes

No

Start: Define your domain

Identify harm types

List positive values  
(helpfulness, etc.)

Draft 10-20 principles

Test on sample prompts

Failures?

Analyze failure modes

Add/refine principles

Deploy with monitoring

Red-team actively

New failures?

Iterate periodically

design\_principles.pycpu-only

```
def design_constitutional_principles():
```

## RL-CAI: AI Feedback for Preference Learning[#](#rl-cai-ai-feedback-for-preference-learning)

ai\_feedback.pycpu-only

```
import numpy as np
```

preference\_generation.pycpu-only

```
import numpy as np
```

## The Full CAI Training Algorithm[#](#the-full-cai-training-algorithm)

cai\_algorithm.pycpu-only

```
def cai_training_algorithm():
```

## Self-Improvement Dynamics[#](#self-improvement-dynamics)

One of the most interesting properties of Constitutional AI is how it creates a **self-improving loop**. Unlike traditional RLHF (where the humans labeling data are a fixed bottleneck), CAI has a positive feedback dynamic:

Larger/Better Base Model

Better Self-Critique

Higher Quality Revisions

Better Preference Labels

Stronger RLHF Training

Improved Final Model

Better Constitution

More Red-Teaming

New Principles

self\_improvement\_dynamics.pycpu-only

```
import numpy as np
```

## Break It: When Self-Critique Fails[#](#break-it-when-self-critique-fails)

break\_it\_self\_critique.pycpu-only

```
import numpy as np
```

## CAI vs RLHF: Empirical Comparison[#](#cai-vs-rlhf-empirical-comparison)

cai\_vs\_rlhf.pycpu-only

```
import numpy as np
```

## Scale Thought Experiment[#](#scale-thought-experiment)

| Aspect | Standard RLHF | Constitutional AI |
| --- | --- | --- |
| **Labels for 7B model** | 50K human | 0 human (AI-generated) |
| **Labels for 70B model** | 200K+ human | 0 human (AI-generated) |
| **Cost per iteration** | $50K-200K | $1K-5K (compute only) |
| **Iteration speed** | Weeks | Days |
| **Harmlessness** | Good | Better |
| **Helpfulness** | Good | Comparable |
| **Red-team robustness** | Moderate | Higher |
| **Human bottleneck** | Yes | No |

scale\_thought\_experiment.pycpu-only

```
def scaling_comparison():
```

## Production Reality[#](#production-reality)

**Anthropic's deployment:**

* Constitutional AI is core to Claude's training
* Multiple iterations of principle refinement
* Combines SL-CAI and RL-CAI in practice
* Red-teaming to discover principle gaps
* Regular constitution updates as new issues emerge

**Key practices:**

1. Start with broad principles, refine based on failures
2. Include both "avoid harm" and "be helpful" principles
3. Use diverse principle sets for different critique rounds
4. Validate AI preferences against held-out human labels
5. Iterate: train, red-team, update constitution, repeat

**Limitations in production:**

* Constitution design requires expertise and iteration
* Some harms are hard to specify as principles
* Adversarial users may find principle gaps
* Still benefits from some human oversight

production\_tips.pycpu-only

```
def production_tips():
```

## Checkpoint Questions[#](#checkpoint-questions)

1. Estimate the cost of generating 100K AI preference labels at $0.001 per label versus 100K human labels at $1.00 per label. If the AI labels achieve 78% agreement with human ground truth and human inter-annotator agreement is 75%, calculate the effective quality gap as a percentage.
2. You are building a medical chatbot and need to design a constitution. Write three principles that balance safety (do not delay emergency care) with helpfulness (answer health questions). For each principle, describe one scenario where it would correctly flag a response and one where it might produce a false positive.
3. After two iterations of CAI training, your model has a red-team success rate of 12%. After a third iteration with an improved base model, the rate drops to 5%. Estimate the self-improvement ceiling if each iteration improves critique quality by 30% of the remaining gap toward perfect (0%) red-team success.

## Research Hooks[#](#research-hooks)

**Self-critique limits:**
When does self-critique fail? Models can't critique what they don't understand. Current research explores using external tools (search, verification) to augment self-critique.

**Constitutional interpretability:**
Can we understand which principles the model is using? Some work on analyzing attention to constitutional content during critique.

**Multi-objective constitutions:**
Balancing harmlessness and helpfulness is hard. How do we design constitutions that navigate trade-offs well? Active research area.

**Recursive improvement:**
If a model can critique and revise, can it improve its own constitution? Early experiments in "constitutional revision" are promising but raise meta-alignment questions.

---

*Next up: RLAIF takes the AI feedback idea further. Instead of just preference labels, can we replace the entire human feedback pipeline with AI?*