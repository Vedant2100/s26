In this tutorial, you will systematically analyze outputs from a trained coding assistant, build a failure taxonomy with root-cause diagnoses, and create a prioritized improvement plan.

With the first alignment pipeline complete and the model trained, the next step is systematic evaluation: identifying what is broken and designing targeted fixes.

## Prerequisites Refresher[#](#prerequisites-refresher)

Before diving into evaluation, let's recap the key concepts from earlier Track 4 lessons.

▶SFT (Supervised Fine-Tuning) Review

**What it does:** Trains model to imitate high-quality instruction-following examples.

**The training objective:**

```
Loss = -log P(output | input)
```

We're maximizing the likelihood of seeing the correct output given the instruction.

**Key decisions:**

* Data quality matters more than size (100 high-quality > 10k low-quality)
* Category balance affects performance
* Example diversity improves generalization

**Common pitfalls:**

* Too much training → memorization
* Too little → poor performance
* Mismatch between training and test distribution

▶Reward Modeling Review

**What it does:** Trains a model to predict which response humans prefer.

**The training objective:**

```
Loss = -log(σ(r_chosen - r_rejected))
```

We want the chosen response to have higher reward than rejected.

**Key decisions:**

* Preference pairs must have clear margin
* Labeling strategy (human vs. LLM vs. automated)
* Diversity of preferences captured

**Common pitfalls:**

* Labeling noise → wrong preferences learned
* Lack of diversity → narrow preference model
* Reward hacking → model exploits loopholes

▶DPO (Direct Preference Optimization) Review

**What it does:** Directly optimizes language model for human preferences without explicit reward model.

**The training objective:**

```
Loss = -log(σ(β * log(P_π(y_w | x) / P_ref(y_w | x)) -
             β * log(P_π(y_l | x) / P_ref(y_l | x))))
```

Maximize likelihood ratio for preferred response, minimize for rejected.

**Key hyperparameters:**

* **β (beta):** Controls strength of preference signal (0.1-1.0)
* **learning rate:** DPO typically uses 1e-6 to 5e-6
* **num\_epochs:** Usually 1-3 (avoids overfitting)

**Common pitfalls:**

* β too high → policy barely moves from reference (weak learning signal, vanishing gradients)
* β too low → KL divergence grows, policy drifts too far from reference
* Too many epochs → distribution divergence

## The Systematic Evaluation Framework[#](#the-systematic-evaluation-framework)

Understanding your baseline performance is crucial before iteration.

Outputs

Evaluation Layers

Quantitative  
(Benchmarks)

Qualitative  
(Manual Review)

Diagnostic  
(Root Cause)

Causal  
(A/B Testing)

Scores

Patterns

Hypotheses

Evidence

### Layer 1: Quantitative Evaluation[#](#layer-1-quantitative-evaluation)

quantitative\_eval.pycpu-only

```
import json
from typing import Dict, List

def quantitative_evaluation():
  """
  Quantitative evaluation on multiple benchmarks.
  """
  print("Quantitative Evaluation Framework")
  print("=" * 70)
  print()

  # Simulated benchmark results
  benchmarks = dict(
      HumanEval=dict(
          baseline=0.08,
          sft_model=0.12,
          dpo_model=0.15,
          n_problems=164,
          description="Python coding problems, pass@1",
      ),
      MBPP=dict(
          baseline=0.06,
          sft_model=0.10,
          dpo_model=0.13,
          n_problems=427,
          description="Mostly basic Python problems",
      ),
      CodeQA=dict(
          baseline=0.15,
          sft_model=0.22,
```

### Layer 2: Qualitative Pattern Analysis[#](#layer-2-qualitative-pattern-analysis)

qualitative\_patterns.pycpu-only

```
from collections import Counter
from typing import List, Dict, Tuple

def qualitative_pattern_analysis():
  """
  Systematic analysis of model behavior patterns.
  """
  print("Qualitative Pattern Analysis")
  print("=" * 70)
  print()

  # Simulated manual review data
  review_data = dict(
      total_reviewed=150,
      success=106,
      partial_success=22,
      failure=22,
  )

  print("OVERVIEW:")
  print("-" * 70)
  print()

  for category, count in review_data.items():
      if category != "total_reviewed":
          pct = (count / review_data["total_reviewed"]) * 100
          print("  %-20s %3d (%5.1f%%)" % (category, count, pct))

  print()
  print("BEHAVIOR PATTERNS:")
```

## Step 1: Systematic Output Analysis[#](#step-1-systematic-output-analysis)

output\_analysis.pycpu-only

```
import numpy as np

def systematic_output_analysis():
  """
  Framework for analyzing model outputs systematically.
  """
  np.random.seed(42)

  print("Systematic Output Analysis")
  print("=" * 60)
  print()

  print("STEP 1: Sample Strategically")
  print("-" * 50)
  print()

  sampling_strategy = dict(
      Random_sample=dict(
          size="50 examples",
          purpose="Unbiased view of typical performance",
          method="Random from test set",
      ),
      Failure_focused_sample=dict(
          size="30 examples",
          purpose="Understand common failure modes",
          method="Cases where model got wrong answer or low score",
      ),
      Edge_case_sample=dict(
          size="20 examples",
          purpose="Find boundary failures",
```

output\_review\_template.pycpu-only

```
def output_review_template():
  """
  Template for reviewing individual model outputs.
  """
  print("Output Review Template")
  print("=" * 60)
  print()

  template = '''
OUTPUT REVIEW FORM
==================

Example ID: _______________
Task Category: [ ] Algorithm  [ ] Data Structure  [ ] Debug  [ ] Other

PROMPT:
[Paste the input prompt here]

MODEL OUTPUT:
[Paste the model response here]

EVALUATION:

1. CORRECTNESS (1-5)
 [ ] 1 - Completely wrong / doesn't compile
 [ ] 2 - Major bugs, wrong approach
 [ ] 3 - Works for basic cases, fails edge cases
 [ ] 4 - Correct with minor issues
 [ ] 5 - Fully correct, handles edge cases
```

## The Post-Training Analysis Framework[#](#the-post-training-analysis-framework)

Artifacts

Iteration Loop

Trained Model

Evaluate  
(Systematic)

Failure Analysis  
(Categorize)

Diagnosis  
(Root Cause)

Plan  
(Prioritize)

Implement  
(Fix)

Benchmark Scores

Failure Taxonomy

Data Gaps

Improvement Plan

## Step 2: Failure Categorization[#](#step-2-failure-categorization)

failure\_taxonomy.pycpu-only

```
import numpy as np

def failure_taxonomy():
  """
  Build a taxonomy of failure modes.
  """
  np.random.seed(42)

  print("Failure Taxonomy for Coding Assistants")
  print("=" * 60)
  print()

  # Simulated analysis results
  failure_categories = dict(
      CORRECTNESS_FAILURES=dict(
          Syntax_errors=dict(
              frequency=0.05,
              severity="High",
              example="Missing colon, unmatched brackets",
              root_cause="Likely tokenization or rare syntax patterns",
          ),
          Logic_errors=dict(
              frequency=0.13,
              severity="High",
              example="Off-by-one, wrong operator",
              root_cause="Insufficient algorithmic training",
          ),
          Missing_edge_cases=dict(
              frequency=0.18,
              severity="Medium",
```

visualize\_failures.pycpu-only

```
import numpy as np

def visualize_failure_distribution():
  """
  Visualize failure patterns for analysis.
  """
  np.random.seed(42)

  print("Failure Distribution Analysis")
  print("=" * 60)
  print()

  # Failure rates by category
  categories = dict(
      Algorithm=dict(success=0.65, failures=["logic", "efficiency", "edge_case"]),
      Data_Structure=dict(success=0.72, failures=["logic", "edge_case"]),
      Debugging=dict(success=0.58, failures=["incomplete", "wrong_fix"]),
      String_Processing=dict(success=0.78, failures=["edge_case"]),
      File_IO=dict(success=0.45, failures=["hallucination", "wrong_api"]),
  )

  print("SUCCESS RATE BY TASK CATEGORY:")
  print("-" * 50)
  print()

  for category, data in categories.items():
      bar_len = int(data["success"] * 40)
      bar = "*" * bar_len + " " * (40 - bar_len)
      print("  %-20s [%s] %.0f%%" % (category.replace("_", " "), bar, data["success"] * 100))
      print("    Main failures: %s" % ", ".join(data["failures"]))
```

## Step 3: Iteration Methodology[#](#step-3-iteration-methodology)

The real power of systematic evaluation is using it to drive iteration. Here's a proven methodology:

Key Artifacts

Iteration Cycle

No

Yes

Evaluate  
(Measure)

Analyze  
(Diagnose)

Plan  
(Prioritize)

Implement  
(Improve)

Verify  
(Test)

Good?

Document & Ship

Benchmark scores

Root cause map

Prioritized backlog

New training run

Before/after comparison

iteration\_framework.pycpu-only

```
def iteration_methodology():
  """
  Systematic iteration framework with concrete steps.
  """
  print("Alignment Model Iteration Methodology")
  print("=" * 70)
  print()

  print("ITERATION CYCLES AND TIMELINE:")
  print("-" * 70)
  print()

  cycles = [
      dict(
          cycle="Iteration 0 (Baseline)",
          time="~4 hours",
          goal="Establish baseline performance",
          actions=[
              "Train base model + SFT",
              "Evaluate on HumanEval/MBPP",
              "Manual review of 50 outputs",
          ],
      ),
      dict(
          cycle="Iteration 1 (First improvement)",
          time="~8-10 hours",
          goal="Target top 2-3 failure modes",
          actions=[
              "Collect data for gaps",
              "Retrain SFT with augmented data",
```

## Step 3: Data Diagnosis[#](#step-3-data-diagnosis)

data\_gap\_analysis.pycpu-only

```
import numpy as np

def diagnose_data_gaps():
  """
  Analyze training data to find gaps causing failures.
  """
  np.random.seed(42)

  print("Training Data Gap Analysis")
  print("=" * 60)
  print()

  print("METHODOLOGY:")
  print("-" * 50)
  print("1. Map failures to potential data gaps")
  print("2. Analyze training data distribution")
  print("3. Identify under-represented categories")
  print("4. Propose data augmentation strategy")
  print()

  print("SFT DATA ANALYSIS (CodeAlpaca-20k):")
  print("-" * 50)
  print()

  # Simulated analysis of training data
  sft_distribution = dict(
      Basic_Python=0.35,
      Algorithms=0.15,
      Data_Structures=0.12,
      String_Processing=0.18,
```

continuous\_improvement.pycpu-only

```
from typing import List, Dict, Tuple

def continuous_improvement_strategy():
  """
  Long-term strategy for continuous improvement.
  """
  print("Continuous Improvement Strategy")
  print("=" * 70)
  print()

  print("IMPROVEMENT LEVERS (by effort/impact):")
  print("-" * 70)
  print()

  levers = [
      dict(
          lever="Data quality filter",
          effort="Low (1-2 hours)",
          impact="Medium (+3-5%)",
          how="Remove low-quality examples from training set",
          risk="Low",
      ),
      dict(
          lever="Category rebalancing",
          effort="Medium (2-4 hours)",
          impact="Medium (+5-8%)",
          how="Oversample failing categories",
          risk="Medium (can cause overfitting)",
      ),
      dict(
```

preference\_data\_diagnosis.pycpu-only

```
def diagnose_preference_data():
  """
  Analyze DPO preference data for issues.
  """
  print("Preference Data Diagnosis")
  print("=" * 60)
  print()

  print("PREFERENCE PAIR QUALITY ANALYSIS:")
  print("-" * 50)
  print()

  quality_issues = dict(
      Margin_too_small=dict(
          frequency="15% of pairs",
          example="Both responses correct, only style differs",
          impact="Weak learning signal",
          fix="Filter for pairs with clear quality difference",
      ),
      Wrong_label=dict(
          frequency="5% of pairs",
          example="'Rejected' is actually better than 'chosen'",
          impact="Teaches wrong preferences",
          fix="Re-label with GPT-4 or human review",
      ),
      Off_topic_chosen=dict(
          frequency="8% of pairs",
          example="Chosen response doesn't address the question",
          impact="Model learns to be unhelpful",
          fix="Filter for task completion in 'chosen'",
```

## Step 4: Improvement Planning[#](#step-4-improvement-planning)

Let me revise the improvement plan template with real data and prioritization:

improvement\_plan.pycpu-only

```
def create_improvement_plan():
  """
  Prioritized improvement plan based on analysis.
  """
  print("Improvement Plan")
  print("=" * 60)
  print()

  print("PRIORITIZATION FRAMEWORK:")
  print("-" * 50)
  print()
  print("Impact = Failure frequency x Severity")
  print("Effort = Data collection + Training time")
  print("Priority = Impact / Effort")
  print()

  improvements = [
      dict(
          id=1,
          name="Add File I/O examples to SFT",
          impact="High (55% failure, common task)",
          effort="Low (easy to collect)",
          priority="P0 - Do first",
          details=dict(
              data_needed="500+ file I/O instruction-output pairs",
              sources=["Python docs", "Real-world scripts", "SO answers"],
              expected_improvement="File I/O: 45% -> 70% success",
          ),
      ),
      dict(
```

prioritization\_matrix.pycpu-only

```
import math

def prioritization_matrix():
  """
  Use impact/effort matrix to prioritize improvements.
  """
  print("Improvement Prioritization Matrix")
  print("=" * 70)
  print()

  print("QUADRANT ANALYSIS:")
  print("-" * 70)
  print()

  improvements = [
      dict(name="Add File I/O examples", impact=9, effort=2),
      dict(name="Fix edge case handling", impact=8, effort=4),
      dict(name="Improve hallucination filtering", impact=6, effort=8),
      dict(name="Category rebalancing", impact=5, effort=2),
      dict(name="Advanced debugging patterns", impact=7, effort=9),
      dict(name="Hyperparameter tuning", impact=3, effort=1),
  ]

  # Calculate score
  for imp in improvements:
      imp["score"] = imp["impact"] / imp["effort"]

  # Sort by score
  sorted_imps = sorted(improvements, key=lambda x: -x["score"])
```

data\_collection\_plan.pycpu-only

```
def data_collection_plan():
  """
  Concrete plan for collecting missing data.
  """
  print("Data Collection Plan")
  print("=" * 60)
  print()

  print("TARGET 1: FILE I/O EXAMPLES (500+ pairs)")
  print("-" * 50)
  print()

  file_io_plan = '''
Sources:
1. Python documentation examples (100 pairs)
 - pathlib operations
 - open/read/write patterns
 - CSV, JSON, pickle handling

2. Real-world scripts (200 pairs)
 - GitHub search: "python file" "def read"
 - Filter for clean, instructive examples

3. StackOverflow (150 pairs)
 - Questions tagged [python] [file-io]
 - Extract question as prompt, accepted answer as output

4. Synthetic generation (50 pairs)
 - Use GPT-4 to generate diverse file I/O tasks
 - Verify outputs are correct
```

## Step 5: A/B Testing and Validation[#](#step-5-ab-testing-and-validation)

Before committing to improvements, validate with A/B testing:

ab\_testing\_validation.pycpu-only

```
def ab_testing_framework():
  """
  A/B testing framework for validating improvements.
  """
  print("A/B Testing Framework for Alignment Models")
  print("=" * 70)
  print()

  print("WHEN TO A/B TEST:")
  print("-" * 70)
  print()

  scenarios = [
      dict(
          change="Adding new training data",
          why_test="New data might have quality issues or biases",
          baseline="Model trained on original data",
          variant="Model trained on original + new data",
      ),
      dict(
          change="Changing hyperparameters",
          why_test="Different params might hurt performance",
          baseline="Previous best hyperparameters",
          variant="New hyperparameter setting",
      ),
      dict(
          change="Filtering/deduplicating data",
          why_test="Removing data might help or hurt",
          baseline="Full dataset",
          variant="Filtered dataset",
```

## Step 6: Evaluation Checklist[#](#step-6-evaluation-checklist)

evaluation\_checklist.pycpu-only

```
def complete_evaluation_checklist():
  """
  Complete evaluation checklist for alignment projects.
  """
  print("Complete Alignment Project Evaluation Checklist")
  print("=" * 70)
  print()

  checklist = """
PRE-TRAINING CHECKLIST
======================

[ ] Data Quality
  [ ] Reviewed random sample of SFT data (20+ examples)
  [ ] Verified instruction-output format is consistent
  [ ] Checked for toxic/harmful content
  [ ] Verified code examples compile/run
  [ ] Analyzed category distribution

[ ] Preference Data Quality
  [ ] Verified chosen > rejected for sampled pairs
  [ ] Checked for label noise
  [ ] Ensured diversity of prompts
  [ ] Verified both responses are non-trivial

[ ] Model Selection
  [ ] Verified base model fits hardware
  [ ] Tested basic generation works
  [ ] Documented model's known limitations
```

## Extended Exercise: Build Your Own Evaluation Plan[#](#extended-exercise-build-your-own-evaluation-plan)

Here's a concrete exercise to practice evaluation design:

▶Exercise: Design an Evaluation for Your Capstone Model

**Scenario:** You've just trained a coding assistant on CodeAlpaca (20k examples) with SFT and DPO. Now you need to evaluate it before deciding on next steps.

**Part 1: Quantitative Design (15 min)**

1. List 3-5 benchmarks you'd run
2. For each, explain WHY it matters (not just "it measures performance")
3. What baseline would you compare to?
4. How would you weight different benchmarks if they conflict?

**Part 2: Qualitative Design (20 min)**

1. How many examples would you review manually? Why that number?
2. Describe your sampling strategy (random? failure-focused? category-stratified?)
3. Design a review form for each example (5-10 questions)
4. How would you handle disagreements/ambiguities?

**Part 3: Root Cause Diagnosis (20 min)**

1. Based on typical coding assistant failure patterns, list 5 potential issues
2. For each, write a diagnostic question (how you'd detect it)
3. For each, propose a fix (data augmentation? hyperparameter change? training procedure?)

**Part 4: Prioritization (10 min)**

1. Create a 2x2 matrix: Impact vs. Effort
2. Place your 5 potential issues on it
3. Rank them for your iteration roadmap
4. Estimate total time for iterations 0-2

**Expected output:** A 2-3 page evaluation and iteration plan document you could give to a colleague.

## Capstone Deliverables[#](#capstone-deliverables)

capstone\_deliverables.pycpu-only

```
def capstone_deliverables():
  """
  Final deliverables for the Track 4 Capstone.
  """
  print("Track 4 Capstone: Final Deliverables")
  print("=" * 70)
  print()

  print("REQUIRED DELIVERABLES:")
  print("-" * 50)
  print()

  deliverables = [
      dict(
          item="1. Trained Model Checkpoints",
          description="SFT and DPO model checkpoints",
          files=["./sft-code-assistant/", "./dpo-code-assistant/"],
          verification="Can load and generate from both models",
      ),
      dict(
          item="2. Training Logs",
          description="TensorBoard or W&B logs showing training curves",
          files=["./logs/sft/", "./logs/dpo/"],
          verification="Loss curves show proper convergence",
      ),
      dict(
          item="3. Benchmark Results",
          description="HumanEval scores for baseline, SFT, and DPO models",
          files=["evaluation_results.json"],
          verification="DPO model shows improvement over baseline",
```

## Common Pitfalls in Evaluation & Iteration[#](#common-pitfalls-in-evaluation-iteration)

common\_pitfalls.pycpu-only

```
def common_evaluation_pitfalls():
  """
  Mistakes that alignment engineers make during evaluation.
  """
  print("Common Pitfalls in Model Evaluation & Iteration")
  print("=" * 70)
  print()

  pitfalls = [
      dict(
          pitfall="Evaluating only on your test set",
          why_bad="You might overfit to evaluation metrics",
          what_to_do="Use multiple benchmarks, test on real-world prompts",
          example="High HumanEval but poor real-world performance",
      ),
      dict(
          pitfall="Cherry-picking examples",
          why_bad="Creates false sense of improvement",
          what_to_do="Use systematic sampling, blind comparison",
          example="Show only successful outputs to justify model release",
      ),
      dict(
          pitfall="Comparing against outdated baseline",
          why_bad="Doesn't show what actually improved",
          what_to_do="Always compare current vs previous iteration",
          example="v1 model gets 60%, v2 gets 62%, but you compare v2 to v0 (45%)",
      ),
      dict(
          pitfall="Ignoring failure modes you can't quantify",
          why_bad="Misses important quality issues",
```

## Scale Thought Experiment[#](#scale-thought-experiment)

scale\_iteration.pycpu-only

```
def scale_iteration_process():
  """
  How does the iteration process scale at different model sizes?
  """
  print("Scaling the Iteration Process")
  print("=" * 60)
  print()

  scales = dict(
      Capstone_1B_model=dict(
          data_collection="Hours (manual curation)",
          training_time="2-3 hours",
          eval_time="30 minutes",
          iteration_cycle="1 day",
          team_size="1 person",
      ),
      Startup_7B_model=dict(
          data_collection="Days (contractors + filtering)",
          training_time="12-24 hours",
          eval_time="2-4 hours",
          iteration_cycle="1 week",
          team_size="2-5 people",
      ),
      Lab_70B_model=dict(
          data_collection="Weeks (dedicated team)",
          training_time="Days to weeks",
          eval_time="12-24 hours",
          iteration_cycle="2-4 weeks",
          team_size="10-30 people",
      ),
```

## Building an Evaluation Dashboard[#](#building-an-evaluation-dashboard)

For larger projects, track improvements systematically:

evaluation\_dashboard.pycpu-only

```
def evaluation_dashboard():
  """
  Build a dashboard to track model improvements over iterations.
  """
  print("Evaluation Dashboard Template")
  print("=" * 70)
  print()

  print("KEY METRICS TO TRACK:")
  print("-" * 70)
  print()

  # Simulated iteration data
  iterations = dict(
      Iteration_0_Baseline=dict(
          date="Day 1",
          sft_loss=2.3,
          dpo_loss=0.68,
          humaneval=0.08,
          mbpp=0.06,
          manual_quality="3.2/5",
          top_failure="Edge cases (20%)",
          model_size="1B",
      ),
      Iteration_1_File_IO=dict(
          date="Day 3",
          sft_loss=2.1,
          dpo_loss=0.62,
          humaneval=0.12,
          mbpp=0.10,
```

## Handling Difficult Cases[#](#handling-difficult-cases)

Some improvements are harder to validate. Here's how to approach them:

difficult\_improvements.pycpu-only

```
def difficult_improvements():
  """
  How to validate improvements that are hard to quantify.
  """
  print("Handling Difficult Improvements")
  print("=" * 70)
  print()

  cases = [
      dict(
          improvement="Reducing hallucinations (fake APIs)",
          challenge="Hard to automate detection",
          solution="Manual sampling + automated checks (does import work?)",
          validation="Random 20 examples, check for non-existent functions",
      ),
      dict(
          improvement="Better code readability",
          challenge="Subjective metric",
          solution="Use multiple reviewers, aggregate scores",
          validation="3 reviewers, score 0-5, average their ratings",
      ),
      dict(
          improvement="Better generalization to new task types",
          challenge="Need diverse test set",
          solution="Collect tasks outside training distribution",
          validation="Create 10 novel tasks, evaluate on those",
      ),
      dict(
          improvement="Faster/shorter code",
          challenge="Multiple 'correct' solutions",
```

## Checkpoint Questions[#](#checkpoint-questions)

1. Failure rates: logic 15%, edge cases 20%, hallucinated APIs 5%, incomplete 8%. Training: 3% File I/O, 55% failure rate. File I/O failures per 100 examples? New % after adding 500 to 20K?
2. Three iterations: 0 (4h, baseline), 1 (10h, +8%), 2 (15h, +5%). Cumulative improvement? Marginal improvement/hour? When to stop?
3. Outputs 180 tokens. DPO chosen=165, rejected=120. Length ratio? If above 1.3, what did DPO teach? Fix?

## Research Hooks[#](#research-hooks)

**Automated failure analysis:**
Can we automate the failure categorization process? LLMs analyzing their own failures to suggest improvements.

**Self-improving models:**
Can a model identify its own weaknesses and suggest training data to address them? Meta-learning for alignment.

**Continuous alignment:**
Instead of discrete iterations, can we have continuous online learning that improves alignment in real-time?

**Multi-objective iteration:**
When improving one dimension (e.g., helpfulness), how do we ensure we don't regress on others (safety)? Pareto-optimal iteration.

**Preference learning dynamics:**
How do human preferences change with model capability? Does the reward model need to evolve as the model improves?

**Failure prediction:**
Can we predict which types of inputs will fail BEFORE evaluation, and prioritize those for improvement?

## Final Iteration Roadmap Template[#](#final-iteration-roadmap-template)

Use this template to plan your own capstone iterations:

your\_iteration\_plan.pycpu-only

```
def your_iteration_roadmap():
  """
  Template for your personal iteration roadmap.
  """
  print("YOUR CAPSTONE ITERATION ROADMAP")
  print("=" * 70)
  print()

  print("ITERATION 0: ESTABLISH BASELINE")
  print("-" * 70)
  baseline = dict(
      goal="Measure starting performance",
      steps=[
          "Train base model + SFT on CodeAlpaca",
          "Run HumanEval and MBPP benchmarks",
          "Manually review 50 examples",
          "Build failure taxonomy",
      ],
      time="4-6 hours",
      success_criteria="Baseline scores recorded, 5+ failure modes identified",
  )
  for key, val in baseline.items():
      print("%s: %s" % (key, val))
  print()

  print("ITERATION 1: TARGET TOP GAPS")
  print("-" * 70)
  iter1 = dict(
      goal="Address 2-3 most impactful failure modes",
      failure_modes=[
```

---

*Congratulations on completing Track 4: Post-Training & Alignment!*

*You've learned not just the theory, but built a working alignment pipeline. The iteration mindset you developed here is the key differentiator between paper-level understanding and production-level capability.*

*Remember: The first version is a starting point. Alignment engineering requires systematic iteration. The difference between mediocre and great engineers isn't intelligence—it's patience, systematic thinking, and willingness to do the unglamorous work of analyzing failures, making targeted improvements, and measuring progress.*

*The frameworks in this lesson scale from your 1B capstone model all the way to frontier 175B+ models. Same principles. Same mindset. Go build something great.*