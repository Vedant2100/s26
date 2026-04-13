In this tutorial, you will wire together a complete alignment pipeline (SFT then DPO) for a coding assistant, estimate memory requirements, and evaluate on HumanEval.

This capstone wires together the components from earlier lessons into a complete alignment pipeline.

## Prerequisites Refresher[#](#prerequisites-refresher)

▶SFT Fundamentals

**What is Supervised Fine-Tuning (SFT)?**

SFT is simple: give the model examples of good behavior, and it learns to imitate them. The loss function is just language modeling loss on the response:

`L_SFT = -log P(response | instruction)`

**Key concepts:**

* **Data format:** Instruction-response pairs in chat template format
* **Optimization:** Minimize next-token prediction error
* **Epochs:** Usually 1-3 (more = overfitting risk)
* **LoRA efficiency:** Only 0.5-1% of parameters are trainable, but 85-95% of performance

**Common mistakes:**

* Using too many epochs on small datasets
* Mismatched prompt formatting between training and inference
* Not validating that loss is decreasing

▶LoRA (Low-Rank Adaptation)

**Why LoRA?**

Instead of tuning all parameters, LoRA adds small trainable matrices to each attention layer:

`W' = W + A * B^T` where `A ∈ R^(d×r)`, `B ∈ R^(d×r)`, `r << d`

**Memory savings:**

* Full fine-tune of 1B model: ~20GB
* LoRA (r=16): ~10GB
* QLoRA (8-bit): ~6GB

**Configuration for this capstone:**

* **r (rank):** 16 (typical range: 8-64)
* **target\_modules:** q\_proj, v\_proj (attention); gate\_proj, up\_proj, down\_proj (FFN)
* **dropout:** 0.05 (regularization)

▶DPO (Direct Preference Optimization)

**How DPO works:**

DPO removes the reward model and optimizes directly on preference pairs. The loss is:

`L_DPO = -log σ(β * log(π(y_w|x) / π_ref(y_w|x)) - β * log(π(y_l|x) / π_ref(y_l|x)))`

where:

* β = KL penalty strength (0.1-0.5 typical)
* y\_w = "winning" (preferred) response
* y\_l = "losing" (less preferred) response
* π\_ref = frozen reference model

**What β does:**

* Low β (0.05): Model can diverge more from reference (higher reward, higher KL)
* High β (0.5): Model stays close to reference (lower reward, lower KL)

**Training strategy:** 1 epoch is usually enough; 2+ risks overfitting.

▶Evaluation Metrics for Code

**HumanEval:**

* 164 Python programming problems from OpenAI
* Metric: pass@1 (does code pass tests on first try?)
* Realistic ceiling for 1-7B models: 10-50%

**MBPP (Mostly Basic Programming Problems):**

* 974 Python problems, less difficult than HumanEval
* Good for seeing if model improved at all

**Manual evaluation:**

* Code formatting and style
* Explanation quality
* Edge case handling
* Efficiency awareness

## The Complete Pipeline[#](#the-complete-pipeline)

Phase 3: Eval

Phase 2: DPO

Phase 1: SFT

Base Model  
(1B params)

Instruction  
Dataset

SFT + LoRA

SFT Model

Preference  
Pairs

DPO Training

Aligned Model

HumanEval

MBPP

Manual Review

## Step 1: Choose Your Base Model[#](#step-1-choose-your-base-model)

model\_selection.pycpu-only

```
def select_base_model():
  """
  Select appropriate base model for the capstone.
  """
  print("Base Model Selection for Coding Assistant")
  print("=" * 60)
  print()

  candidates = dict(
      TinyLlama_1_1B=dict(
          params="1.1B",
          context=2048,
          gpu_memory_sft="~8GB with LoRA",
          gpu_memory_dpo="~12GB",
          strengths="Fast training, fits T4",
          weaknesses="Limited reasoning capacity",
          recommended_for="Learning/experimentation",
      ),
      Phi_2=dict(
          params="2.7B",
          context=2048,
          gpu_memory_sft="~12GB with LoRA",
          gpu_memory_dpo="~20GB",
          strengths="Strong code understanding",
          weaknesses="Needs A100 for DPO",
          recommended_for="Quality-focused training",
      ),
      CodeLlama_7B=dict(
          params="7B",
          context=16384,
```

## Step 2: Prepare Instruction Dataset[#](#step-2-prepare-instruction-dataset)

prepare\_sft\_data.pycpu-only

```
import numpy as np

def prepare_code_instruction_dataset():
  """
  Prepare instruction dataset for coding assistant SFT.
  """
  print("Preparing Instruction Dataset")
  print("=" * 60)
  print()

  print("DATASET SOURCES:")
  print("-" * 50)

  datasets = dict(
      CodeAlpaca_20k=dict(
          size="20K examples",
          format="instruction-output pairs",
          quality="Medium (GPT-3.5 generated)",
          use_case="General code instructions",
      ),
      evol_codealpaca_v1=dict(
          size="110K examples",
          format="instruction-output pairs",
          quality="Higher (evolved complexity)",
          use_case="Diverse coding tasks",
      ),
      CodeContests=dict(
          size="13K problems",
          format="problem-solution pairs",
          quality="High (competition verified)",
```

data\_loading\_script.pycpu-only

```
def data_loading_implementation():
  """
  Implementation code for loading and formatting the dataset.
  """
  print("Data Loading Implementation")
  print("=" * 60)
  print()

  code = '''
# data_utils.py - Dataset preparation for SFT

from datasets import load_dataset
from transformers import AutoTokenizer

def load_code_alpaca():
  """Load and preprocess CodeAlpaca dataset."""
  dataset = load_dataset("sahil2801/CodeAlpaca-20k")
  return dataset["train"]

def format_instruction(example, tokenizer):
  """Convert to chat format."""

  # Build the prompt
  if example.get("input", ""):
      user_content = f"{example['instruction']}\n\nInput: {example['input']}"
  else:
      user_content = example["instruction"]

  # Apply chat template
  messages = [
```

## Step 3: SFT with LoRA[#](#step-3-sft-with-lora)

sft\_training\_script.pycpu-only

```
def sft_training_implementation():
  """
  Complete SFT training script with LoRA.
  """
  print("SFT Training Implementation")
  print("=" * 60)
  print()

  code = '''
# sft_train.py - Supervised Fine-Tuning with LoRA

import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  TrainingArguments,
  Trainer,
  DataCollatorForLanguageModeling,
)
from peft import (
  LoraConfig,
  get_peft_model,
  prepare_model_for_kbit_training,
)
from data_utils import prepare_dataset

# ============================================
# CONFIGURATION
# ============================================
```

sft\_expected\_results.pycpu-only

```
import numpy as np

def sft_expected_results():
  """
  What to expect from SFT training.
  """
  np.random.seed(42)

  print("SFT Expected Results")
  print("=" * 60)
  print()

  print("TRAINING METRICS (TinyLlama on CodeAlpaca):")
  print("-" * 50)
  print()

  # Simulate training curve
  epochs = [1, 2, 3]
  train_losses = [2.4, 1.8, 1.5]
  eval_losses = [2.2, 1.9, 1.7]

  print("%-10s %-15s %-15s" % ("Epoch", "Train Loss", "Eval Loss"))
  print("-" * 40)

  for e, tl, el in zip(epochs, train_losses, eval_losses):
      print("%-10d %-15.2f %-15.2f" % (e, tl, el))

  print()
  print("EXPECTED TIMINGS:")
  print("-" * 50)
```

## Step 4: Prepare Preference Dataset[#](#step-4-prepare-preference-dataset)

preference\_data\_preparation.pycpu-only

```
def prepare_preference_dataset():
  """
  Create preference pairs for DPO training.
  """
  print("Preparing Preference Dataset for DPO")
  print("=" * 60)
  print()

  print("PREFERENCE DATA SOURCES:")
  print("-" * 50)
  print()

  sources = dict(
      Option_A=dict(
          title="Use existing dataset",
          dataset="argilla/ultrafeedback-binarized-preferences-cleaned",
          size="~60K pairs",
          pros="Ready to use, diverse",
          cons="Not code-specific",
      ),
      Option_B=dict(
          title="Filter for code",
          dataset="HuggingFaceH4/ultrafeedback_binarized",
          filter="Keep only code-related prompts",
          size="~5K pairs after filtering",
          pros="More relevant",
          cons="Smaller dataset",
      ),
      Option_C=dict(
          title="Generate your own",
```

preference\_data\_script.pycpu-only

```
def preference_data_implementation():
  """
  Implementation for preparing preference data.
  """
  print("Preference Data Implementation")
  print("=" * 60)
  print()

  code = '''
# preference_data.py - Prepare preference pairs for DPO

from datasets import load_dataset
from transformers import AutoTokenizer

def load_code_preferences():
  """
  Load and filter UltraFeedback for code-related preferences.
  """
  # Load the binarized preferences dataset
  dataset = load_dataset(
      "HuggingFaceH4/ultrafeedback_binarized",
      split="train_prefs"
  )

  # Keywords to identify code-related prompts
  code_keywords = [
      "code", "function", "program", "script", "implement",
      "algorithm", "python", "javascript", "java", "c++",
      "debug", "fix", "error", "bug", "class", "method",
      "array", "list", "loop", "recursion", "api",
```

## Step 5: DPO Training[#](#step-5-dpo-training)

dpo\_training\_script.pycpu-only

```
def dpo_training_implementation():
  """
  Complete DPO training script.
  """
  print("DPO Training Implementation")
  print("=" * 60)
  print()

  code = '''
# dpo_train.py - Direct Preference Optimization Training

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig
from preference_data import prepare_dpo_dataset

# ============================================
# CONFIGURATION
# ============================================

SFT_MODEL_PATH = "./sft-code-assistant"  # From Step 3
OUTPUT_DIR = "./dpo-code-assistant"

# DPO configuration
DPO_CONFIG = DPOConfig(
  output_dir=OUTPUT_DIR,

  # Core DPO hyperparameters
  beta=0.1,                       # KL penalty strength (0.1-0.5 typical)
```

dpo\_expected\_results.pycpu-only

```
import numpy as np

def dpo_expected_results():
  """
  What to expect from DPO training.
  """
  np.random.seed(42)

  print("DPO Expected Results")
  print("=" * 60)
  print()

  print("KEY METRICS TO MONITOR:")
  print("-" * 50)
  print()

  metrics = [
      ("Loss", "Should decrease from ~0.69 (random) to ~0.3-0.5"),
      ("Accuracy", "Chosen vs rejected accuracy: 65% -> 80%+"),
      ("Reward margin", "Difference in implicit rewards: increasing"),
      ("KL divergence", "Should stay bounded (< 10-20)"),
  ]

  for metric, description in metrics:
      print("  %s:" % metric)
      print("    %s" % description)
      print()

  print("TYPICAL TRAINING CURVE:")
  print("-" * 50)
```

## Step 6: Evaluation[#](#step-6-evaluation)

evaluation\_script.pycpu-only

```
def evaluation_implementation():
  """
  Evaluation pipeline for the trained model.
  """
  print("Evaluation Implementation")
  print("=" * 60)
  print()

  code = '''
# evaluate.py - Evaluate the aligned coding assistant

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.evaluation import evaluate_functional_correctness
import json

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "./dpo-code-assistant"
BASELINE_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ============================================
# EVALUATION FUNCTIONS
# ============================================

def generate_code(model, tokenizer, prompt, max_tokens=256):
  """Generate code completion."""
```

expected\_benchmark\_results.pycpu-only

```
import numpy as np

def expected_benchmark_results():
  """
  Expected results on coding benchmarks.
  """
  np.random.seed(42)

  print("Expected Benchmark Results")
  print("=" * 60)
  print()

  print("HUMANEVAL RESULTS (pass@1):")
  print("-" * 50)
  print()

  results = [
      ("TinyLlama-1.1B (base)", 0.08),
      ("TinyLlama-1.1B + SFT", 0.12),
      ("TinyLlama-1.1B + SFT + DPO", 0.15),
      ("---", None),
      ("Phi-2 (base)", 0.47),
      ("Phi-2 + SFT + DPO", 0.52),
      ("---", None),
      ("CodeLlama-7B (base)", 0.34),
      ("CodeLlama-7B + SFT + DPO", 0.42),
  ]

  print("%-35s %-10s" % ("Model", "pass@1"))
  print("-" * 45)
```

## Complete Pipeline Summary[#](#complete-pipeline-summary)

full\_pipeline\_summary.pycpu-only

```
def full_pipeline_summary():
  """
  Summary of the complete alignment pipeline.
  """
  print("Complete Alignment Pipeline Summary")
  print("=" * 70)
  print()

  pipeline = """
STEP-BY-STEP EXECUTION GUIDE
============================

1. SETUP (~10 minutes)
 pip install transformers peft trl datasets accelerate bitsandbytes
 pip install human-eval  (For evaluation)

2. PREPARE DATA (~5 minutes)
 - Download CodeAlpaca-20k for SFT
 - Filter UltraFeedback for code preferences

3. SFT TRAINING (~45-60 minutes on T4)
 - Load TinyLlama-1.1B
 - Apply LoRA (r=16, target Q/K/V/O + FFN)
 - Train for 3 epochs
 - Save checkpoint

4. DPO TRAINING (~30-45 minutes on T4)
 - Load SFT checkpoint as policy
 - Load SFT checkpoint as frozen reference
 - Train for 1 epoch on preferences
```

## Iteration Strategies[#](#iteration-strategies)

After your first pipeline run, you'll want to iterate. Here's a systematic approach:

Quality

Speed

Scores

Yes

No

Run Pipeline v1

Analyze Results

What Failed?

Improve Data

Optimize Hyperparams

Tune for Benchmark

Run Pipeline v2

Compare Results

Improved?

Deploy v2

Debug & Rethink

▶Iteration 1: Verify Pipeline Works

**Goal:** Get the full pipeline running end-to-end, even if results are mediocre.

**What to check:**

* SFT training completes without OOM
* Loss decreases (not flat, not NaN)
* Eval loss is reasonable (not massively higher than train)
* DPO training starts and runs 1 full epoch
* Model generates coherent code (manual spot check)

**Skip:** Hyperparameter tuning, data filtering, extensive evaluation

**Checklist:**

* SFT model saves successfully
* Can load SFT model and generate text
* DPO training runs without crashes
* DPO loss is reasonable (not 0.69 or NaN)
* DPO model saves successfully

▶Iteration 2: Improve Data Quality

**Goal:** Your results will be limited by data, not hyperparams. Fix the data.

**For SFT dataset:**

* Filter out very short/long examples
* Remove examples with syntax errors
* Ensure variety in instruction types
* Check for data contamination with eval set

**For DPO preferences:**

* Remove pairs where both are equally good
* Remove pairs where both are equally bad
* Verify "chosen" is actually better than "rejected"
* Filter out non-code examples

**Quick quality check:**
Spot check 10 random SFT examples: loop over `dataset.shuffle(seed=42).select(range(10))`, print instruction and response, and verify quality.

**Expected impact:** +3-5% on downstream benchmarks

▶Iteration 3: Hyperparameter Tuning

**Goal:** Squeeze more performance from the same data.

**High-impact hyperparameters (in order):**

1. **SFT learning rate:** Try 1e-4, 2e-4, 5e-4 (current: 2e-4)
2. **DPO beta:** Try 0.05, 0.1, 0.2 (current: 0.1)
3. **DPO learning rate:** Try 1e-5, 5e-5, 1e-4 (current: 5e-5)
4. **LoRA rank:** Try 8, 16, 32 (current: 16)

**Quick sweep strategy:**
Test 2-3 configurations per run with different combos of `sft_lr` (1e-4, 2e-4, 5e-4) and `dpo_beta` (0.05, 0.1, 0.2).

**Expected impact:** +2-3% with right tuning, -5% with wrong tuning

## Advanced: Multi-Round DPO[#](#advanced-multi-round-dpo)

Once you have a working pipeline, try this sophisticated iteration strategy:

multi\_round\_dpo.pycpu-only

```
def multi_round_dpo_strategy():
  """
  Run multiple rounds of DPO with different preference datasets.
  This is how big labs iterate: SFT once, DPO many times.
  """
  print("Multi-Round DPO Strategy")
  print("=" * 60)
  print()

  strategy = """
MULTI-ROUND DPO PIPELINE
========================

Round 1 (General): DPO on diverse code preferences
Input: SFT model
Data: UltraFeedback (code filtered)
Output: v1 model
Expected improvement: +5-10% on benchmarks

Round 2 (Specialized): DPO on HumanEval-like problems
Input: v1 model
Data: Sample v1 model, rank outputs with pass/fail on HumanEval
Output: v2 model
Expected improvement: +3-5% on HumanEval specifically

Round 3 (Polish): DPO on quality/style preferences
Input: v2 model
Data: Comparisons of code formatting, naming, comments
Output: v3 model (production ready)
Expected improvement: +2-3%, significant qualitative improvements
```

iteration\_tracking.pycpu-only

```
def iteration_tracking_system():
  """
  Track iterations to understand what works.
  """
  print("Iteration Tracking System")
  print("=" * 60)
  print()

  import json
  from datetime import datetime

  code = '''
# iteration_tracker.py

class IterationTracker:
  """Track experiments across the pipeline."""

  def __init__(self, experiment_name):
      self.experiment = experiment_name
      self.iterations = []

  def log_iteration(self, iteration_num, config, results):
      """Log one complete pipeline run."""
      self.iterations.append({
          "timestamp": datetime.now().isoformat(),
          "iteration": iteration_num,
          "config": config,
          "results": results,
      })
```

## Break It: Common Pipeline Failures[#](#break-it-common-pipeline-failures)

break\_it\_pipeline.pycpu-only

```
def pipeline_failure_modes():
  """
  Common failures in the SFT -> DPO pipeline and how to fix them.
  """
  print("Pipeline Failure Modes")
  print("=" * 60)
  print()

  failures = dict(
      SFT_overfitting=dict(
          symptoms=[
              "Train loss very low, eval loss high",
              "Model memorizes training examples",
              "Poor generalization to new prompts",
          ],
          causes=[
              "Too many epochs (>3-4 on small datasets)",
              "Learning rate too high",
              "Dataset too small",
          ],
          fixes=[
              "Reduce epochs to 1-2",
              "Add dropout (LoRA dropout 0.1)",
              "Use larger/more diverse dataset",
          ],
      ),
      DPO_not_learning=dict(
          symptoms=[
              "Loss stuck at ~0.69 (random)",
              "Accuracy stays at 50%",
```

## Scale Thought Experiment: From 1B to 70B[#](#scale-thought-experiment-from-1b-to-70b)

What breaks when we scale this pipeline up? Let's think through the implications:

scaling\_analysis.pycpu-only

```
def scaling_analysis():
  """
  How the alignment pipeline changes as we scale up.
  """
  print("Scaling Analysis: 1B -> 70B")
  print("=" * 70)
  print()

  models = [
      dict(
          name="TinyLlama-1.1B",
          params=1.1e9,
          sft_memory_gb=8,
          sft_time_t4_min=50,
          sft_time_a100_min=15,
          dpo_memory_gb=12,
          dpo_time_t4_min=40,
          dpo_time_a100_min=12,
      ),
      dict(
          name="Phi-2",
          params=2.7e9,
          sft_memory_gb=12,
          sft_time_t4_min=120,
          sft_time_a100_min=35,
          dpo_memory_gb=20,
          dpo_time_t4_min=100,
          dpo_time_a100_min=25,
      ),
      dict(
```

1B Model  
1 GPU

Knowledge  
works

Pipeline  
works

Quick  
iteration

7B Model  
1 GPU LoRA

Better  
results

Longer  
training

More  
data needed

70B Model  
8 GPUs

Production  
quality

Distributed  
training

Data  
critical

## Exercises[#](#exercises)

### Exercise 1: Pipeline Architecture[#](#exercise-1-pipeline-architecture)

**Problem:** Design a variant where you run SFT and DPO in parallel (separate from the sequential pipeline we built). What are the pros and cons?

**Thinking guide:**

* How would loss be different?
* What about KL divergence tracking?
* Memory implications?

### Exercise 2: Data Debugging[#](#exercise-2-data-debugging)

**Problem:** Your DPO dataset has 1000 preference pairs, but loss doesn't decrease after 100 steps. Write code to:

1. Sample 10 random pairs and check format
2. Verify `chosen` is actually better than `rejected` (manual judgment)
3. Check for duplicates or data leakage
4. Plot length distribution

### Exercise 3: Hyperparameter Grid Search[#](#exercise-3-hyperparameter-grid-search)

**Problem:** You have 2 hours and want to find the best (sft\_lr, dpo\_beta) pair. You can afford 3 full runs. Design a smart search strategy.

**Constraints:**

* Each run takes ~40 minutes
* You care about HumanEval pass@1
* You want to explore the space efficiently

**Hint:** What would an ML engineer do? (Answer: start with reasonable defaults, then explore one direction that looks promising)

### Exercise 4: Scaling to Larger Models[#](#exercise-4-scaling-to-larger-models)

**Problem:** Your TinyLlama pipeline works, but you want to try Phi-2 (2.7B) or CodeLlama-7B. What needs to change?

**Checklist:**

* Memory estimates (SFT + DPO)
* Batch size adjustments
* Learning rate scaling
* Expected speedup/slowdown
* GPU requirements (T4, A100, H100?)

## Checkpoint Questions[#](#checkpoint-questions)

1. 1.1B params, LoRA r=16, 7 modules, d=2048. Total LoRA params? Percentage of total?
2. DPO: loss 0.69 to 0.36, accuracy 82%, KL=8.5 after 500 steps. Healthy? What if KL exceeds 50?
3. HumanEval (164): baseline 8%, aligned 15%. Additional problems solved? Scale to 7B (baseline 34%): estimated aligned score?

## Research Hooks[#](#research-hooks)

**Pipeline variations:**
The SFT -> DPO pipeline is one approach. Alternatives include SFT -> RLHF, or direct preference learning (skipping SFT). When does each approach work best?

**Iterative refinement:**
Real alignment involves multiple rounds. Can you run DPO multiple times with progressively harder preferences? When do you see diminishing returns?

**Data efficiency:**
How much preference data is "enough"? Research suggests diminishing returns after a few thousand high-quality pairs. Can you quantify the data/performance tradeoff?

**Combining objectives:**
Can you train SFT and DPO simultaneously? Or add additional objectives (e.g., safety, helpfulness, honesty) as separate weighted DPO terms? How do conflicting objectives affect convergence?

**Preference data synthesis:**
Instead of annotating preferences, can you generate them automatically (e.g., with GPT-4)? What's the quality/cost tradeoff vs. human annotation?

---

*You've built a complete alignment pipeline from scratch! In Part 2, we'll analyze the results, identify failures, and plan improvements.*