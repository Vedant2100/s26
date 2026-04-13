1,000 high-quality examples often beats 100,000 low-quality ones. The LIMA paper proved this — data quality dominates quantity for instruction tuning.

## Learning Progression (Easy -> Hard)[#](#learning-progression-easy-hard)

Use this sequence as you read:

1. Start with `Prerequisites Refresher: What Even Is SFT?` to build core intuition and shared vocabulary.
2. Move to `Understanding SFT Loss and Convergence` to understand the mechanism behind the intuition.
3. Apply the idea in `Quality Metrics: How to Measure Data Quality` with concrete examples or implementation details.
4. Challenge your understanding in the failure-mode section and check what breaks first.
5. Then zoom out to scale-level tradeoffs so the same concept holds at larger model and system sizes.
6. Map the concept to production constraints to understand how teams make practical tradeoffs.

## Prerequisites Refresher: What Even Is SFT?[#](#prerequisites-refresher-what-even-is-sft)

*Flow bridge: Start here; this section establishes the base mental model for the rest of the lesson.*

▶Quick refresher: Supervised Fine-Tuning (SFT)

Supervised fine-tuning takes a **pretrained language model** (e.g., GPT-3 base, LLaMA base) and teaches it to follow instructions by training it on instruction-response pairs.

**The basic setup:**

```
Instruction: "What is machine learning?"
Response: "Machine learning is a subset of AI where systems learn from data..."

↓ (repeat ~1000 times)

Model learns: "When given an instruction, generate a helpful response"
```

**Key difference from pretraining:**

* **Pretraining** learns language patterns from trillions of tokens
* **SFT** learns *behavior and format* from thousands of examples

This is why SFT is so different from pretraining:

* You're not adding knowledge (the model already knows about ML)
* You're teaching the model *how to behave when asked a question*
* Poor examples teach it to be unhelpful, verbose, or wrong

**Training dynamics:**

* SFT is fast (hours, not weeks)
* Small changes in data have **outsized effects** on model behavior
* One contradictory example can undo the learning from 100 consistent examples

## Instructor Lens[#](#instructor-lens)

## Understanding SFT Loss and Convergence[#](#understanding-sft-loss-and-convergence)

*Flow bridge: Building on Prerequisites Refresher: What Even Is SFT?, this section adds the next layer of conceptual depth.*

▶Why does SFT training work differently than pretraining?

During pretraining, the model sees billions of examples, and averaging works well. During SFT, you have thousands of examples, and distribution matters enormously.

**The mathematical reality:**

* Pretraining loss: averaged over trillions of tokens → outliers don't matter
* SFT loss: averaged over thousands of examples → **each example is 0.1% of your signal**

This means:

1. One bad example out of 1000 corrupts 0.1% of training signal
2. But if that bad example contradicts the other 999, the model oscillates
3. Models are surprisingly good at detecting contradictions

**Example:**

```
Examples 1-999: "Always end responses with a period."
Example 1000: "Here's an example without proper punctuation"

Result: Model learns inconsistent behavior.
```

During convergence, the model's loss might decrease, but its behavior becomes **incoherent** because it's trying to fit contradictory instructions.

## Quality Metrics: How to Measure Data Quality[#](#quality-metrics-how-to-measure-data-quality)

*Flow bridge: Building on Understanding SFT Loss and Convergence, this section adds the next layer of conceptual depth.*

Before diving into filters, you need to define what "quality" means. Here are the standard metrics:

Quality Metrics

Correctness  
Are answers factually accurate?

Consistency  
Is behavior uniform across examples?

Completeness  
Are answers thorough?

Diversity  
Do examples cover different types?

Clarity  
Is the response well-formatted?

quality\_metrics.pycpu-only

```
import json
from collections import Counter

class DatasetQualityAnalyzer:
  """Analyze quality metrics across a dataset."""

  def __init__(self, examples):
      self.examples = examples

  def score_correctness(self):
      """
      Correctness: Use a reference model or human labels.
      For now, heuristic: responses without common error patterns.
      """
      error_patterns = {
          "i don't know": 0.3,  # Unhelpful
          "idk": 0.2,
          "as an ai": 0.0,  # ChatGPT-ism
          "i cannot": 0.1,  # Too many refusals
      }

      scores = []
      for ex in self.examples:
          resp = ex.get("response", "").lower()
          score = 1.0
          for pattern, penalty in error_patterns.items():
              if pattern in resp:
                  score -= penalty
          scores.append(max(0, score))
```

## The LIMA Result[#](#the-lima-result)

*Flow bridge: Building on Quality Metrics: How to Measure Data Quality, this section adds the next layer of conceptual depth.*

lima\_insight.pycpu-only

```
import numpy as np

def simulate_quality_vs_quantity():
  """
  Illustrate the LIMA paper's finding:
  1K high-quality ≈ 52K medium-quality > 100K low-quality
  """
  configs = [
      {"name": "100K low-quality", "n": 100000, "quality": 0.3},
      {"name": "52K medium-quality (Alpaca)", "n": 52000, "quality": 0.6},
      {"name": "1K high-quality (LIMA)", "n": 1000, "quality": 0.95},
      {"name": "10K high-quality", "n": 10000, "quality": 0.90},
  ]

  # Simplified model: performance = f(n, quality)
  # Diminishing returns on quantity, linear on quality
  def estimate_performance(n, quality):
      quantity_contribution = np.log10(n + 1) / 6  # Logarithmic in quantity
      quality_contribution = quality  # Linear in quality
      return 0.3 * quantity_contribution + 0.7 * quality_contribution

  print("Quality vs Quantity in SFT")
  print("=" * 60)
  print()
  print("%-30s %10s %10s %12s" % ("Config", "Examples", "Quality", "Performance"))
  print("-" * 65)

  results = []
  for config in configs:
      perf = estimate_performance(config["n"], config["quality"])
```

## Dimensions of Data Quality[#](#dimensions-of-data-quality)

*Flow bridge: Building on The LIMA Result, this section adds the next layer of conceptual depth.*

Quality Dimensions

Correctness  
Factually accurate  
No errors

Diversity  
Different task types  
Various formats

Formatting  
Consistent style  
Clean structure

Helpfulness  
Actually useful  
Complete answers

High Quality  
Training Data

quality\_dimensions.pycpu-only

```
# Examples illustrating each quality dimension

quality_examples = {
  "correctness": {
      "bad": {
          "instruction": "What is the capital of Australia?",
          "response": "Sydney is the capital of Australia."  # WRONG - it's Canberra
      },
      "good": {
          "instruction": "What is the capital of Australia?",
          "response": "Canberra is the capital of Australia."
      }
  },
  "diversity": {
      "bad": [  # All the same type
          {"instruction": "What is X?", "response": "X is..."},
          {"instruction": "What is Y?", "response": "Y is..."},
          {"instruction": "What is Z?", "response": "Z is..."},
      ],
      "good": [  # Different types
          {"instruction": "What is X?", "response": "X is..."},
          {"instruction": "How do I do Y?", "response": "To do Y, first..."},
          {"instruction": "Compare A and B", "response": "A and B differ in..."},
          {"instruction": "Write code for Z", "response": "def z(): ..."},
      ]
  },
  "formatting": {
      "bad": {
          "instruction": "explain machine learning",  # No capitalization
          "response": "machine learning is ai. it learns from data. models are trained."  # Poor formatting
```

## Quality Filter 1: Perplexity-Based[#](#quality-filter-1-perplexity-based)

*Flow bridge: Building on Dimensions of Data Quality, this section adds the next layer of conceptual depth.*

Low-perplexity responses (from the base model's perspective) are often higher quality.

perplexity\_filter.pycpu-only

```
import numpy as np

def estimate_perplexity(text, vocab_probs):
  """
  Estimate perplexity of text given vocabulary probabilities.
  Lower perplexity = model finds text more "natural"
  """
  # Simplified: assume each word has independent probability
  words = text.lower().split()
  log_probs = []

  for word in words:
      # Get probability (or low default for unknown words)
      prob = vocab_probs.get(word, 0.0001)
      log_probs.append(np.log(prob))

  # Perplexity = exp(-mean(log_prob))
  if log_probs:
      return np.exp(-np.mean(log_probs))
  return float('inf')

# Simulated vocabulary with common words having higher probability
vocab_probs = {
  "the": 0.07, "is": 0.05, "a": 0.04, "to": 0.03, "of": 0.03,
  "and": 0.03, "in": 0.02, "that": 0.02, "for": 0.02, "it": 0.02,
  "machine": 0.001, "learning": 0.001, "data": 0.001, "model": 0.001,
  "python": 0.0005, "code": 0.0005, "function": 0.0005,
  # Low-quality indicators
  "asdf": 0.00001, "lol": 0.0001, "idk": 0.0001,
}
```

## Quality Filter 2: Classifier-Based[#](#quality-filter-2-classifier-based)

*Flow bridge: Building on Quality Filter 1: Perplexity-Based, this section adds the next layer of conceptual depth.*

Train a classifier to distinguish high-quality from low-quality responses.

classifier\_filter.pycpu-only

```
import numpy as np

def quality_classifier(response):
  """
  Simplified quality classifier based on heuristics.
  In practice, this would be a trained neural network.
  """
  score = 0.5  # Start neutral

  # Length check (not too short, not too long)
  length = len(response)
  if 50 < length < 2000:
      score += 0.1
  elif length < 20:
      score -= 0.2

  # Structure check
  if "\n" in response:  # Has formatting
      score += 0.1
  if "1." in response or "- " in response:  # Has lists
      score += 0.1
  if "````" in response:  # Has code blocks
      score += 0.1

  # Content quality signals
  if response[0].isupper():  # Proper capitalization
      score += 0.1
  if response.endswith(".") or response.endswith("?"):  # Proper ending
      score += 0.05
```

## Advanced Filtering Strategies[#](#advanced-filtering-strategies)

*Flow bridge: Building on Quality Filter 2: Classifier-Based, this section adds the next layer of conceptual depth.*

Raw Data  
(100K examples)

Filter 1: Length  
Remove too short/long

Filter 2: Perplexity  
Remove incoherent text

Filter 3: Diversity  
Remove duplicates

Filter 4: Classifier  
Score quality

Filter 5: Consistency  
Check format

Curated Data  
(1-5K examples)

Rejected Examples  
(95K)

filtering\_pipeline.pycpu-only

```
class FilteringPipeline:
  """Multi-stage filtering pipeline to curate SFT data."""

  def __init__(self, examples):
      self.examples = examples
      self.rejected_reasons = {}

  def filter_length(self, min_chars=50, max_chars=5000):
      """Remove too short or too long responses."""
      filtered = []
      for ex in self.examples:
          resp = ex.get("response", "")
          if min_chars <= len(resp) <= max_chars:
              filtered.append(ex)
          else:
              self.rejected_reasons.setdefault("length", 0)
              self.rejected_reasons["length"] += 1
      self.examples = filtered
      return self

  def filter_perplexity(self, max_perplexity=200):
      """Remove incoherent responses using perplexity heuristic."""
      # Simplified: count ratio of common to uncommon words
      filtered = []
      common_words = {"the", "a", "is", "and", "to", "of", "in", "that", "for", "it"}

      for ex in self.examples:
          resp = ex.get("response", "").lower().split()
          if not resp:
              continue
```

## Quality Filter 3: Deduplication[#](#quality-filter-3-deduplication)

*Flow bridge: Building on Advanced Filtering Strategies, this section adds the next layer of conceptual depth.*

Remove near-duplicate examples to maximize diversity.

deduplication.pycpu-only

```
import numpy as np
from collections import defaultdict

def minhash_signature(text, num_hashes=100):
  """
  Create MinHash signature for near-duplicate detection.
  """
  # Create shingles (n-grams)
  words = text.lower().split()
  shingles = set()
  for i in range(len(words) - 2):
      shingle = " ".join(words[i:i+3])
      shingles.add(shingle)

  if not shingles:
      return [0] * num_hashes

  # Simple hash function simulation
  signature = []
  for seed in range(num_hashes):
      min_hash = float('inf')
      for shingle in shingles:
          # Deterministic "hash" based on seed
          h = hash(shingle + str(seed)) % (2**31)
          min_hash = min(min_hash, h)
      signature.append(min_hash)

  return signature

def jaccard_similarity(sig1, sig2):
```

## Break It: What Happens With Mixed-Quality Data?[#](#break-it-what-happens-with-mixed-quality-data)

*Flow bridge: Now that the core mechanism is clear, stress-test it under realistic failure conditions.*

The biggest mistake teams make: **mixing high-quality and low-quality data without separating them.** The model learns to be inconsistent.

break\_it\_mixed\_quality.pycpu-only

```
import numpy as np

def simulate_sft_with_mixed_data():
  """
  Simulate SFT training with different data compositions.
  Show how mixed quality hurts consistency.
  """

  configs = [
      {
          "name": "100% high-quality",
          "high_pct": 1.0,
          "low_pct": 0.0,
          "description": "All 1K examples are excellent"
      },
      {
          "name": "90% high + 10% low",
          "high_pct": 0.9,
          "low_pct": 0.1,
          "description": "900 good + 100 bad examples"
      },
      {
          "name": "50% high + 50% low",
          "high_pct": 0.5,
          "low_pct": 0.5,
          "description": "500 good + 500 bad examples"
      },
      {
          "name": "100% low-quality",
          "high_pct": 0.0,
```

## Break It: Quality Comparison[#](#break-it-quality-comparison)

*Flow bridge: Now that the core mechanism is clear, stress-test it under realistic failure conditions.*

break\_it\_quality.pycpu-only

```
import numpy as np

def simulate_training(dataset_quality, num_examples, epochs=3):
  """
  Simulate SFT with different quality datasets.
  """
  # Learning dynamics depend on quality
  noise_factor = 1 - dataset_quality
  learning_rate = 0.1 * dataset_quality  # Cleaner data = faster learning

  alignment = 0.2
  consistency = 0.5  # High-quality data leads to consistent behavior

  for epoch in range(epochs):
      # Alignment improves with examples
      alignment += learning_rate * np.log10(num_examples + 1) * (1 - alignment)

      # But noisy data hurts consistency
      consistency -= noise_factor * 0.1 * epoch

      # Add noise
      alignment += np.random.normal(0, noise_factor * 0.05)
      alignment = max(0, min(1, alignment))
      consistency = max(0.2, min(1, consistency))

  # Final score combines alignment and consistency
  return {
      "alignment": alignment,
      "consistency": consistency,
      "overall": alignment * consistency
```

## Break It: When Filtering Gets Too Aggressive[#](#break-it-when-filtering-gets-too-aggressive)

*Flow bridge: Now that the core mechanism is clear, stress-test it under realistic failure conditions.*

Counterintuitive: **Filtering too hard is actually worse than filtering too light.** If you throw away 98% of data, you might be left with data that's so narrow it causes overfitting.

break\_it\_filter\_tradeoff.pycpu-only

```
import numpy as np

def simulate_filter_aggressiveness():
  """
  Show the tradeoff: aggressive filtering → high quality but narrow diversity.
  Moderate filtering → decent quality with good coverage.
  """

  filtering_levels = [
      {"name": "No filtering", "retention": 1.0, "quality": 0.4},
      {"name": "Light (70% kept)", "retention": 0.7, "quality": 0.6},
      {"name": "Moderate (40% kept)", "retention": 0.4, "quality": 0.75},
      {"name": "Aggressive (10% kept)", "retention": 0.1, "quality": 0.88},
      {"name": "Extreme (2% kept)", "retention": 0.02, "quality": 0.95},
  ]

  print("BREAK IT: Filtering Aggressiveness Tradeoff")
  print("=" * 85)
  print()

  print("%-25s %10s %10s %10s %10s" % ("Filtering Level", "Retention", "Quality", "Coverage", "Overall"))
  print("-" * 85)

  results = []

  for config in filtering_levels:
      retention = config["retention"]
      quality = config["quality"]

      # Coverage: How diverse is the filtered dataset?
```

## Designing Annotation Guidelines[#](#designing-annotation-guidelines)

*Flow bridge: Building on Break It: When Filtering Gets Too Aggressive, this section adds the next layer of conceptual depth.*

annotation\_guidelines.pycpu-only

```
annotation_guidelines = """
SFT DATA ANNOTATION GUIDELINES
==============================

GOAL: Create instruction-response pairs that teach ideal assistant behavior.

RESPONSE QUALITY CHECKLIST:
---------------------------
[ ] CORRECT: Factually accurate, no errors
[ ] HELPFUL: Actually addresses the user's need
[ ] COMPLETE: Doesn't leave user hanging
[ ] CLEAR: Well-organized, easy to follow
[ ] APPROPRIATE LENGTH: Not too brief, not verbose

FORMATTING STANDARDS:
--------------------
- Start response with capital letter
- Use markdown formatting where helpful:
- Code blocks for code
- Numbered lists for steps
- Bullet points for options
- End with period or appropriate punctuation
- Use \n\n between paragraphs

EXAMPLE OF GOOD RESPONSE:
-------------------------
Instruction: How do I read a CSV file in Python?

Response:
You can read a CSV file in Python using the pandas library:
```

## Synthetic Data Generation: Scaling Data Cheaply[#](#synthetic-data-generation-scaling-data-cheaply)

*Flow bridge: With the local mechanism in place, extend it to larger model, context, and system scales.*

One key insight: **You don't need human-written data for all instruction types.** You can generate synthetic data from the base model and filter aggressively.

▶Why is synthetic data risky?

Synthetic data is generated by an LLM (GPT-4, Llama, etc.) rather than written by humans. It's cheap and fast, but has problems:

1. **Distribution collapse**: The generator tends to produce similar examples
2. **Hallucinations**: Generated data can contain factual errors
3. **Shallow diversity**: Tends to reuse same phrases and structures
4. **Self-reinforcement**: If you use the same generator for SFT, you're training a model on its own failures

The key: **aggressive filtering** of synthetic data. Only use the top 5-10% of generated examples.

synthetic\_generation.pycpu-only

```
import random

class SyntheticDataGenerator:
  """Generate candidate instruction-response pairs."""

  def __init__(self, seed_instructions=None):
      self.seed_instructions = seed_instructions or self._default_seeds()
      self.templates = self._instruction_templates()

  def _default_seeds(self):
      """Seed instructions to expand from."""
      return [
          "What is machine learning?",
          "How do I use Python?",
          "Explain neural networks",
          "What is data science?",
          "How do I debug code?",
      ]

  def _instruction_templates(self):
      """Templates for generating variations."""
      return [
          "What is {topic}?",
          "How do I {action}?",
          "Explain {topic}",
          "Compare {topic1} and {topic2}",
          "Write code that {action}",
          "What are the benefits of {topic}?",
          "List the steps to {action}",
      ]
```

## Quality-Quantity Budget Analysis[#](#quality-quantity-budget-analysis)

*Flow bridge: Building on Synthetic Data Generation: Scaling Data Cheaply, this section adds the next layer of conceptual depth.*

budget\_analysis.pycpu-only

```
import numpy as np

def analyze_annotation_budget(total_budget_usd, cost_per_example):
  """
  Given a fixed budget, how should we balance quality vs quantity?
  """
  configs = [
      {"name": "Low-quality (crowd)", "cost": 2, "quality": 0.4},
      {"name": "Medium-quality (trained)", "cost": 10, "quality": 0.7},
      {"name": "High-quality (expert)", "cost": 50, "quality": 0.9},
      {"name": "Excellent (researcher)", "cost": 200, "quality": 0.98},
  ]

  results = []

  for config in configs:
      num_examples = total_budget_usd // config["cost"]
      # Performance estimate
      quantity_factor = np.log10(num_examples + 1) / 5
      quality_factor = config["quality"]
      performance = 0.3 * quantity_factor + 0.7 * quality_factor

      results.append({
          "name": config["name"],
          "cost": config["cost"],
          "num_examples": num_examples,
          "quality": config["quality"],
          "performance": performance,
      })
```

## Scale Thought Experiment[#](#scale-thought-experiment)

*Flow bridge: With the local mechanism in place, extend it to larger model, context, and system scales.*

| Budget | Recommended Approach |
| --- | --- |
| **$1K** | 50-100 expert-written examples + AI augmentation |
| **$10K** | 1K high-quality human + 10K filtered synthetic |
| **$100K** | 10K expert human + extensive quality filtering pipeline |
| **$1M+** | Dedicated annotation team + multiple quality tiers |

## Practical: Combining Multiple Data Sources[#](#practical-combining-multiple-data-sources)

*Flow bridge: Building on Scale Thought Experiment, this section adds the next layer of conceptual depth.*

Real production systems don't use a single source of data. They combine:

Human-Written (10%)  
Expert knowledge  
~500 examples

Synthetic (30%)  
GPT-4 generated  
Filtered heavily  
~1500 examples

Crowd-Sourced (40%)  
MTurk/contractors  
Quality-filtered  
~2000 examples

Internal (20%)  
Company logs  
Real user feedback  
~1000 examples

5K SFT Examples  
Diverse, High Quality

High-Quality  
SFT Model

combining\_sources.pycpu-only

```
class MultiSourceDataset:
  """Combine and balance multiple data sources."""

  def __init__(self):
      self.sources = {
          "human": {
              "examples": 500,
              "quality": 0.95,
              "cost_per_ex": 50,
              "description": "Expert-written (researchers, annotators)"
          },
          "synthetic": {
              "examples": 5000,  # Before filtering
              "quality": 0.65,  # Needs aggressive filtering
              "cost_per_ex": 0.50,
              "description": "GPT-4 generated (filtered to top 10-20%)"
          },
          "crowd": {
              "examples": 3000,
              "quality": 0.70,
              "cost_per_ex": 5,
              "description": "MTurk/contractor (moderate quality)"
          },
          "internal": {
              "examples": 1000,
              "quality": 0.80,
              "cost_per_ex": 0,
              "description": "From product logs (real interactions)"
          },
      }
```

## Production Reality[#](#production-reality)

*Flow bridge: Carry these tradeoffs into production constraints and team-level operating decisions.*

**OpenAI's InstructGPT:**

* 40 contractors, extensive training
* Multi-stage review process
* Quality scoring and filtering
* ~13K final SFT examples

**Meta's Llama 2:**

* Multiple vendor partners
* Internal red team review
* Iterative quality improvement
* ~27K examples for SFT

**The LIMA approach:**

* Graduate students as annotators
* Stack Overflow, wikiHow as sources
* Aggressive curation (98% rejection rate!)
* Final: 1,000 examples

## Common Pitfalls and How to Avoid Them[#](#common-pitfalls-and-how-to-avoid-them)

*Flow bridge: Now that the core mechanism is clear, stress-test it under realistic failure conditions.*

▶Pitfall 1: Annotation style inconsistency

**Problem:** Different annotators write in different styles. One writes terse responses, another writes verbose essays. Model learns to match the majority style, which may be wrong for your use case.

**Solution:**

* Create detailed annotation guidelines with examples
* Have a single person (or small team) review all annotations
* Use format checkers: script that flags stylistic outliers
* Consider post-processing to normalize formatting

▶Pitfall 2: Over-weighting recent examples

**Problem:** If you keep adding new examples over time, the model's behavior drifts toward recent examples. Distributional shift accumulates.

**Solution:**

* Treat SFT dataset as fixed across versions
* If you want to update behavior, create a NEW SFT run
* Track which examples led to which behavior changes
* Version your datasets: sft\_v1, sft\_v2, etc.

▶Pitfall 3: Not tracking annotation provenance

**Problem:** You don't know where each example came from. If the model fails, you can't identify bad sources.

**Solution:**

* Add metadata: source, annotator, timestamp, quality\_score
* Log rejection reasons for discarded examples
* Track which examples influenced final behavior
* Create audit trail for regulatory compliance

▶Pitfall 4: Synthetic data mimicry

**Problem:** If you use GPT-4 to generate examples and then train on them, you're training a model to imitate GPT-4's style, not to be better.

**Solution:**

* Generate synthetic data, then filter heavily (keep top 5-10%)
* Use synthetic data as a **starting point**, not the final dataset
* Always include human-written examples as "anchors"
* Monitor for distributional drift toward generator's style

▶Pitfall 5: Ignoring edge cases

**Problem:** Your curated dataset is clean but misses edge cases, jailbreaks, and adversarial inputs. Model fails in production.

**Solution:**

* Reserve 10-20% of annotation budget for adversarial examples
* Include examples of: refusals, ambiguous inputs, trick questions, long contexts
* Test model on held-out edge cases before deployment
* Create red team examples: "How could someone misuse this?"

## Checklist: Is Your SFT Data High Quality?[#](#checklist-is-your-sft-data-high-quality)

*Flow bridge: Building on Common Pitfalls and How to Avoid Them, this section adds the next layer of conceptual depth.*

Before you start training, verify:

* **Diversity**: `num_unique_instruction_types >= 5` different task types
* **Consistency**: 95%+ examples follow same formatting rules
* **Completeness**: 95%+ examples are substantive (>100 chars)
* **Correctness**: <5% of examples contain factual errors (spot check)
* **Scale**: You have 500-5K examples (not 100K)
* **Deduplication**: <5% near-duplicate examples
* **Synthetic ratio**: <50% of dataset is synthetic/generated
* **Annotation quality**: Single annotator or tight review process
* **Versioning**: Dataset is tracked in version control
* **Metadata**: Each example has source, quality score, timestamp

If you fail any of these, **stop and fix it before training.** Bad SFT data is worse than no SFT.

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. Can you do this without notes: Define "quality" for instruction data (correctness, diversity, formatting, consistency)?
2. Can you do this without notes: Implement quality filters: perplexity-based, classifier-based, deduplication?
3. Can you do this without notes: Design annotation guidelines that maximize quality?

## Research Hooks[#](#research-hooks)

*Flow bridge: Use this practical baseline to frame the open research questions that remain unresolved.*

**Optimal curation strategies:**
What's the most efficient way to turn 100K noisy examples into 1K excellent ones? Active learning? LLM-as-judge? Human-in-the-loop?

**Synthetic data quality:**
GPT-4 generated data is cheap but has distributional biases. How do we detect and correct these biases? Can we measure "imitation distance" between generated and real examples?

**Quality metrics:**
Can we automatically measure response quality without human labels? What features predict human preference? Is perplexity actually correlated with quality?

**Data contamination:**
How do we detect if our SFT data accidentally includes test data from downstream benchmarks? What's the impact of 1% contamination?

**Few-shot learning:**
Can we achieve similar performance with 100 examples if we choose them perfectly? Is there a theoretical limit to data efficiency?

---

*Next up: You don't always need to train all the parameters. LoRA and other efficient methods let you fine-tune 70B models on consumer hardware.*