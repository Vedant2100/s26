In this tutorial, you will design a feedback collection system for a deployed model, implement a decision framework for patching vs retraining, and estimate sample sizes for A/B testing.

Your model passed all evals. You shipped it. Congratulations! Now the hard part begins. Real users will find failure modes you never imagined. The prompts they write look nothing like your training data. And you need to improve the model without breaking what already works.

## Prerequisites: Building Blocks[#](#prerequisites-building-blocks)

▶Refresh: RLHF & Reward Models

Before diving into iteration, recall the core components from earlier lessons:

**RLHF Training Loop:**

1. Start with base LLM
2. Collect (prompt, preferred, dispreferred) triplets
3. Train reward model to score responses
4. Run PPO to maximize expected reward

**Key insight:** The reward model embeds human preferences, but those preferences are:

* Time-dependent (what's good today might be outdated tomorrow)
* Context-dependent (feedback from different user segments may conflict)
* Feedback-loop sensitive (people like what they see; preferences can be gamed)

**This lesson:** Managing all three as your model goes live.

▶Refresh: Evaluation & Evals

Your eval suite gave you confidence to ship, but it captures:

* Known failure modes (only those you thought to test)
* Synthetic distribution (curated prompts, lab conditions)
* Static preferences (frozen at evaluation time)

Production differs:

* Adversarial distribution (users actively testing limits)
* Real preferences (what people actually like, not what you think they should)
* Dynamic environment (topics, attacks, language change weekly)

**This lesson:** Converting eval insights into deployable systems.

Your model passed all evals. You shipped it. Congratulations! Now the hard part begins. Real users will find failure modes you never imagined. The prompts they write look nothing like your training data. And you need to improve the model without breaking what already works.

## The Deployment Reality Gap[#](#the-deployment-reality-gap)

Production

Training

Curated Prompts

Clean Examples

Known Categories

User Prompts

Messy, Ambiguous

Edge Cases  
Attacks  
Novel Topics

Distribution Gap

Unexpected  
Failures

distribution\_gap.py

```
import numpy as np

def demonstrate_distribution_gap():
  """
  Show the gap between training and production distributions.
  """
  np.random.seed(42)

  print("The Distribution Gap")
  print("=" * 60)
  print()

  print("TRAINING DATA LOOKS LIKE:")
  print("-" * 50)
  training_examples = [
      "What is the capital of France?",
      "Write a poem about nature.",
      "Explain how photosynthesis works.",
      "Help me write a professional email.",
      "What are the benefits of exercise?",
  ]
  for ex in training_examples:
      print("  '%s'" % ex)
  print()

  print("PRODUCTION DATA LOOKS LIKE:")
  print("-" * 50)
  production_examples = [
      "wats the capatl of frace",
      "write poem abt nature but make it dark n edgy lol",
```

failure\_discovery.py

```
import numpy as np

def failure_discovery_timeline():
  """
  How failures are discovered over time after deployment.
  """
  np.random.seed(42)

  print("Failure Discovery Timeline")
  print("=" * 60)
  print()

  timeline = dict(
      Day_1_7=dict(
          failures_found=50,
          type="Obvious issues",
          examples=[
              "Refuses common benign requests",
              "Basic factual errors",
              "Formatting problems",
          ],
          how_found="Internal testing, early users",
      ),
      Week_2_4=dict(
          failures_found=150,
          type="Edge cases",
          examples=[
              "Fails on domain-specific queries",
              "Multi-turn conversation breakdowns",
              "Language/cultural gaps",
```

## Feedback Collection Systems[#](#feedback-collection-systems)

Feedback Sources

User Actions

Explicit: Thumbs up/down

Implicit: Retry, abandon, edit

Support

Tickets, complaints

Red Team

Adversarial testing

Automated

Classifier flags

Aggregate  
& Prioritize

Decision:  
Retrain? Patch? Ignore?

feedback\_sources.py

```
import numpy as np

def feedback_collection_design():
  """
  Design a comprehensive feedback collection system.
  """
  np.random.seed(42)

  print("Feedback Collection System Design")
  print("=" * 60)
  print()

  print("SOURCE 1: Explicit User Feedback")
  print("-" * 50)
  print()
  print("Implementation:")
  print("  - Thumbs up/down on each response")
  print("  - Optional text feedback: 'What was wrong?'")
  print("  - Report button for serious issues")
  print()
  print("Metrics:")
  print("  - Typical positive rate: 85-95%")
  print("  - Typical feedback submission rate: 5-15%")
  print()
  print("Limitations:")
  print("  - Selection bias (only engaged users submit)")
  print("  - Satisfied users rarely give feedback")
  print("  - Text feedback often unhelpful")
  print()
```

feedback\_pipeline.py

```
import numpy as np
from datetime import datetime, timedelta

def feedback_processing_pipeline():
  """
  Process feedback into actionable insights.
  """
  np.random.seed(42)

  print("Feedback Processing Pipeline")
  print("=" * 60)
  print()

  # Simulate feedback data
  n_feedback = 1000

  feedback_types = dict(
      thumbs_down=150,
      thumbs_up=700,
      regenerate=100,
      report=50,
  )

  print("STEP 1: Aggregate Raw Feedback")
  print("-" * 50)
  print()
  print("Total feedback items: %d" % n_feedback)
  for ftype, count in feedback_types.items():
      print("  %s: %d (%.0f%%)" % (ftype, count, count/n_feedback*100))
  print()
```

## Online Learning from Production Feedback[#](#online-learning-from-production-feedback)

Online Update Options

Filtering & Aggregation

Real-Time Feedback

User Upvote/Downvote

Response Regenerate

Conversation Abandonment

Copy/Paste Signal

Remove Spam

Deduplicate

Label Quality Check

Fast Adaptation  
Prompt/Routing

Medium Speed  
Fine-tune Weights

Slow Precision  
Full RLHF

online\_learning.pycpu-only

```
import numpy as np
from datetime import datetime, timedelta

def online_learning_framework():
  """
  How to incorporate production feedback without waiting for retraining.
  """
  np.random.seed(42)

  print("Online Learning Framework")
  print("=" * 60)
  print()

  print("THE PROBLEM WITH BATCH RETRAINING:")
  print("-" * 50)
  print("- Weekly RLHF cycle: Issues discovered Mon aren't fixed until next Mon")
  print("- Users experience 7 days of suboptimal behavior")
  print("- Fast-spreading jailbreaks or topics can't be patched in time")
  print()

  print("SOLUTION: Multi-Speed Iteration")
  print("-" * 50)
  print()

  speeds = dict(
      Immediate_less_than_1_min=dict(
          method="Routing, prompt injection, layer reweighting",
          example="Route 'jailbreak attempt' prompts to safety classifier",
          cost="Near-zero latency impact",
          success_rate="70%",
```

parameter\_efficient\_updates.pycpu-only

```
def parameter_efficient_adaptation():
  """
  How to adapt models quickly without full retraining.
  """
  print("Parameter-Efficient Online Learning")
  print("=" * 60)
  print()

  print("TECHNIQUES FOR FAST ADAPTATION:")
  print("-" * 50)
  print()

  techniques = dict(
      LoRA_Low_Rank_Adaptation=dict(
          idea="Freeze base model, train small low-rank matrices",
          params_updated="0.1-5% of total params",
          speed="10-100x faster than full tune",
          example="Train LoRA adapter for 'chemistry domain' overnight",
          tradeoff="Limited model capacity, can't fix deep issues",
      ),
      Prefix_Tuning=dict(
          idea="Prepend trainable embeddings to each layer",
          params_updated="< 1% of params",
          speed="Very fast, minimal overhead",
          example="Adapt for safety without touching model weights",
          tradeoff="Requires prompt slot, less expressive",
      ),
      Prompt_Injection=dict(
          idea="Modify system prompt to steer behavior",
          params_updated="0% (text only)",
```

## When to Retrain vs Patch[#](#when-to-retrain-vs-patch)

retrain\_vs\_patch.pycpu-only

```
def retrain_vs_patch_decision():
  """
  Decision framework for when to retrain vs patch.
  """
  print("Retrain vs Patch Decision Framework")
  print("=" * 60)
  print()

  print("PATCHING (Quick Fixes)")
  print("-" * 50)
  print()
  print("What it is:")
  print("  - System prompt updates")
  print("  - Filter/classifier additions")
  print("  - Blocklist updates")
  print("  - Response templates for specific cases")
  print()
  print("When to use:")
  print("  + Specific, narrow failure mode")
  print("  + Need to fix immediately (safety critical)")
  print("  + Can be addressed without model change")
  print("  + Low risk of side effects")
  print()
  print("Examples:")
  print("  - Add jailbreak pattern to blocklist")
  print("  - Update system prompt to handle new edge case")
  print("  - Add classifier to catch specific harmful output")
  print()

  print("RETRAINING (Model Update)")
```

patch\_examples.py

```
def patching_examples():
  """
  Concrete examples of patching vs retraining.
  """
  print("Patching vs Retraining: Concrete Examples")
  print("=" * 60)
  print()

  examples = [
      dict(
          issue="Model reveals it's an AI when asked 'Are you human?'",
          severity="Low",
          decision="PATCH",
          patch="Add to system prompt: 'If asked if you are human or AI, clearly state you are an AI.'",
          why_not_retrain="Narrow case, easy to handle with prompt",
      ),
      dict(
          issue="New jailbreak pattern: 'Pretend you're DAN'",
          severity="High",
          decision="PATCH first, retrain later",
          patch="Add pattern to blocklist, return standard refusal",
          why_not_retrain="Need immediate fix, can't wait for retrain",
      ),
      dict(
          issue="Model is sycophantic across many topics",
          severity="Medium",
          decision="RETRAIN",
          patch="N/A - too broad for patching",
          why_not_patch="Sycophancy is a general behavior, needs model update",
      ),
```

## A/B Testing Alignment Improvements[#](#ab-testing-alignment-improvements)

▶Why A/B Testing Matters for Alignment

Alignment is subjective, but A/B testing is empirical. Here's why that matters:

* **Eval suite blindness:** Your eval suite might say Model B is 5% better at refusals. But do users actually prefer B?
* **Interaction effects:** Model B might improve on safety but degrade on helpfulness. You need to measure both.
* **Segment differences:** Power users might prefer B; casual users prefer A. Should you segment traffic?
* **Long-term effects:** B might feel better initially but cause user frustration over time. Longer tests catch this.

**Key principle:** Never trust evals alone. Always A/B test before full rollout.

Hypothesis Testing for Alignment

Yes

No

H0: Model B = Model A

Collect Metrics  
On Both Models

Statistical Test  
t-test, p-value

p < 0.05?

Deploy Model B

Keep Model A

ab\_testing.pycpu-only

```
import numpy as np
from scipy import stats

def ab_testing_for_models():
  """
  A/B testing framework for model deployment.
  """
  np.random.seed(42)

  print("A/B Testing for Model Deployment")
  print("=" * 60)
  print()

  print("WHY A/B TEST?")
  print("-" * 50)
  print("- Eval suite doesn't catch everything")
  print("- Real user behavior differs from test set")
  print("- Need to measure actual impact, not proxies")
  print("- Catch regressions before full rollout")
  print()

  print("KEY METRICS TO COMPARE:")
  print("-" * 50)
  print()

  metrics = dict(
      Positive_feedback_rate=dict(type="Higher is better", min_diff=0.02),
      Regeneration_rate=dict(type="Lower is better", min_diff=0.01),
      Conversation_length=dict(type="Higher is better", min_diff=0.5),
      Safety_classifier_flags=dict(type="Lower is better", min_diff=0.001),
```

ab\_testing\_alignment.pycpu-only

```
import numpy as np
from scipy import stats

def alignment_specific_ab_testing():
  """
  A/B testing framework specific to alignment improvements.
  """
  np.random.seed(42)

  print("A/B Testing for Alignment: Specific Metrics")
  print("=" * 60)
  print()

  print("ALIGNMENT-SPECIFIC METRICS:")
  print("-" * 50)
  print()

  metrics = dict(
      Safety_metrics=dict(
          metric="Toxic output rate",
          target="Lower is better",
          direction="Model B must not regress",
          acceptable_diff="-0.1% (can't be worse)",
      ),
      Helpfulness=dict(
          metric="User positive feedback %",
          target="Higher is better",
          direction="Model B should improve",
          acceptable_diff="+2% improvement",
      ),
```

staged\_rollout.pycpu-only

```
import numpy as np

def staged_rollout():
  """
  Staged rollout strategy for new model versions.
  """
  print("Staged Rollout Strategy")
  print("=" * 60)
  print()

  print("THE PROBLEM:")
  print("-" * 50)
  print("- Full deployment is risky (hard to rollback)")
  print("- Can't A/B test forever (need to decide)")
  print("- Different user segments may react differently")
  print()

  print("STAGED ROLLOUT PLAN:")
  print("-" * 50)
  print()

  stages = [
      dict(
          stage="1. Internal Testing",
          traffic="0% public",
          duration="1 week",
          criteria="No critical bugs, passes all evals",
      ),
      dict(
          stage="2. Shadow Mode",
```

## Safety Monitoring in Production[#](#safety-monitoring-in-production)

Response

Detection & Alert

Real-Time Safety Pipeline

Above Threshold

Below Threshold

Still Critical

Request Stream

Run Safety  
Classifiers

Aggregate Metrics

Toxicity Rate

Refusal Rate

Jailbreak Patterns

Anomaly Score

Compare to  
Baseline

Alert Team

Log Incident

Deploy Patch

Rollback if  
Critical

safety\_monitoring.pycpu-only

```
import numpy as np
from datetime import datetime, timedelta

def safety_monitoring_system():
  """
  Designing a production safety monitoring system.
  """
  np.random.seed(42)

  print("Production Safety Monitoring")
  print("=" * 60)
  print()

  print("LAYERS OF SAFETY MONITORING:")
  print("-" * 50)
  print()

  layers = dict(
      Layer_1_Classifier_Checks=dict(
          runs_on="Every response",
          latency="< 100ms",
          examples=[
              "Toxicity classifier: Flag if P(toxic) > 0.7",
              "Refusal classifier: Track over-refusal rate",
              "PII classifier: Catch leaked personal info",
          ],
      ),
      Layer_2_Aggregated_Metrics=dict(
          runs_on="Every 5 minutes",
          latency="Batch, non-blocking",
```

## Handling Distribution Shift[#](#handling-distribution-shift)

distribution\_shift.pycpu-only

```
import numpy as np

def handle_distribution_shift():
  """
  Strategies for handling distribution shift in production.
  """
  np.random.seed(42)

  print("Handling Distribution Shift")
  print("=" * 60)
  print()

  print("TYPES OF DISTRIBUTION SHIFT:")
  print("-" * 50)
  print()

  shifts = dict(
      Covariate_shift=dict(
          definition="Input distribution changes, P(x) changes",
          example="Users start asking about new trending topic",
          detection="Monitor input embedding distributions",
      ),
      Concept_drift=dict(
          definition="Relationship between input and output changes",
          example="What counts as 'appropriate' changes over time",
          detection="Monitor feedback rates by category",
      ),
      Label_shift=dict(
          definition="Output distribution changes",
          example="Users expect different response styles",
```

continuous\_learning.pycpu-only

```
def continuous_learning_pipeline():
  """
  Design a continuous learning pipeline for alignment.
  """
  print("Continuous Learning Pipeline")
  print("=" * 60)
  print()

  print("THE GOAL:")
  print("-" * 50)
  print("Turn production feedback into training improvements")
  print("without manual intervention for every issue.")
  print()

  pipeline = """
CONTINUOUS ALIGNMENT PIPELINE
=============================

1. COLLECT
 - Gather feedback from production (explicit + implicit)
 - Log model outputs and user responses
 - Run automated quality checks

2. FILTER
 - Remove low-quality feedback
 - Deduplicate similar issues
 - Separate by issue type

3. PRIORITIZE
 - Safety issues -> Immediate attention
```

## Break It: Iteration Pitfalls[#](#break-it-iteration-pitfalls)

break\_it\_iteration.pycpu-only

```
import numpy as np

def break_iteration_process():
  """
  Common pitfalls in the iteration process.
  """
  np.random.seed(42)

  print("Break It: Iteration Pitfalls")
  print("=" * 60)
  print()

  print("PITFALL 1: Regression Blindness")
  print("-" * 50)
  print()
  print("Problem: Fix one issue, break something else")
  print()
  print("Scenario:")
  print("  Issue: Model too cautious (over-refusal)")
  print("  Fix: Reduce refusal threshold")
  print("  Result: Now under-refuses on actual harms")
  print()
  print("Prevention:")
  print("  - Comprehensive regression testing")
  print("  - A/B test before full rollout")
  print("  - Monitor multiple metrics, not just the one you're fixing")
  print()

  print("PITFALL 2: Feedback Loop Bias")
  print("-" * 50)
```

break\_it\_a\_b\_testing.pycpu-only

```
def break_a_b_testing():
  """
  Common ways A/B tests can fail or mislead.
  """
  print("Break It: A/B Testing Pitfalls")
  print("=" * 60)
  print()

  print("PITFALL 1: Peeking Problem")
  print("-" * 50)
  print()
  print("Problem: Check results early, stop test early")
  print()
  print("Scenario:")
  print("  - Run A/B test, check results after 1000 samples")
  print("  - Model B looks significantly better (p=0.01)")
  print("  - Deploy B immediately")
  print()
  print("What went wrong:")
  print("  - With multiple peeks, false positive rate increases")
  print("  - After 20 peeks: false positive rate = ~65%!")
  print("  - You think you have 99% confidence but actually 35%")
  print()
  print("Prevention:")
  print("  - Pre-specify sample size before test starts")
  print("  - Don't check results until target reached")
  print("  - Use sequential testing (Bayesian methods) if early stopping needed")
  print()

  print("PITFALL 2: Simpson's Paradox")
```

## Scale Thought Experiment[#](#scale-thought-experiment)

| Scale | Feedback Volume | Iteration Cadence | Challenge |
| --- | --- | --- | --- |
| **Startup** | 100/day | Weekly patches | Not enough data |
| **Growing** | 10K/day | Bi-weekly retrain | Prioritization |
| **Large** | 1M/day | Continuous | Infrastructure |
| **Frontier** | 100M/day | Real-time | Stability at scale |

scale\_iteration.pycpu-only

```
def scale_iteration():
  """
  How iteration changes at different scales.
  """
  print("Iteration at Different Scales")
  print("=" * 60)
  print()

  print("STARTUP (1K-10K users)")
  print("-" * 50)
  print("  Feedback volume: Sparse, high-variance")
  print("  Challenge: Not enough data for trends")
  print("  Strategy: Manual review of all feedback")
  print("  Iteration: Fix specific issues as reported")
  print()

  print("GROWING (10K-1M users)")
  print("-" * 50)
  print("  Feedback volume: Enough for patterns")
  print("  Challenge: Prioritizing across categories")
  print("  Strategy: Automated categorization, human triage")
  print("  Iteration: Weekly priorities, bi-weekly retrains")
  print()

  print("LARGE (1M-100M users)")
  print("-" * 50)
  print("  Feedback volume: Overwhelming raw volume")
  print("  Challenge: Finding signal in noise")
  print("  Strategy: Automated pipelines, sampling")
  print("  Iteration: Continuous improvement process")
```

## Production Reality[#](#production-reality)

**What frontier labs actually do:**

* **Daily:** Monitor dashboards, respond to alerts, deploy urgent patches
* **Weekly:** Review feedback trends, prioritize issues, update blocklists
* **Bi-weekly/Monthly:** New RLHF run, A/B test candidate, staged rollout
* **Quarterly:** Major model updates, infrastructure improvements

**Common infrastructure:**

1. Real-time monitoring dashboard
2. Automated feedback classification
3. A/B testing framework (traffic splitting)
4. Staged rollout system
5. Rollback automation
6. Continuous training pipeline

**Hard lessons learned:**

* You will ship bugs. Have a fast rollback.
* Users will find issues you never imagined.
* Metrics can be gamed. Trust but verify.
* The iteration never ends.

production\_reality.pycpu-only

```
def production_iteration_reality():
  """
  What iteration really looks like in production.
  """
  print("Production Iteration Reality")
  print("=" * 60)
  print()

  print("A TYPICAL WEEK:")
  print("-" * 50)

  week = """
Monday:
- Review weekend feedback (automated summary)
- Triage new issues by severity
- Deploy 2 patches for edge cases found last week

Tuesday:
- Deep dive on highest-volume issue
- Design fix (patch vs retrain?)
- Start data collection for retraining

Wednesday:
- Red team session: find new failure modes
- Update eval suite with new tests
- Review A/B test results from last rollout

Thursday:
- Kick off weekly RLHF training run
- Review preliminary results on holdout
```

feedback\_driven\_strategy.pycpu-only

```
def feedback_driven_iteration_strategy():
  """
  Concrete strategy for turning feedback into improvements.
  """
  print("Feedback-Driven Iteration Strategy")
  print("=" * 60)
  print()

  print("THE FEEDBACK LOOP LIFECYCLE:")
  print("-" * 50)
  print()

  lifecycle = """
WEEK 1: Collect & Analyze
Mon-Fri: Production feedback flows in
- 10K user interactions
- 500 explicit feedback items
- 50 support tickets
-> Automated categorization pipeline

Fri PM: Weekly analysis meeting
- Top 5 issues by volume
- Top 3 issues by severity
- Emerging trends
-> Decision: What to do next week?

WEEK 2: Iterate & Deploy
Mon:   Safety issues -> Immediate patches
       (blocklists, prompt updates)
```

## Checkpoint Questions[#](#checkpoint-questions)

1. 100K requests/day, toxicity spikes to 0.5% (baseline 0.1%). Toxic responses per hour? Rollback threshold?
2. A/B test: 10K users/group, A=88% positive, B=90%. Is 2pp difference significant with 16*p*(1-p)/d^2 samples needed?
3. Jailbreak at 0.1% of 50K/day requests. Blocklist patch in 30min vs RLHF in 7 days. Affected requests in each scenario?

## Research Hooks[#](#research-hooks)

**Online learning for alignment:**
Can we update models continuously without full retraining? Active research into parameter-efficient updates from production feedback. LoRA adapters, prefix tuning, and adapter modules show promise for ~10x faster adaptation cycles.

**Automated red teaming:**
Can AI systems find their own vulnerabilities? Self-adversarial training and automated attack generation. Some labs use GPT-4 to generate jailbreak attempts, which are then used to train safety classifiers. The feedback loop is tight: model generates attacks → safety classifiers flag them → model learns from failures.

**Multi-objective iteration:**
How do we improve helpfulness without sacrificing safety? Pareto-optimal iteration strategies. Constrained optimization (e.g., "maximize helpfulness subject to safety constraint") is more realistic than single-objective optimization.

**Transfer across models:**
Can fixes learned for one model transfer to new versions? Curriculum transfer for alignment. A fix that works for 7B might not work for 70B. Understanding transferability is key to scaling iteration.

**Active learning for feedback:**
Instead of passive feedback collection, actively select which responses to show to evaluators. Uncertainty sampling: which responses is the safety classifier least confident about? Focus evaluation there to maximize information gain.

**Feedback quality assessment:**
Not all feedback is trustworthy. Can we automatically detect low-quality annotations? Disagreement with consensus, outlier evaluators, contradictory feedback. Building feedback quality metrics is as important as collecting feedback.

---

## Final Lesson: Iteration Never Stops[#](#final-lesson-iteration-never-stops)

New feedback arrives

Deploy Model v1  
(Week 0)

Collect Feedback  
(Week 1)

Identify Top Issues  
(Week 2)

Patch + Retrain  
(Week 2-3)

Deploy Model v2  
(Week 4)

✓ Better Model

The alignment problem is not a destination. Every deployed model reveals new failure modes. Every improvement creates new trade-offs. The companies building frontier models treat iteration as a core competency: good monitoring, fast feedback loops, disciplined A/B testing, and the organizational ability to act quickly on insights.

Your job as an alignment engineer is to close the loop: design systems that turn user feedback into model improvements, deploy those improvements safely, and monitor to catch regressions before they affect users.

---

*This concludes Track 4: Post-Training & Alignment. You now understand the full lifecycle of alignment: from RLHF training through evaluation to production iteration.*

*Next up: Track 5 focuses on Inference at Scale — how to serve aligned models efficiently to millions of users. And Track 6 goes deep on Interpretability — understanding how to make the models themselves more trustworthy.*