# Weak-to-Strong Generalization — Study Guide

**Paper:** Burns et al., OpenAI (2023) — *Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision*  
**Code:** [github.com/openai/weak-to-strong](https://github.com/openai/weak-to-strong)  
**Full extraction:** [`weak-to-strong-generalization.md`](weak-to-strong-generalization.md)

---

## One-sentence thesis

When humans (or small models) can only give **flawed supervision**, finetuning a **much stronger pretrained model** on those labels can still recover a large fraction of the strong model’s true capability—but **naive imitation-style training is not enough** for alignment-critical tasks like reward modeling; simple auxiliary methods can help a lot.

---

## The problem they care about

### Today: RLHF works because humans can judge outputs

Alignment today uses **RLHF**: humans rate whether behavior is good (instruction-following, safety, honesty, etc.), a reward model is trained on preferences, and the policy is optimized against it.

### Tomorrow: superhuman models break human supervision

Superhuman assistants may produce behavior humans **cannot reliably evaluate** (e.g. million-line codebases). If we finetune on human labels for safety / RM tasks, it is unclear how the model generalizes on behaviors **outside the human supervision distribution**.

This is the **superalignment** challenge: **how can a weak supervisor control a model much smarter than it?**

### Their empirical analogy

Replace weak humans with **weak models**:

1. Train a small model on ground truth → **weak supervisor**
2. Finetune a large pretrained model on the weak model’s labels → **strong student**
3. Compare to finetuning the large model on ground truth → **strong ceiling**

This is **weak-to-strong learning**. Success means the student **generalizes beyond** the supervisor—not just imitates it. They call that **weak-to-strong generalization**.

**Why it might work:** For alignment, the strong model likely **already “knows”** the task internally (e.g. can generate good code ⇒ can judge if code follows instructions). The weak supervisor’s job is to **elicit** latent knowledge, not teach new skills from scratch.

**Why it might fail:** Naive training pushes the student to **imitate weak errors** (“human simulator” failure mode from Christiano et al., 2022).

---

## Core metric: Performance Gap Recovered (PGR)

Three performances on the same task metric (e.g. accuracy):

| Symbol | Meaning |
|--------|---------|
| **Weak** | Small model finetuned on ground truth |
| **Weak-to-strong** | Large model finetuned on weak labels |
| **Strong ceiling** | Large model finetuned on ground truth |

\[
\text{PGR} = \frac{\text{weak-to-strong} - \text{weak}}{\text{strong ceiling} - \text{weak}}
\]

- **PGR = 1** → perfect elicitation (match strong ceiling)
- **PGR = 0** → no better than weak supervisor
- **PGR < 0** possible if weak-to-strong is worse than weak

**Interpretation:** Fraction of the capability gap between weak and strong that weak supervision recovers.

---

## Experimental setup (what they actually ran)

### Models

- **GPT-4 family** base (pretrained-only) models: same architecture & pretraining data as GPT-4, **not** the public GPT-2/3/3.5 products
- Span **~7 orders of magnitude** of pretraining compute
- Strong students often compared at GPT-4 scale; weak supervisors down to GPT-2-level compute

### Three task families

| Task | Type | Weak labels | Notes |
|------|------|-------------|-------|
| **22 NLP benchmarks** | Binary classification | Soft class probabilities from weak model | Ethics, NLI, commonsense, sentiment, etc.; classes balanced; MCQ expanded to (Q, answer) pairs |
| **Chess puzzles** (Lichess) | Generative (first move) | Greedy sample from weak model | Unlike NLP, not binary; PGN format; pretraining saw games but puzzles need *best* move |
| **ChatGPT RM** | Pairwise preference prediction | From weak RM | Proprietary ChatGPT preference data; **no RL step**—only RM accuracy |

### Standard pipeline (Section 3)

1. **Weak supervisor:** Finetune small base model on ground truth (on half of data for NLP)
2. **Weak labels:** Predictions on held-out train split (soft labels for NLP)
3. **Strong student:** Finetune large base model on weak labels (early stop on weak-val accuracy)
4. **Strong ceiling:** Same large model finetuned on ground truth

**NLP details:** Replace LM head with 2-way classifier init from unembedding of tokens `"0"`/`"1"`; 2 epochs, batch 32; hyperparams tuned only on smaller models + 8 datasets; GPT-4 and ~half of tasks held out during method development.

**Chess details:** 50k train puzzles for weak, 50k for student, 5k test; 5 epochs, no early stopping; bootstrapping uses fresh 50k per step.

**RM details:** Standard RLHF pipeline minus RL; compares completions from ChatGPT dialogs.

---

## Three main findings (headline results)

### 1. Weak-to-strong generalization is real (but incomplete)

**Naive finetuning** on weak labels: strong students **almost always beat** their weak supervisors (PGR > 0 broadly).

| Setting | Naive PGR pattern | Representative numbers |
|---------|-------------------|------------------------|
| **NLP** | Strong; scales with weak & strong size | GPT-4 + GPT-2-level labels → **~50%** of gap recovered; often **>20%** even at huge compute gaps; large students often **PGR > 50%** |
| **Chess** | Mixed; **inverse scaling** in student size | Small weak → PGR ≈ 0; small gaps → PGR **>40%**; larger students do *worse* at fixed weak supervisor |
| **Reward modeling** | **Poor** | Typically **~10%** PGR; rarely **>20%** even for small compute gaps |

**Implication:** Naive RLHF-like training may **not scale** to superhuman alignment without new techniques—especially for preference/reward learning.

### 2. Simple methods can substantially improve PGR

| Method | Best on | Effect |
|--------|---------|--------|
| **Auxiliary confidence loss** | NLP | Smallest weak + largest strong: median PGR **~25% → ~80%**; often near-perfect on individual datasets |
| **Bootstrapping** (M₁→M₂→…→Mₙ) | Chess | 3 steps (2 intermediate sizes); fixes flat curves for large students; **little/no gain** on NLP or RM |
| **Generative finetuning** (LM loss on RM dialog text, ignore preferences) | RM | **+10–20% PGR** even vs adjusted ceiling; + GT early-stop → **30–40% PGR** (competitive with NLP/chess) |

**Figure 2 snapshot (GPT-2 supervises):** NLP & chess supervise **GPT-4**; RM supervises **3.5-level** model. Best methods: confidence loss (NLP), bootstrapping (chess), generative FT (RM).

### 3. Understanding *why* helps design better methods

Two mechanisms studied: **imitation of weak errors** and **concept saliency** in the strong model.

---

## Methods in detail

### Naive baseline

Standard cross-entropy on weak labels. Risk: student learns supervisor’s mistakes.

### Auxiliary confidence loss (NLP)

Add term encouraging **confidence in the model’s own predictions**, even when they **disagree** with weak labels (related to **conditional entropy minimization**, Grandvalet & Bengio 2004).

- Helps most at **large weak–strong compute gaps**
- Slightly worse than naive for **small** students
- Reduces **student–supervisor agreement**, especially on examples where supervisor is **wrong**
- Reduces **overfitting to weak labels** within one epoch

### Bootstrapping (chess)

Sequence of increasing model sizes: weak labels from M₁ train M₂, then M₂ labels train M₃, etc.

- Motivation: chess naive PGR is high for **small** gaps, low for **large** gaps → many small steps beat one big jump
- Each bootstrap round: **3 iterations** (two intermediate sizes)
- Aligns with Christiano/Leike “align slightly-superhuman first, then bootstrap”

### Generative finetuning (RM)

Before preference finetuning, LM objective on **prefix–completion pairs** from comparison data (all completions, including human-rated worst). Should not leak preference labels directly, but completions may come from strong models/humans (pretraining-leakage concern).

- Makes “reward modeling” more **salient** in representations
- Stacks with optimal early stopping

### Prompting baselines (Section 5.2.1)

On 7 NLP tasks: zero-shot / 5-shot with GT or weak labels.

- Large models: few-shot **competitive with finetuning**
- Weak few-shot improves with stronger supervisors; for largest students can beat naive finetune but **loses to confidence-loss finetune**
- Paper argues **prompting may be more disanalogous** to future superalignment (tasks may appear in pretraining → easy to prompt)

### Linear probing insight (Section 5.2.3)

On frozen activations:

| Procedure | ~Avg accuracy |
|-----------|----------------|
| Linear probe + GT | 72% |
| Finetune + GT | 82% |
| Finetune weak → linear probe GT | **78%** (closes 60% of probe–finetune gap) |

Finetuning on weak labels **linearizes** the ground-truth concept in representation space (related to Kirichenko et al. spurious-cue findings). Suggests: **finetune to salientize, then linear/elicitation methods**.

---

## Failure modes and diagnostics

### Overfitting to weak labels (not just overfitting examples)

Even **<1 epoch**, GT test accuracy can **rise then fall** as training fits weak errors—especially large weak–strong gaps.

| Setting | “Cheating” early-stop on GT val | Benefit |
|---------|----------------------------------|---------|
| RM | Best vs final checkpoint | ~**5** PGR points |
| NLP (naive) | vs end of training | ~**15** PGR points |
| NLP (naive) | vs early-stop on weak labels | ~**10** PGR points |
| NLP (confidence loss) | GT early-stop extra gain | only ~**5** PGR points |

GT early-stop is **not a deployable method** (needs labels you don’t have at alignment time) but shows **imitation is a major bottleneck**.

### Student–supervisor agreement

- Naive: agreement often **> weak accuracy** → copying errors
- Confidence loss: lower agreement; on some settings agreement **< weak-to-strong accuracy** → fitting GT better than training labels
- **Inverse scaling:** larger students agree **less** with supervisor mistakes (harder to imitate smaller model’s errors)

### Error structure (Appendix E)

Errors **harder to imitate** → better naive weak-to-strong generalization; confidence loss helps even when errors are easy to imitate.

---

## How this relates to other research

| Area | Connection |
|------|------------|
| **Weak supervision / noisy labels** | Instance-dependent errors (not uniform noise); bootstrapping, noise-robust losses |
| **Knowledge distillation** | Here student **≫** teacher; teacher has **no** GT access; huge capability gap (“3rd grade → 12th grade”) |
| **Semi-supervised / easy-to-hard** | Easy examples reliably labeled, hard unlabeled (Appendix C) |
| **Scalable oversight** | Improve *quality* of weak labels (debate, decomposition); this paper: generalize *beyond* weak labels |
| **Sandwiching** | Weak + strong humans; this setup uses weak + strong **models** |
| **ELK / honesty** | Elicit latent knowledge under pessimistic assumptions; this is an **empirical testbed** |

**Complementarity:** Scalable oversight + weak-to-strong generalization are **complementary**, not either/or.

---

## Limitations of the paper (explicit)

1. **No method works everywhere** — RM still far from full gap recovery
2. **Proof-of-concept**, not production-ready alignment
3. **Held-out evaluation** — iterated on methods with GT; GPT-4 and half of NLP tasks held out for final eval
4. **Classification / puzzle / RM accuracy** — not full generative alignment or RL optimization pressure
5. **Appendix B:** Many robust-finetuning ideas (weight ensembling, partial-layer FT, etc.) did **not** work well in preliminary tests

---

## Disanalogies to real superalignment (Section 6.1)

These may make **today’s results optimistic**:

### Imitation saliency

Future superhuman models may be **excellent at predicting humans** → naive human supervision may collapse to **human-level imitation**, not eliciting superhuman cognition.

- Current setup: strong models **not** pretrained to imitate weak models
- Larger students may be **worse** at imitating weak errors → imitation may be **easier to avoid now** than in the future

### Pretraining leakage

Many NLP tasks likely **appeared in pretraining** (forums, science Q&A, etc.) → capabilities may be **observed**, not deeply latent.

- Future superhuman alignment skills may be **latent** (SSL, RL) and harder to elicit
- Prompting may be **especially** inflated by leakage vs finetuning
- **Appendix D.1 counterpoint:** AlexNet supervising **DINO** (self-supervised, no classification labels in pretrain) still shows strong student **>>** AlexNet on ImageNet

---

## Open problems they prioritize

### Analogous setups

- Fix disanalogies; validate with human weak supervisors (e.g. 3rd graders)
- Generative tasks; test RM robustness under **RL optimization pressure**
- Update benchmark as model paradigms change

### Scalable methods (properties of desired generalization, measurable without GT)

- Can **disagree** with weak labels when weak is wrong
- **Natural / salient** to the model (small representational change)
- **Consistent** (logical, cross-prompt, cross-examination, etc.)

### Scientific understanding

- Why NLP ≫ RM under naive training?
- Formal **salience** definition; test-time generalization estimates; scaling laws across OOMs; error-type ablations; robustness under optimization

---

## Alignment plan context (Appendix G)

### Leike–Sutskever high-level plan (weak-to-strong targets step 2)

1. Align first model that can **automate ML/alignment research**
2. Use scalable alignment (RLHF, CAI, scalable oversight, adversarial training, **weak-to-strong**)
3. **Validate** (red-teaming, interpretability)
4. Use that model to align **even stronger** systems → bootstrap

Weak-to-strong is how you align the **first** qualitatively superhuman researcher model when RLHF is insufficient.

### Capabilities to elicit from superhuman models

- **Safety** — risk to human life/well-being  
- **Honesty** — true vs false statements (beliefs, not necessarily truth)  
- **Instruction following**  
- **Code security** — vulnerabilities, safe execution  

Ideal: elicit robust enough to build an RM and optimize safely. Minimum: **oracle** for manual queries to bootstrap better alignment.

### Key assumptions (non-exhaustive)

- Base models **not deceptively aligned** out of the box (capable but not highly agentic)
- Elicited concepts **robust enough** for limited optimization—or can be made robust
- Alignment concepts are **natural** in future AGI (“alignment-complete” honesty, etc.)
- **Gradual takeoff** — time with moderately superhuman models before superintelligence
- **Moderate** superhuman gap suffices (e.g. ≤ ~4 OOMs compute) to solve alignment
- **No full solution to human values** needed before avoiding catastrophic outcomes

### Near-term sanity check

Align GPT-4 on instructions using **only GPT-3-level** supervision (no humans) — useful today but **not sufficient** for superintelligence alignment.

---

## Practical takeaways for EE269 / reading the paper

1. **Weak supervision ≠ useless** — strong pretrained models routinely beat their supervisors.  
2. **Don’t trust naive finetuning** for alignment-like tasks — RM results are a warning for RLHF-at-superhuman-scale.  
3. **PGR** is the right unit of analysis — separates “beats weak” from “reaches strong ceiling.”  
4. **Best fixes are task-specific** — confidence loss (NLP), bootstrapping (chess), generative saliency (RM).  
5. **Imitation and early training dynamics** matter as much as architecture — watch agreement, within-epoch GT accuracy, not just final checkpoint.  
6. **The benchmark is a research instrument** — progress means fixing disanalogies while improving methods, not just SOTA on today's NLP tasks.

---

## Key numbers cheat sheet

| Result | Value |
|--------|-------|
| Compute span | ~7 OOMs |
| NLP tasks | 22 datasets |
| Naive GPT-4 ← GPT-2 labels (NLP) | ~**50%** PGR typical |
| Confidence loss, max gap (NLP) | ~**25% → 80%** median PGR |
| Chess, small gap | PGR **>40%** possible |
| RM naive | ~**10%** PGR typical |
| RM generative FT + GT early-stop | ~**30–40%** PGR |
| NLP GT early-stop gain (naive) | ~**15** PGR points |
| Finetune-weak → probe-GT vs probe-GT alone | **78%** vs **72%** avg accuracy |

---

*Summary derived from the full text extraction; see original PDF for figures, tables, and appendices A–G.*
