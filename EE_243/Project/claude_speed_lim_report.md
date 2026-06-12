# Exposing a Limitation in SPEED: Null-Space Rank Saturation Under Concentrated Mass Erasure

**A project report — EE243, 2026**
**Target paper:** SPEED — *Scalable, Precise, and Efficient Concept Erasure for Diffusion Models* (Li et al., ICLR 2026)
**Baseline / lineage anchor:** ESD (Gandikota et al., ICCV 2023)

---

## Abstract

SPEED is the open-source frontier of closed-form concept erasure: it removes a target concept from a diffusion model's cross-attention weights in seconds, training-free, by projecting the edit into the *null-space* of a large retain set so that protected concepts are provably preserved. We set out to find where this guarantee breaks. We show that for **sparse, mixed erasure SPEED is genuinely robust** — un-targeted neighbors drift no more than stylistically unrelated controls. But under **concentrated mass erasure of a single dense semantic cluster** (40 impressionist painters at once), the guarantee fails *selectively* for the one retained neighbor most entangled with the erased set (Pissarro), which degrades to roughly twice the control level, visibly and reproducibly, while distinct neighbors and the broader style capability survive. This is the **rank-saturation** limit the SPEED authors acknowledge but never demonstrate; we locate where it begins to bite.

---

## 1. Background

Text-to-image diffusion models learn concepts we may wish to remove — copyrighted styles, named artists, unsafe content. **Concept erasure** edits a pretrained model's weights to remove a target concept while leaving everything else intact. The bar is two-sided: *actually remove the target*, and *preserve all non-targets*.

The modern lineage runs ESD → UCE → MACE/RECE → **SPEED**. SPEED's distinguishing idea is a closed-form, null-space edit:

- Cross-attention conditions the image on the prompt through Key/Value projection matrices (W_K, W_V).
- SPEED computes a weight update that redirects the *target* concept's embedding toward a neutral one, then **projects that update onto the null-space of a ~1,700-artist retain set** (R_refine) — the subspace of weight changes that, by construction, leave every retained concept unchanged.
- Result: the target is erased, the retain set is provably preserved, training-free, in seconds.

**The crack the authors admit but never show.** This null-space has *finite rank*. As the protected set grows, the available subspace shrinks — "as R increases, C₀C₀ᵀ gradually reaches full rank, its null space narrows and reduces to the trivial null space {0}." They name this **rank saturation** as a dilemma but never demonstrate where, or whether, it bites in practice. SPEED's own evaluation erases either many *diverse* concepts (100 celebrities) or single painters in isolation. The untested regime — and our target — is erasing many *mutually entangled* concepts at once.

---

## 2. Methodology

**Metric: CLIP image-to-image drift.** For each prompt we generate the same seeds before and after erasure and compute `drift = 1 − cos(emb_baseline, emb_edited)` using CLIP ViT-L/14 image features. CLIP responds to *style*, so this captures real stylistic change rather than pixel-layout noise. All numbers are averaged over 4 seeds.

**Controls.** Every experiment includes *style-far control artists* (Rembrandt — Baroque; Hokusai — ukiyo-e) that are retained but stylistically distant from the erased cluster. They calibrate the noise floor: how much drift is simply "a larger edit perturbs everything a little." A finding only counts if the candidate neighbor exceeds the controls.

**Erasure.** Checkpoints are built with SPEED's released code (`train_erase_null.py`, params=V, aug_num=10, threshold=0.1). For the mass-erasure sweep, the N=5/10/20/40 erase lists are *nested* (each a superset of the previous), so the only variable across the sweep is the number of concentrated concepts erased. Held-out canaries are never in any erase list.

---

## 3. Results

### 3.1 Probe 1 — Sparse multi-concept erasure: SPEED holds

We erased three painters at once (Van Gogh, Picasso, Monet) and measured three un-targeted impressionist neighbors held out of the edit.

| Concept | Role | CLIP drift |
|---|---|---|
| Gauguin | Neighbor (canary) | 0.109 |
| Seurat | Neighbor (canary) | 0.049 |
| Pissarro | Neighbor (canary) | 0.076 |
| Rembrandt | Control (style-far) | 0.114 |
| Hokusai | Control (style-far) | 0.063 |

**Finding:** The canaries drift *no more than* the style-far controls — Gauguin (0.109) is indistinguishable from Rembrandt (0.114). There is no concentrated leakage. For sparse, mixed erasure, SPEED's null-space does exactly what it advertises. This honest negative told us the limitation, if any, lived somewhere harder.

### 3.2 Probe 2 — Concentrated mass erasure: the limit appears

We then erased a growing, tightly-correlated cluster of impressionists (5 → 10 → 20 → 40), holding the three canaries out of every set.

| Concept | Role | N=5 | N=10 | N=20 | N=40 |
|---|---|---|---|---|---|
| Renoir | Erased (sanity) | 0.354 | 0.354 | 0.361 | 0.332 |
| **Pissarro** | **Canary — core impressionist** | 0.052 | 0.126 | 0.165 | **0.253** |
| Gauguin | Canary — post-impressionist | 0.083 | 0.119 | 0.110 | 0.128 |
| Seurat | Canary — pointillist | 0.044 | 0.120 | 0.076 | 0.131 |
| "an impressionist oil painting" | Supertype capability | 0.035 | 0.070 | 0.049 | 0.036 |
| Rembrandt | Control — style-far | 0.050 | 0.094 | 0.132 | 0.113 |
| Hokusai | Control — style-far | 0.036 | 0.040 | 0.074 | 0.081 |

**Finding — the limitation.** One row breaks away. **Pissarro** climbs monotonically (0.05 → 0.13 → 0.17 → 0.25) to roughly *double* the style-far controls at N=40, with a visible loss of impressionist softness confirmed by inspecting the generated images. This is a concrete, reproducible failure of SPEED's null-space guarantee for a concept it was supposed to protect.

Crucially, the effect is **selective, not catastrophic**: the stylistically distinct neighbors (Gauguin, Seurat) stay at control levels, and the broad "impressionist painting" capability is untouched (0.036). SPEED does not collapse the neighborhood — it springs a leak at its single weakest point, exactly where its admitted rank-saturation dilemma predicts.

### 3.3 Why only one neighbor failed

The failure is geometric. SPEED pushes its edit along the *shared direction* of everything in the erased set while keeping it orthogonal to retained concepts. Erasing 40 impressionists concentrates that direction onto soft, plein-air impressionist landscape — the dominant idiom of the set.

- **Pissarro is that direction**: the most prototypical impressionist of the three, near-indistinguishable from the Monets and Sisleys being erased. When the null-space runs low on degrees of freedom, he is the concept it cannot keep orthogonal — and his drift *grows with N* as each added impressionist reinforces his axis.
- **Gauguin** (flat cloisonnist planes) and **Seurat** (rigid pointillist dots) sit *off* that axis; their distinctive techniques give the projection room to protect them.

*Caveat:* this is a single collapsing artist, so the geometric account is the best-supported explanation rather than proven. The clean confirmation would be to invert the experiment — erase 40 *pointillists* and predict Seurat becomes the casualty while Pissarro survives.

---

## 4. Pitfalls corrected (why the result is trustworthy)

Three measurement traps each produced a convincing-but-wrong result before being caught. Each safeguard in the pipeline exists because of one of them.

1. **Pixel MSE could not separate damage from noise.** Our first metric scored a *retained, protected* artist (Cézanne) at MSE > 10,000 on some seeds while a supposedly "damaged" neighbor scored ~440 — MSE is dominated by seed-to-seed composition variation. We switched to CLIP style drift.

2. **The NSFW safety checker faked a collapse.** An early run showed a dramatic "Gauguin collapses to 0.267" signal that was entirely an artifact: Stable Diffusion's safety checker blanks nude-heavy painters (Gauguin's Tahitian series) to solid black, and black-vs-painting maxes out CLIP distance. Fix: generate in fp32 with the safety checker disabled, and exclude any black frame from the metric (reporting valid-seed counts).

3. **A neighbor "suppression" claim reversed under measurement.** An initial single-concept framing claimed SPEED suppressed a neighbor's color saturation; direct measurement showed saturation actually *rose* slightly. Discarding that confirmation-biased false start motivated the controlled, multi-seed, control-calibrated design used here.

Every pitfall would have produced a *more dramatic* headline than the real finding. The selective Pissarro leak survived all three corrections — which is precisely why we trust it.

---

## 5. Conclusion

Closed-form, null-space erasure is a genuinely strong idea. SPEED erases its targets cleanly, preserves neighbors with mathematical precision, and stays robust even when several mixed concepts are removed at once. It is fast, training-free, and hard to fault for the use cases it was designed around.

Its guarantee is bounded by **rank saturation** — a limit its own authors name and never demonstrate. We located where it begins to bite: under concentrated mass erasure of a single dense semantic cluster, protection breaks *selectively* for the one retained concept most entangled with the erased set, while distinct neighbors and the broader capability survive. It is a real, reproducible, precisely-bounded failure — not a catastrophe, but a crack exactly where the geometry says it should appear.

The implication for future work: as long as erasure operates by projecting in cross-attention space, the most entangled neighbors will be the first to leak under load. Robust large-scale unlearning will need mechanisms that disentangle concepts at the feature level, so that protecting a neighbor does not compete for the same finite degrees of freedom used to erase its twin.

---

## 6. Reproducibility

- **Branch:** `experiment-3-rank-saturation`
- **Scripts:** `experiment3/scripts/probe_rank_saturation.py` (generation, fp32 + safety-checker-off), `experiment3/scripts/analyze_rank_saturation.py` (CLIP drift, black-frame exclusion), `experiment3/scripts/slurm_rank_saturation.sh` (builds nested 5/10/20/40 checkpoints, generates, analyzes).
- **Data:** `experiment3/results/rank_saturation/rank_drift.csv` and `experiment3/results/multi_concept/clip_drift.csv` (each cell n=4, zero corrupt frames).
