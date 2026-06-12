# Lineage Diagram

```mermaid
graph LR
  SD[Stable Diffusion<br/>(Rombach et al. CVPR’22)] --> ESD[Erasing Concepts (Gandikota et al. ICCV’23)]
  SLD[Safe Latent Diffusion<br/>(Schramowski et al. CVPR’23)] --> ESD
  SD --> Ablation[Ablating Concepts (Kumari et al. ICCV’23)]
  ESD --> UCE[Unified Concept Editing (Gandikota et al. WACV’24)]
  UCE --> MACE[Mass Concept Erasure (Lu et al. CVPR’24)]
  UCE --> RECE[RECE (Gong et al. ECCV’24)]
  SD --> AdvUnlearn[AdvUnlearn (Zhang et al. NeurIPS’24)]
  SD --> RACE[R.A.C.E. (Kim et al. ECCV’24)]
  SD --> FMN[Forget-Me-Not (Zhang et al. CVPRW’24)]
  SD --> GLoCE[GLoCE (Lee et al. CVPR’25)]
  SD --> CPE[CPE (Lee et al. ICLR’25)]
  GLoCE --> NLCE[NLCE (Shi et al. CVPR’26)]
  SD --> HiRM[HiRM (Lee et al. ICLR’26)]
  MACE --> SPEED[SPEED (Gupta et al. ICLR’26)]
  MACE --> ETC[ETC (Seo et al. CVPR’26)]
  SD --> Prototype[Prototype-Guided CE (Cai et al. CVPR’26)]
```

# Paper Comparison

| Paper (Citation) | Year (Venue) | Summary & Key Contributions | Code / Weights | Cites / Extended by |
|---|---|---|---|---|
| **Erasing Concepts from Diffusion Models**<br/>R. Gandikota *et al.* (ICCV 2023) | 2023 (ICCV) | Fine-tunes a text-conditional diffusion model to **permanently remove** a target concept (e.g. artist style, object class, NSFW content) using only its name with negative guidance. Benchmarks on par with Safe Latent Diffusion. | Code: GitHub✅<br/>Weights: fine-tuned checkpoints✅ | Cites Safe Latent Diffusion; extended by UCE, FMN, etc. |
| **Ablating Concepts in Diffusion Models**<br/>N. Kumari *et al.* (ICCV 2023) | 2023 (ICCV) | Fine-tunes the model so that images with the target concept produce the same distribution as an **anchor concept** (e.g. “Grumpy Cat”→“cat”); this **prevents generating** the target concept while preserving related content. | Code: GitHub✅ (uses Stable Diffusion weights) | Cites: Safe Latent Diffusion; concurrent with ESD. |
| **Unified Concept Editing (UCE)**<br/>R. Gandikota *et al.* (WACV 2024) | 2024 (WACV) | **Closed-form model editing** (no training) that simultaneously debiases, erases styles, and censors content. Scales to **hundreds of edits at once** on text-to-image models. | Code: GitHub✅ (works on SD, SDXL) | Cites: ESD, ablation, etc.; extended by MACE, SPEED, ETC. |
| **Mass Concept Erasure (MACE)**<br/>S. Lu *et al.* (CVPR 2024) | 2024 (CVPR) | LoRA-based fine-tuning framework to erase **up to 100 concepts**. Uses closed-form updates and **LoRA adapters** to erase multiple concepts with one-shot updates. | Code: GitHub✅ (builds on SD) | Cites: UCE; extended by SPEED, ETC. |
| **Forget-Me-Not (FMN)**<br/>E. Zhang *et al.* (CVPRW 2024) | 2024 (CVPRW) | Lightweight attention-resteering method to “forget” specific IDs, objects, or styles in ~30 sec without harming other capabilities. Introduces a *Memorization Score* and ConceptBench for evaluation. | Code: GitHub✅<br/>Weights: None (uses SD) | Cites: ESD, UCE; extended by none (baseline for comparison). |
| **RECE (Reliable & Efficient CE)**<br/>C. Gong *et al.* (ECCV 2024) | 2024 (ECCV) | Closed-form concept erasure: iteratively compute new token embeddings that align with “safe” (negated) concepts. Achieves quick erasure (seconds) with minimal drift. | Code: GitHub✅ (requires SD) | Cites: UCE; extended by none known. |
| **AdvUnlearn**<br/>Y. Zhang *et al.* (NeurIPS 2024) | 2024 (NeurIPS) | Integrates *adversarial training* into concept erasure. Optimizes a text encoder for erasing (nudity, objects, styles) **robust** to adversarial prompts, balancing erasure with generation quality. | Code: GitHub✅<br/>Weights: plug-and-play model<br/> | Cites: standard erasure methods; extended by RACE. |
| **R.A.C.E. (Robust Adversarial CE)**<br/>C. Kim *et al.* (ECCV 2024) | 2024 (ECCV) | Adversarial training framework for concept erasure. Specifically defends against prompt attacks by training on adversarial text embeddings, cutting attack success rate by ~30 points for “nudity”. | Code: GitHub✅ (builds on AdvUnlearn ideas) | Cites: AdvUnlearn; extended by none known. |
| **GLoCE (Gated Low-Rank CE)**<br/>B.H. Lee *et al.* (CVPR 2025) | 2025 (CVPR) | *Training-free*, **localized** erasure via a gated low-rank adapter on UNet activations. Only the target region’s concept is suppressed, preserving the rest of the image. | Code: GitHub✅ | Cites: None prior; extended by NLCE. |
| **CPE (Concept Pinpoint Eraser)**<br/>B.H. Lee *et al.* (ICLR 2025) | 2025 (ICLR) | Uses a *residual attention gate* to erase a target concept while minimally affecting others. Focuses on spatially precise removal. | Code: GitHub✅ | Cites: None; concept erasure for fine control. |
| **NLCE (Neighbor-Aware LCE)**<br/>Z. Shi *et al.* (CVPR 2026) | 2026 (CVPR) | Training-free three-stage pipeline: (1) attenuate target in embeddings, (2) attention-guided gating, (3) localized hard erasure. **Preserves semantically related (“neighbor”) concepts** while removing target. | Code: GitHub✅ | Cites: GLoCE; no known extensions yet. |
| **SPEED**<br/>A. Gupta *et al.* (ICLR 2026) | 2026 (ICLR) | Training-free *null-space editing*: iteratively projects out target concept directions via importance pruning and embedding constraints. Scales to **hundreds of concepts** with modules for diversity (IPF, DPA, IEC). | Code: GitHub✅<br/>Models: Pretrained released✅ | Cites: MACE, UCE; complements ETC. |
| **ETC (Erasing Thousands of Concepts)**<br/>H. Seo *et al.* (CVPR 2026) | 2026 (CVPR) | Fine-tuning with *optimal transport* and mixture-of-experts adapters to erase **2000+ concepts** simultaneously, preserving the rest. | Code: GitHub✅ | Cites: MACE, UCE; large-scale concept erasure. |
| **Prototype-Guided CE**<br/>Y. Cai *et al.* (CVPR 2026) | 2026 (CVPR) | Training-free method for **broad/abstract concepts**: cluster latent embeddings to form concept “prototypes”, then use them as negative prompts at inference. Achieves much more reliable removal of wide-ranging concepts (e.g. “violence”). | Code: GitHub✅ | Cites: none prior; addresses limitations of prior methods. |

**Notes:** All methods build on Stable Diffusion (latent diffusion) and often use classifier-free guidance. Code links point to official repos; most use public base models (e.g. Stable Diffusion). Checkpoints for base models are public; only ESD and SPEED explicitly release edited model weights. 

# Open Problems & Project Ideas

- **Broad/Abstract Concepts:** Existing methods struggle to erase complex concepts (e.g. “violence”, “sexual”) that manifest in many ways. *Candidate project:* Extend prototype-guided erasure or develop new clustering-based guidance to cover diverse concept modes (code available).  
- **Semantic Context Preservation:** Removing a concept often inadvertently degrades related content (neighbor classes). *Candidate project:* Build on **Localized Erasure** methods (GLoCE/NLCE) to improve fidelity of non-target parts, using the available code of GLoCE or NLCE.  
- **Adversarial Robustness:** Adversarial prompts can reverse erasure. Designing defenses and robust training remains open. *Candidate project:* Implement and test AdvUnlearn or R.A.C.E. (code exists) against new attacks.  
- **Scalability & Efficiency:** Scaling erasure to many concepts (and on-device) is challenging. *Candidate project:* Experiment with **SPEED** or **ETC** (both have public code/models) to extend their null-space edits or mixture-of-experts schemes to new concept sets.  
- **Evaluation & Metrics:** Measuring erasure (without harm) is nontrivial. (Forget-Me-Not introduced M-Score for ID/object/style forgetting). *Candidate project:* Build benchmarks integrating public code (e.g. ConceptBench) to evaluate and compare erasure methods.  

Each above area is ripe for a class project; candidates with public code (UCE, MACE, GLoCE, NLCE, SPEED, Prototype) provide a solid starting point. All mentioned methods have open repositories (links above) and rely on standard diffusion checkpoints.  

**Sources:** The above summaries and claims are drawn from the papers and code linked (see citations).