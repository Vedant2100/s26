# Exhaustive Cross-Dataset Evaluation Matrix

## Key Methodological Differences

Before reading the table, understand these critical distinctions:

| Dimension | EFFORT / UCF | UNITE | SigLIP Foundation |
|---|---|---|---|
| **Weight Source** | Author-provided (downloaded) | Self-trained by us | Self-trained by us |
| **Training Framework** | DeepfakeBench (`training/train.py`) | Custom script (`train_unite_final_dfbench.py`) | Custom script (`train_safety_classifier.py`) |
| **Preprocessing** | DeepfakeBench face-crop pipeline | SigLIP transforms via `build_ff_dfbench_transform` | SigLIP transforms |
| **Evaluation Framework** | DeepfakeBench (`training/test.py`) | Custom `evaluate()` function | Custom eval loop |
| **Reported Metric** | **AUC** | **Accuracy** | **AUC** |

> [!WARNING]
> **EFFORT/UCF report AUC. UNITE reports Accuracy. These are NOT directly comparable in the same column.** AUC measures ranking quality across all thresholds; accuracy is a single-threshold binary count. To make them comparable, we either need to add AUC to UNITE or recompute EFFORT/UCF accuracy.

---

## Category 1: Zero-Shot Cross-Domain Transfer
*"I have weights trained on Dataset A. How well do they work on Dataset B with NO fine-tuning?"*

| Method | Weights From | Eval Dataset | Metric | Value | Slurm Job / Source |
|---|---|---|---|---|---|
| EFFORT | Author pretrained (FF++) | Celeb-DF-v1 | AUC | **93%** | `adaptation.slurm` (inference step) |
| EFFORT | Author pretrained (FF++) | FF++ | AUC | **92%** | `adaptation.slurm` (inference step) |
| UCF | Author pretrained (FF++) | Celeb-DF-v1 | AUC | **86%** | DeepfakeBench test.py |
| UCF | Author pretrained (FF++) | FF++ | AUC | **98%** | DeepfakeBench test.py |
| UNITE | Self-trained (FF++) | Celeb-DF-v1 | Acc | **⏳ PENDING** | Job 251457 |
| UNITE | Self-trained (FF++) | FF++ | Acc | **86.71%** | Job 251300 (training eval) |

---

## Category 2: Domain Adaptation (Fine-tuning)
*"I take weights pre-trained on FF++, fine-tune them on Celeb-DF. How do they perform on Celeb-DF AND does performance degrade on the original FF++ domain?"*

| Method | Base Weights | Adapted On | Eval Dataset | Metric | Value | Slurm Job / Source |
|---|---|---|---|---|---|---|
| EFFORT | Author pretrained (FF++) | Celeb-DF-v1 | Celeb-DF-v1 | AUC | **99%** | `adaptation.slurm` |
| EFFORT | Author pretrained (FF++) | Celeb-DF-v1 | FF++ | AUC | **85%** | `adaptation.slurm` (backwards test) |
| UCF | Author pretrained (FF++) | Celeb-DF-v1 | Celeb-DF-v1 | AUC | *Not run* | — |
| UCF | Author pretrained (FF++) | Celeb-DF-v1 | FF++ | AUC | *Not run* | — |
| UNITE | Self-trained (FF++) | Celeb-DF-v1 | Celeb-DF-v1 | Acc | **92.00%** | Job 251334 |
| UNITE | Self-trained (FF++) | Celeb-DF-v1 | FF++ | Acc | **⏳ PENDING** | Job 251458 |

---

## Category 3: Supervised Training (From Scratch on Target)
*"I train the model directly on the target dataset. What's the ceiling performance?"*

| Method | Trained On | Eval Dataset | Metric | Value | Slurm Job / Source |
|---|---|---|---|---|---|
| UNITE | Celeb-DF-v1 | Celeb-DF-v1 | Acc | **98.00%** | Job 251330 |
| UNITE | FF++ | FF++ | Acc | **86.71%** | Job 251300 |
| SigLIP Foundation | Celeb-DF-v1 | Celeb-DF-v1 | AUC | **93.76%** | Job 251328 |
| SigLIP Foundation | Celeb-DF-v1 | Celeb-DF-v1 | Acc | **91.00%** | Job 251328 |
| SigLIP Foundation | FF++ | FF++ | AUC | **❌ FAILED** | Job 251325 (SIGABRT) |

---

## Gaps & Action Items

| # | Gap | Impact | Fix |
|---|---|---|---|
| 1 | **UNITE reports Accuracy, not AUC** | Cannot directly compare UNITE vs EFFORT/UCF in same column | Add AUC computation to UNITE's `evaluate()` |
| 2 | **UNITE zero-shot on Celeb-DF still pending** | Missing key transfer metric | Job 251457 in queue |
| 3 | **UNITE backwards eval still pending** | Missing catastrophic forgetting metric | Job 251458 in queue |
| 4 | **SigLIP on FF++ crashed** | Missing foundation model baseline on FF++ | Re-run with `--num_workers 2` |
| 5 | **UCF adaptation not run** | Missing UCF fine-tuning comparison | Run UCF adaptation similar to EFFORT |
| 6 | **Different preprocessing pipelines** | Potential confound | Acceptable if noted in paper |
