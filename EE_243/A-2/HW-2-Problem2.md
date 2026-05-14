# HW-2 Problem 2 (Extracted from PDF)

## Problem Statement

2. (5 pts) In this problem, you will train a Vision Transformer (ViT)-style model entirely from scratch. Your code should have the full pipeline: data loading, model definition, training, evaluation, and analysis. Create a single notebook named `ViT training.ipynb` that is runnable end-to-end on Google Colab (free tier).

### (a) Dataset Selection

Choose any two image-classification datasets, each containing at least 10 unique classes (e.g. CIFAR-10, CIFAR-100, STL-10, EuroSAT, Oxford-Pets, Flowers-102, Food-101, etc.). Please do not use MNIST and its variants (Fashion-MNIST, KMNIST, EMNIST, etc.) as these datasets are too simple to meaningfully exercise a transformer architecture.

### (b) Model

Design and implement a ViT-like architecture. Your model must include, at minimum:

- A patch-embedding layer,
- At least two transformer encoder blocks (multi-head self-attention + feed-forward network),
- Positional encoding (learned or fixed),
- A classification head.

You are free to experiment with patch size, embedding dimension, number of heads, depth, dropout, and any other architectural choices. Hybrid designs that incorporate convolutional components (e.g. convolutional patch embeddings) are also permitted.

### (c) Training

Train your model on both datasets separately using cross-entropy loss. Track and record training loss, training accuracy, and test/validation accuracy per epoch. Your test accuracy must be >= 40% on both datasets.

### (d) Robustness to Domain Shifts

After training, evaluate your model’s robustness by adding Gaussian noise/blur to the test images at various noise levels. Report accuracy at each noise level for both datasets. Discuss how performance degrades and whether the two datasets exhibit different sensitivity to noise.

### (e) Plots & Discussion

Include the following plots for each dataset:

- Training loss vs. epoch,
- Training accuracy and test accuracy vs. epoch (on the same axes),
- Test accuracy vs. Gaussian noise level sigma.

From your training curves, state whether the model is over-fitted, under-fitted, or well-fitted, and explain your reasoning. Compare behaviour across the two datasets.

