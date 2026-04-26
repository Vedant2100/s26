# EE 243 — Assignment 1 Report

---

## Problem 1: Image Denoising via Multiresolution Decomposition

### Approach

I built a Gaussian–Laplacian pyramid from scratch to denoise `cameraman_noisy.jpg`. At each pyramid level, the image is blurred with a hand-written 5×5 Gaussian kernel (σ=1.0), downsampled by 2×, and the difference between the original and upsampled-downsampled version is stored as the Laplacian (high-frequency detail) layer.

To denoise, I reconstruct from the coarsest Gaussian level upward, but scale each Laplacian layer by a factor < 1 (the **laplacian threshold**) before adding it back. Since noise is predominantly high-frequency, this attenuates noise while preserving the low-frequency structure from the Gaussian base levels.

### Noisy Input Image

![Noisy cameraman image](report_images/p1_cell2_0.png)

### Experiments and Findings

**Threshold = 0.3, Levels = 2, σ = 1.0** — Good denoising. Most noise removed while retaining the main structure of the cameraman image.

![Threshold 0.3, Levels 2, Sigma 1.0](report_images/p1_cell5_1.png)

**Threshold = 0.6, Levels = 2, σ = 1.0** — More detail is preserved but also more residual noise, since 60% of the high-frequency content is kept.

![Threshold 0.6, Levels 2, Sigma 1.0](report_images/p1_cell7_2.png)

**Threshold = 0.3, Levels = 4, σ = 1.0** — Too aggressive — with 4 levels of decomposition, significant structural information is lost and cannot be recovered even after adding back the detail layers.

![Threshold 0.3, Levels 4, Sigma 1.0](report_images/p1_cell9_3.png)

**Threshold = 0.3, Levels = 2, σ = 1.5** — The higher standard deviation produces a stronger blur at each level, causing additional loss of edge structure compared to σ = 1.0.

![Threshold 0.3, Levels 2, Sigma 1.5](report_images/p1_cell11_4.png)

**Conclusion:** A 2-level pyramid with σ = 1.0 and threshold = 0.3 provides the best balance between noise removal and structure preservation. Increasing levels or sigma leads to over-smoothing, while increasing the threshold retains more noise.

---

## Problem 2: Feature Matching with VGG-16

### Approach

Using the provided template, I extracted feature maps from a pre-trained VGG-16 model at two intermediate layers (5th and 17th). Saliency-based keypoints are detected from the feature maps, L2-normalized descriptors are computed at each keypoint, and matching is performed using a brute-force matcher with Lowe's ratio test.

### Experiments and Findings

**Layer 5, Ratio = 0.5** — 0 matches. Early layers capture low-level features (edges, textures) which are not discriminative enough to distinguish individual keypoints between two views. All descriptors look similar, so the ratio test rejects everything.

![Layer 5, Ratio 0.5](report_images/p2_cell7_0.png)

**Layer 17, Ratio = 0.5** — 0 matches. Deeper features are more semantic, but the strict 0.5 ratio threshold still rejects all matches.

![Layer 17, Ratio 0.5](report_images/p2_cell9_1.png)

**Layer 17, Ratio = 0.8** — 1 match found. With a more lenient threshold, one valid correspondence passes the ratio test.

![Layer 17, Ratio 0.8](report_images/p2_cell11_2.png)

**Conclusion:** Deeper VGG layers produce more discriminative descriptors for feature matching. The matching ratio threshold controls the strictness of the test — lower values demand highly distinctive matches and may reject valid correspondences, while higher values accept more matches at the risk of including false positives.

---

## Problem 3: PCA Image Reconstruction

### Approach

I vectorized 110 face images (each 77,760 pixels) from the dataset, computed the mean image, and subtracted it to obtain zero-mean vectors. Since d >> N, I used the efficient dual PCA trick: computing the smaller N×N matrix K = X^TX/(N−1) and recovering the full eigenvectors as U = X·V.

### (a) & (b) Eigenvalue Analysis

Only **8 components** are needed to explain 80% of the total variance. The eigenvalues decay rapidly, indicating that face images share significant structural similarity and can be compactly represented.

![Eigenvalue analysis and cumulative variance](report_images/p3_cell7_0.png)

### (c) & (d) Reconstruction with k=8

Reconstructing with 8 components yields an MSE of **854.84**. The reconstructed faces capture overall shape, lighting, and coarse features but lose fine details like expression lines.

![Reconstruction comparison: original vs reconstructed](report_images/p3_cell10_1.png)

### (e) Impact of Varying k

![Reconstruction at different k values](report_images/p3_cell12_2.png)

![MSE vs number of components](report_images/p3_cell12_3.png)

Error decreases monotonically with more components, with diminishing returns after ~20 components. At k=100 (91% of total), reconstruction is nearly perfect.

| Components (k) | Relative Error |
|---|---|
| 5 | 0.175 |
| 8 (80% variance) | 0.146 |
| 10 | 0.133 |
| 20 | 0.099 |
| 50 | 0.055 |
| 100 | 0.002 |

### (f) Clustering

Using KMeans (k=10, matching the actual number of individuals) on the 8-dimensional PCA coefficients yields a clustering accuracy of **58.18%**. Some subjects (e.g., subject01, subject02, subject05, subject06) cluster cleanly into single groups, while others get mixed. PCA coefficients capture gross appearance similarities but are not sufficient for reliable person identification, likely because lighting and expression variation within the same individual can exceed inter-person differences at this low dimensionality.

---

## Problem 4: EM/GMM Image Segmentation

### Approach

I implemented the Expectation-Maximization algorithm for a Gaussian Mixture Model from scratch (no sklearn GMM). Each pixel's RGB values serve as the feature vector. The model initializes K=5 Gaussians with random pixel means and covariances of 10·I for numerical stability, then iterates E-step (compute responsibilities) and M-step (update parameters).

### Results

![Original peppers image](report_images/p4_cell1_0.png)

![Original vs Segmented (K=5)](report_images/p4_cell4_1.png)

With K=5, the GMM separates the image into major color-intensity regions and preserves the main pepper shapes, shadow regions, and overall scene structure. Similar shades are merged together, producing a cleaner, more abstract version of the original.

However, the segmentation is coarse rather than precise — some parts of the same pepper are split into different clusters because of lighting and shading variations, and some regions appear speckled because the model uses only RGB color features with no spatial information. Overall, the result is effective for coarse color-based segmentation but not sufficient for precise object boundaries.

---

## Problem 5: FDA Fourier Style Transfer

### Approach

I implemented the Fourier Domain Adaptation (FDA) method from the CVPR 2020 paper. The core idea: in the frequency domain, an image's **amplitude** encodes appearance (color, brightness, tone) while its **phase** encodes structure (edges, shapes). By swapping only the **low-frequency amplitude** from the target image into the source, we transfer the target's visual style while preserving the source's content.

The key equation: A_new = M_β · A_target + (1 − M_β) · A_source, where M_β is a binary mask selecting the center low-frequency region. The stylized image is then reconstructed via inverse FFT using A_new with the source's original phase.

### Result

![Source, Target, and Stylized images](report_images/p5_cell3_0.png)

With β=0.01, the stylized image retains the source image's structure and edges while adopting the target image's overall color tone and lighting characteristics. The effect is subtle at this small β value, as only a very narrow band of low frequencies is swapped.
