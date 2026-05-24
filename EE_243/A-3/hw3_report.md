# EE 243 — Assignment 3 Report

**Name:** Vedant Borkute  
**Date:** May 24, 2026

---

## Problem 1 — Homography Estimation

Using OpenCV's `findChessboardCorners`, we successfully extracted a one-to-one correspondence of corner points from **Image 2** and **Image 9**. Using the Direct Linear Transform (DLT) algorithm without built-in OpenCV homography estimation functions, the estimated Homography matrix $H$ (where $P_9 = H P_2$) is:

$$
H = \begin{bmatrix}
-2.26900384 & 5.06187321 & 933.200306 \\
-0.929268948 & -1.15354922 & 1348.65019 \\
0.00170613214 & 0.00829184924 & 1.00000000
\end{bmatrix}
$$

The corner reprojection Root Mean Square Error (RMSE) when mapping points from Image 2 to Image 9 using this matrix is **9.0721 px**.

Below is the visualization of the mapped points:

![Homography Visualization](/Users/EndUser/Downloads/Repos/s26/EE_243/A-3/Problem1/images/output_0.png)

---

## Problem 2 — Fundamental Matrix Estimation

Corner points were detected in the left and right stereo images (`viprectification_deskLeft.png` and `viprectification_deskRight.png`) using the Good Features to Track algorithm. We extracted 8x8 patches around these points and processed them through a **ResNet50** model to obtain feature descriptors. 

By matching these descriptors and iteratively applying the **8-point algorithm** with RANSAC, we evaluated candidate fundamental matrices using the **Sampson distance**.

Out of 48 successfully matched points, the fundamental matrix with the **least mean error** across the dataset is:

$$
F = \begin{bmatrix}
-9.08511326 \times 10^{-6} & 1.00310634 \times 10^{-3} & -1.66663073 \times 10^{-1} \\
-9.63109795 \times 10^{-4} & -1.06204141 \times 10^{-5} & 2.02329772 \times 10^{-1} \\
1.76859614 \times 10^{-1} & -2.24650099 \times 10^{-1} & 9.21703401 \times 10^{-1}
\end{bmatrix}
$$

The corresponding **Mean Sampson Error** for this matrix is **88.948381**.

---

## Problem 3 — 3D Gaussian Splatting

A short video was captured in a courtyard on the UCR campus. We extracted 363 frames and used COLMAP (via a compute node) to recover the ground-truth camera poses and intrinsics. We then optimized a **3D Gaussian Splatting** model over the dataset and generated renders corresponding to each frame's camera pose. 

The pipeline was executed successfully on an HPC cluster, leading to the following final results:

*   **Number of training iterations:** 30,000
*   **Final number of Gaussians:** 572,195
*   **PSNR on training views:** 23.483
*   **SSIM on training views:** 0.6953

Below is a frame from the generated side-by-side comparison video (Ground Truth vs. 3D Gaussian Splatting):

![Ground Truth vs 3DGS Video Frame](/Users/EndUser/Downloads/Repos/s26/EE_243/A-3/Problem3/gt_vs_3dgs_frame.png)

### Failure Modes Observed
While the overall 3DGS render successfully captured the geometry, structural depth, and lighting of the courtyard, a few failure modes were visually apparent when comparing the render side-by-side with the ground truth:

1.  **Floaters in Free Space:** Small, hazy, semi-transparent artifacts (floaters) are visible floating in the air, particularly around the edges of thin structures like the tree leaves and lamp posts against the bright sky.
2.  **Blur on Fine Textures:** The high-frequency fine details in the scene, such as the texture of the grass and the individual leaves on the bushes, appear noticeably softer and blurrier than the ground truth.
3.  **Exposure Drift / Specularities:** 3DGS struggles slightly with view-dependent reflections on the large glass windows in the background, causing them to look somewhat flat or inconsistent as the camera moves. 

---
