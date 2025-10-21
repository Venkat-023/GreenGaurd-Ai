Green Guard AI 🌱

Green Guard AI is a state-of-the-art plant disease detection and health monitoring system leveraging deep learning with UGNet for precise leaf and disease segmentation. The system enables farmers and agronomists to detect plant diseases early and estimate severity, empowering proactive crop management.
Project Overview

Green Guard AI leverages UGNet (U-Net with Gated Convolutions) for plant leaf segmentation and combines patch-based early and late fusion analysis for robust disease detection and severity estimation.

Segments leaves and diseased regions accurately.

Handles multiple crops and disease types.

Estimates disease severity relative to the original predicted region.

The system is designed for real-world agricultural applications to provide actionable insights for farmers.

Features

UGNet Segmentation: High-precision leaf and disease region segmentation.

Patch-Based Analysis: Supports early and late fusion for robust disease detection.

Severity Estimation: Accurately calculates disease severity.

User-Friendly: Ready for integration with web or mobile apps.

Extensible: Add new crops, diseases, or datasets easily.

System Architecture
flowchart LR
A[Input Image] --> B[Preprocessing]
B --> C[UGNet Segmentation]
C --> D[Patch Extraction]
D --> E[Early Fusion Module]
D --> F[Late Fusion Module]
E --> G[Disease Classification]
F --> G
G --> H[Severity Estimation]
H --> I[Output: Disease Type & Severity]

Methodology & Intuition

Image Acquisition: Collect high-resolution plant images.

Preprocessing: Normalize, augment, and handle missing data.

UGNet Segmentation: Segment leaves and disease regions using gated convolutions for sharper edges.

Patch-Based Analysis: Divide segmented regions into patches for multi-scale analysis.

Early Fusion: Merge features before classification.

Late Fusion: Merge predictions after independent patch classification.

Severity Estimation: Predict disease severity based on the original segmented area, avoiding patch spacing errors.

Output Generation: Annotated images with disease type and severity scores.

Technologies Used

Python 3.10+

TensorFlow / Keras

OpenCV, NumPy, Pandas

Matplotlib / Seaborn for visualization

UGNet Architecture for segmentation
Results & Evaluation 📊

UGNet Training (70 epochs)

Training Performance:
| Epoch         | Dice Coefficient | IoU Metric | Loss   |
| ------------- | ---------------- | ---------- | ------ |
| Initial (1)   | 0.4508           | 0.2972     | 0.9401 |
| Midpoint (35) | 0.7225           | 0.5705     | 0.4981 |
| Final (70)    | 0.8499           | 0.7421     | 0.2697 |
Validation Performance:
| Epoch         | Val Dice Coefficient | Val IoU Metric | Val Loss |
| ------------- | -------------------- | -------------- | -------- |
| Initial (1)   | 0.2359               | 0.1349         | 7.9577   |
| Midpoint (35) | 0.6780               | 0.5233         | 0.5888   |
| Final (70)    | 0.7398               | 0.5952         | 0.4936   |
Insights:

Steady improvement in Dice and IoU across epochs.

Validation metrics confirm good generalization with minimal overfitting.

Segmentation quality is sufficient for downstream patch-based disease classification and severity estimation.

Visualizations:

Dice & IoU curves over epochs

Input vs predicted masks

Severity heatmaps
