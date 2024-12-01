# Overview

As an overview:

1. Train an SAE on activations.
2. Train a linear probe on semantic segmentation task using ADE20K.
3. Establish baseline metrics for the linear probe.
4. Automatically identify feature vectors in the SAE's \(W_dec\) matrix for each class in ADE20K.
5. Suppress those features in the vision transformer's activations before applying the linear probe.
6. Record class-specific metrics before and after suppression.

Details can be found below.

# Train an SAE on ViT Activations

# Train a Linear Probe on Semantic Segmentation

# Establish Linear Probe Baseline Metrics

# Identify Class-Specific Feature Vectors in the SAE
