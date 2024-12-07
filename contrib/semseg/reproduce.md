This sub-module reproduces the results from Section 4.2 of our paper.

# Overview

As an overview:

0. Record ViT activations for ADE20K.
1. Train an SAE on activations.
2. Train a linear probe on semantic segmentation task using ADE20K.
3. Establish baseline metrics for the linear probe.
4. Automatically identify feature vectors in the SAE's \(W_dec\) matrix for each class in ADE20K.
5. Suppress those features in the vision transformer's activations before applying the linear probe.
6. Record class-specific metrics before and after suppression.

Details can be found below.

# Record ViT Activations for SAE and Linear Probe Training

# Train an SAE on ViT Activations

# Train a Linear Probe on Semantic Segmentation

Now train a linear probe on the activations.

```sh
uv run python -m contrib.semseg train \
  --train-acts.shard-root /local/scratch/stevens.994/cache/saev/a860104bf29d6093dd18b8e2dccd2e7efdfcd9fac35dceb932795af05187cb9f/ \
  --train-acts.no-scale-mean \
  --train-acts.no-scale-norm \
  --val-acts.shard-root /local/scratch/stevens.994/cache/saev/c6756186d1490ac69fab6f8efb883a1c59d44d0594d99397051bfe8e409ca91d/ \
  --val-acts.no-scale-mean \
  --val-acts.no-scale-norm \
  --imgs.root /research/nfs_su_809/workspace/stevens.994/datasets/ade20k/ \
  --sweep contrib/semseg/sweep.toml
```


# Establish Linear Probe Baseline Metrics

# Identify Class-Specific Feature Vectors in the SAE

```sh
uv run python -m contrib.semseg visuals \
  --sae-ckpt checkpoints/ercgckr1/sae.pt \
  --acts.shard-root /local/scratch/stevens.994/cache/saev/a860104bf29d6093dd18b8e2dccd2e7efdfcd9fac35dceb932795af05187cb9f/ \
  --acts.no-scale-mean \
  --acts.no-scale-norm \
  --imgs.root /research/nfs_su_809/workspace/stevens.994/datasets/ade20k/
```

This will tell you to run a particular command in order to generate visuals for use with `saev.interactive.features`.

# Manipulate ViT Activations

```sh
uv run python -m contrib.semseg manipulate \
  --acts.shard-root /local/scratch/stevens.994/cache/saev/c6756186d1490ac69fab6f8efb883a1c59d44d0594d99397051bfe8e409ca91d/ \
  --acts.no-scale-mean \
  --acts.no-scale-norm \
  --imgs.root /research/nfs_su_809/workspace/stevens.994/datasets/ade20k/ \
  --sae-ckpt checkpoints/ercgckr1/sae.pt \
  --probe-ckpt checkpoints/semseg/lr_0_003__wd_0_1/model_ep199_step4000.pt \
  --ade20k-classes 29 \
  --sae-latents 8541 5818 10230
```
