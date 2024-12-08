You can reproduce our semantic segmentation control experiments from our preprint by following these instructions.

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
  --train-acts.shard-root /local/scratch/$USER/cache/saev/a860104bf29d6093dd18b8e2dccd2e7efdfcd9fac35dceb932795af05187cb9f/ \
  --val-acts.shard-root /local/scratch/$USER/cache/saev/c6756186d1490ac69fab6f8efb883a1c59d44d0594d99397051bfe8e409ca91d/ \
  --imgs.root /nfs/$USER/datasets/ade20k/ \
  --sweep contrib/semseg/sweep.toml
```

# Establish Linear Probe Baseline Metrics

```sh
uv run python -m contrib.semseg validate \
  --imgs.root /nfs/$USER/datasets/ade20k/ \
  --acts.shard-root /local/scratch/$USER/cache/saev/51cf4f907e562213fff9cc094b0b5259e546f1f9f72633a725888e15f798bc30/
```

Then you can look in `./logs/contrib/semseg` for `hparam-sweeps.png` to see what learning rate/weight decay combination is best.

# Identify Class-Specific Feature Vectors in the SAE

Given that you have an SAE trained on the DINOv2 ImageNet-1K training data

```sh
uv run python -m contrib.semseg visuals \
  --sae-ckpt checkpoints/ercgckr1/sae.pt \
  --acts.shard-root /local/scratch/$USER/cache/saev/a860104bf29d6093dd18b8e2dccd2e7efdfcd9fac35dceb932795af05187cb9f \
  --imgs.root /nfs/$USER/datasets/ade20k
```

This will tell you to run a particular command in order to generate visuals for use with `saev.interactive.features`.
