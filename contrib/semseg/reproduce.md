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
  --train-acts.shard-root $TRAIN_SHARDS \
  --val-acts.shard-root $VAL_SHARDS \
  --imgs.root /nfs/$USER/datasets/ade20k/ \
  --sweep contrib/semseg/sweep.toml
```

# Establish Linear Probe Baseline Metrics

```sh
uv run python -m contrib.semseg validate \
  --imgs.root /nfs/$USER/datasets/ade20k/ \
  --acts.shard-root $VAL_SHARDS
```

Then you can look in `./logs/contrib/semseg` for `hparam-sweeps.png` to see what learning rate/weight decay combination is best.
