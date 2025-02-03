You can reproduce our semantic segmentation control experiments from our preprint by following these instructions.

As an overview:

1. Record ViT activations for ADE20K.
2. Train a linear probe on semantic segmentation task using ADE20K.
3. Establish baseline metrics for the linear probe.
4. Manipulate the activations using the proposed SAE features.
5. Be amazed. :)

Details can be found below.

# Record ViT Activations for SAE Inference

For SAE inference, we want the second-to-last layer (as discussed).

# Train a Linear Probe on Semantic Segmentation

Now train a linear probe on the activations.

```sh
uv run python -m contrib.semseg train \
  --train-acts.shard-root $TRAIN_SHARDS \
  --train-acts.layer -1 \
  --val-acts.shard-root $VAL_SHARDS \
  --val-acts.layer -1 \
  --imgs.root /nfs/$USER/datasets/ade20k/ \
  --sweep contrib/semseg/sweep.toml
```

# Establish Linear Probe Baseline Metrics

```sh
uv run python -m contrib.semseg validate \
  --imgs.root /nfs/$USER/datasets/ade20k/ \
  --acts.shard-root $VAL_SHARDS \
  --acts.layer -1
```

Then you can look in `./logs/contrib/semseg` for `hparam-sweeps.png` to see what learning rate/weight decay combination is best.
