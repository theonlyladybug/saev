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

```sh
uv run python -m contrib.semseg train \
  --sweep contrib/semseg/sweep.toml \
  --imgs.root /$NFS/$USER/datasets/ade20k
```

# Establish Linear Probe Baseline Metrics

```sh
uv run python -m contrib.semseg validate \
  --imgs.root /$NFS/$USER/datasets/ade20k
```

Then you can look in `./logs/contrib/semseg` for `hparam-sweeps.png` to see what learning rate/weight decay combination is best.
