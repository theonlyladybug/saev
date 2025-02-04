You can reproduce our semantic segmentation control experiments from our preprint by following these instructions.

As an overview:

1. Train a linear probe on semantic segmentation task using ADE20K.
2. Measure linear probe baseline metrics.
3. Manipulate the activations using the proposed SAE features.
4. Be amazed. :)

Details can be found below.

# Record ViT Activations for SAE Inference

For SAE inference, we want the second-to-last layer (as discussed).

# Train a Linear Probe on Semantic Segmentation

Train a linear probe on DINOv2 activations from ADE20K.
It's fixed with DINOv2 because of patch size, but the code could be extended to different ViTs.

```sh
uv run python -m contrib.semseg train \
  --sweep contrib/semseg/sweep.toml \
  --imgs.root /$NFS/$USER/datasets/ade20k
```

# Measure Linear Probe Baseline Metrics

Check which learning rate/weight decay combination is best for the linear probe.

```sh
uv run python -m contrib.semseg validate \
  --imgs.root /$NFS/$USER/datasets/ade20k
```

Then you can look in `./logs/contrib/semseg` for `hparam-sweeps.png` to see what learning rate/weight decay combination is best.

# Manipulate the Activations
