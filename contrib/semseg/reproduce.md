You can reproduce our semantic segmentation examples from our preprint by following these instructions.

1. Train a linear probe on semantic segmentation task using ADE20K.
2. Measure linear probe baseline metrics.
3. Manipulate the activations using the proposed SAE features.
4. Be amazed. :)

Details can be found below.

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

You need an SAE that's been trained on DINOv2's activations on ImageNet.
Then you can run both the frontend server and the backend server:

**Frontend:**

```sh
uv run python -m http.server
```

Then navigate to [http://localhost:8000/web/apps/semseg/](http://localhost:8000/web/apps/semseg/).

**Backend:**

This is a little trickier because the backend server lives on Huggingface spaces and talks to a personal Cloudflare server.

[TODO]
