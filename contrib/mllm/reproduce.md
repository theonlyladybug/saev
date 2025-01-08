# Reproduce

You can reproduce our text generation experiments from our preprint by following these instructions.

The big overview is:

1. Train an SAE on the ImageNet-1K patch activations from a MLLM's vision encoder, from the second-to-last layer.
2. Show that you get meaningful features, through visualizations.
3. Manipulate the activations using the proposed SAE features.
4. Generate text after manipulating the vision features.

To do these steps:

## Record ImageNet-1K activations

```sh
uv run python -m saev activations \
  --model-family moondream2 \
  --model-ckpt vikhyatk/moondream2:2024-08-26 \
  --no-cls-token \
  --n-patches-per-img 729 \
  --d-vit 1152 \
  --layers -2 \
  --dump-to /local/scratch/$USER/cache/saev/ \
  --vit-batch-size 512 \
  data:imagenet-dataset
```

## Train an SAE on Activations

```sh
uv run python -m saev train \
  --sweep configs/preprint/baseline.toml \
  --data.shard-root /local/scratch/$USER/cache/saev/$SHARDS/ \
  --data.layer -2 \
  --sae.d-vit 768
```
