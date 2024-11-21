# Guide to Training SAEs on Vision Models

1. Record ViT activations and save them to disk.
2. Train SAEs on the activations.
3. Visualize the learned features from the trained SAEs.
4. (your job) Propose trends and patterns in the visualized features.
5. (your job, supported by code) Construct datasets to test your hypothesized trends.
6. Confirm/reject hypotheses using `probing` package.

`saev` helps with steps 1, 2 and 3.

.. note:: `saev` assumes you are running on NVIDIA GPUs. On a multi-GPU system, prefix your commands with `CUDA_VISIBLE_DEVICES=X` to run on GPU X.

## Record ViT Activations to Disk

To save activations to disk, we need to specify:

1. Which model we would like to use
2. Which layers we would like to save.
3. Where on disk and how we would like to save activations.
4. Which images we want to save activations for.

The `saev.activations` module does all of this for us.

Run `uv run python -m saev activations --help` to see all the configuration.

In practice, you might run:

```sh
uv run python -m saev activations \
  --model-group clip \
  --model-ckpt ViT-B-32/openai \
  --d-vit 768 \
  --n-patches-per-img 49 \
  --layers -2 \
  --dump-to /local/scratch/$USER/cache/saev \
  --n-patches-per-shard 2_4000_000 \
  data:imagenet-dataset
```

This will save activations for the CLIP-pretrained model ViT-B/32, which has a residual stream dimension of 768, and has 49 patches per image (224 / 32 = 7; 7 x 7 = 49).
It will save the second-to-last layer (`--layer -2`).
It will write 2.4M patches per shard, and save shards to a new directory `/local/scratch$USER/cache/saev`.


.. note:: A note on storage space: A ViT-B/16 will save 1.2M images x 197 patches/layer/image x 1 layer = ~240M activations, each of which take up 768 floats x 4 bytes/float = 3072 bytes, for a **total of 723GB** for the entire dataset. As you scale to larger models (ViT-L has 1024 dimensions, 14x14 patches are 224 patches/layer/image), recorded activations will grow even larger.

This script will also save a `metadata.json` file that will record the relevant metadata for these activations, which will be read by future steps.
The activations will be in `.bin` files, numbered starting from 000000.


