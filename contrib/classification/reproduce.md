# Reproduce

You can reproduce our classification control experiments from our preprint by following these instructions.

The big overview (as described in our paper) is:

1. Train an SAE on the ImageNet-1K [CLS] token activations from a CLIP ViT-B/16, from the 11th (second-to-last) layer.
2. Show that you get meaningful features, through visualizations.
3. Train a linear probe on the [CLS] token activations from  a CLIP ViT-B/16, from the 11th layer, on the Oxford Flowers-102 dataset.
4. Show that we get good accuracy.
5. Manipulate the activations using the proposed SAE features.
6. Be amazed. :)

To do these steps:

## Record ImageNet-1K activations

## Train an SAE

```sh
uv run python -m saev train --sweep configs/preprint/classification.toml --data.shard-root /local/scratch/stevens.994/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8/ --data.patches cls --data.layer -2 --sae.d-vit 768
```

## Visualize the SAE Features

`bd97z80b` was the best checkpoint from my sweep.

```sh
uv run python -m saev visuals \
  --ckpt checkpoints/bd97z80b/sae.pt \
  --dump-to /research/nfs_su_809/workspace/stevens.994/saev/features/bd97z80b \
  --sort-byt cls \
  --data.shard-root /local/scratch/stevens.994/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8/ \
  --data.layer -2 \
  --data.patches cls \
  --log-freq-range -2.5 -1.5 \
  --log-value-range 0.0 1.0 \
  images:imagenet-dataset
```

You can see some neat features in here by using `saev.interactive.features` with `marimo`.

## Record Oxford Flowers-102 Activations

For each `$SPLIT` in "train", "valid" and "test":

```sh
uv run python -m saev activations \
  --model-family clip \
  --model-ckpt ViT-B-16/openai \
  --d-vit 768 \
  --n-patches-per-img 196 \
  --layers -2 \
  --dump-to /local/scratch/$USER/cache/saev \
  --n-patches-per-shard 2_4000_000 \
  data:image-folder-dataset \
  --data.root /nfs/datasets/flowers102/$SPLIT
```

## Train a Linear Probe

```sh
uv run python -m contrib.classification train \
  --n-workers 32 \
  --train-acts.shard-root /local/scratch/$USER/cache/saev/$TRAIN \
  --val-acts.shard-root /local/scratch/$USER/cache/saev/$VAL \
  --train-imgs.root /nfs/$USER/datasets/flowers102/train \
  --val-imgs.root /nfs/$USER/datasets/flowers102/val \
  --sweep contrib/classification/sweep.toml
```

Then look at `logs/contrib/classification/hparam-sweeps.png`. It probably works for any of the learning rates above 1e-5 or so.

## Manipulate

Now we will manipulate the inputs to the probe by using the directions proposed by the SAE trained on ImageNet-1K and observe the changes in the linear model's predictions.
There are two ways to do this:

1. The marimo dashboard, which requires that you run your own inference.
2. The online dashboard, which is more polished but offers less control.

Since you have gone through the effort of training all this stuff, you probably want more control and have the hardware for inference.

Run the marimo dashboard with:

```sh
uv run marimo edit contrib/classification/interactive.py
```
