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

## Record Oxford Flowers-102 Activations

## Train a Linear Probe

## Manipulate
