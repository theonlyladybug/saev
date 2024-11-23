# Faithfulness

This module demonstrates that SAE features are faithful and that the underlying vision model does in fact depend on the features to make its predictions.

It demonstrates this through an interactive dashboard and through larger-scale quantitative experiments.

## Dashboard

First, record activations for the ADE20K dataset.

```sh
uv run python -m saev activations \
  --model-group clip \
  --model-ckpt ViT-B-16/openai \
  --d-vit 768 \
  --n-patches-per-img 196 \
  --layers -2 \
  --dump-to /local/scratch/$USER/cache/saev \
  --n-patches-per-shard 2_4000_000 \
  data:ade20k-dataset \
  --data.root /research/nfs_su_809/workspace/stevens.994/datasets/ade20k/images
```
