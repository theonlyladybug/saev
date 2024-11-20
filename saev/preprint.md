# Instructions to Reproduce Our Preprint Results

Environment variables used throughout this script:

* `ACTIVATION_DIR`: the root directory to dump/load ViT activations. I set it to `/local/scratch/$USER/cache/saev`.
* `CKPTS_DIR`: the root directory that to dump/load SAE checkpoints. I set it to `/nfs/$USER/saev/checkpoints`.

## Save Activations

Save ViT activations for ViT-L/14 CLIP, ViT-L/14 DINOv2 (with registers), SigLIP and V-JEPA.

```sh
uv run main.py activations \
  --dump-to $ACTIVATION_DIR \
  --model-org clip --model-ckpt ViT-L-14/openai \
  --d-vit 1024 --layers -2 --n-patches-per-img 256 \
  data:imagenet-dataset
```


```sh
uv run main.py activations \
  --dump-to /local/scratch/stevens.994/cache/saev/ \
  --model-org dinov2 --model-ckpt dinov2_vitl14_reg \
  --d-vit 1024 --layers -2 --n-patches-per-img 256 \
  data:imagenet-dataset
```

```sh
uv run main.py activations \
  --dump-to $ACTIVATION_DIR \
  --model-org siglip --model-ckpt ViT-SO400M-14-SigLIP/webli \
  --d-vit 1024 --layers -2 --n-patches-per-img 256 \
  data:imagenet-dataset
```

# Train SAEs

Then train patch-level SAEs on the activations.

`$HASH` is the hash for a particular set of activations. For example,  my DINOv2 ImageNet-1K activations are in `$ACTIVATION_DIR/44a593b1a6fbcf56811fdd076aa4173cff331ef3921453d3f5b40f3eb386cff7`

```sh
uv run main.py sweep \
  --sweep configs/preprint/baseline.toml \
  --n-patches 100_000_000 \
  --ckpt-path $CKPTS_DIR \
  --data.shard-root $ACTIVATIONS_DIR/$HASH \
  --data.patches patches \
  --data.layer -2
```

# Store Top-K Images



