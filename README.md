# saev - Sparse Auto-Encoders for Vision

Implementation of sparse autoencoders (SAEs) for vision transformers (ViTs) in PyTorch.

## About

saev is a package for training sparse autoencoders (SAEs) on vision transformers (ViTs) in PyTorch.
It also includes an interactive webapp for looking through a trained SAE's features.

Originally forked from [HugoFry](https://github.com/HugoFry/mats_sae_training_for_ViTs) who forked it from [Joseph Bloom](https://github.com/jbloomAus/SAELens).

Read [logbook.md](logbook.md) for a detailed log of my thought process.

See [related-work.md](related-work.md) for a list of works training SAEs on vision models.
Please open an issue or a PR if there is missing work.

## Using `saev`

Sweep LR for 10M patches on the second-to-last layer of the [CLS] token.

```sh
uv run main.py sweep \
  --sweep configs/baseline.toml \
  --n-patches 10_000_000 \
  --data.shard-root /local/scratch/stevens.994/cache/saev/4dc22752a94c350ea6045599290cfbc31e3ee96b213d485318e434362b3bbdda \
  --data.patches cls \
  --data.layer -2
```

Generate webapp images:

```sh
uv run main.py webapp \
  --ckpt ./checkpoints/cr6sl257/sae.pt \
  --data.shard-root /local/scratch/stevens.994/cache/saev/4dc22752a94c350ea6045599290cfbc31e3ee96b213d485318e434362b3bbdda \
  --dump-to /local/scratch/stevens.994/cache/saev/webapp/cr6sl257
```

Then run the webapp with:

```sh
uv run marimo edit webapp.py
```

And make sure `webapp_dir` is `"/local/scratch/stevens.994/cache/saev/webapp/cr6sl257"`.
