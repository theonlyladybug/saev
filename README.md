# saev - Sparse Auto-Encoders for Vision

Implementation of sparse autoencoders (SAEs) for vision transformers (ViTs) in PyTorch.

## About

saev is a package for training sparse autoencoders (SAEs) on vision transformers (ViTs) in PyTorch.
It also includes an interactive webapp for looking through a trained SAE's features.

Originally forked from [HugoFry](https://github.com/HugoFry/mats_sae_training_for_ViTs) who forked it from [Joseph Bloom](https://github.com/jbloomAus/SAELens).

Read [logbook.md](logbook.md) for a detailed log of my thought process.

See [related-work.md](related-work.md) for a list of works training SAEs on vision models.
Please open an issue or a PR if there is missing work.

## Installation

Installation is supported with [uv](https://docs.astral.sh/uv/).
saev will likely work with pure pip, conda, etc. but I will not formally support it.

To install, clone this repository (maybe fork it first if you want).

In the project root directory, run `uv run python -m saev --help`.
The first invocation should create a virtual environment and show a help message.

## Using `saev`

See the [docs](https://samuelstevens.me/saev/) for an overview.


## Roadmap

1. Train models with data scaling (norm, mean) turned on.
2. Train models on ViT-L/14 datasets.
3. Semantic segmentation baseline with linear probe.
4. ADE20K experiment to demonstrate faithfulness.
