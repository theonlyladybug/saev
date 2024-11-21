# Guide to Training SAEs on Vision Models

1. Record ViT activations and save them to disk.
2. Train SAEs on the activations.
3. Visualize the learned features from the trained SAEs.
4. (your job) Propose trends and patterns in the visualized features.
5. (your job, supported by code) Construct datasets to test your hypothesized trends.
6. Confirm/reject hypotheses using `probing` package.

`saev` helps with steps 1, 2 and 3.

## Record ViT Activations to Disk
