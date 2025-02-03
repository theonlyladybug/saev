# To Do

A list of everything I need to do for a release.

## Code

* [ ] Guh

## Preprint

* [ ] Experiments
* [ ] Writing

## Release

* [ ] Interactive demos
* [ ] Module docstrings
* [ ] Class docstrings
* [ ] Method docstrings
* [ ] Function docstrings

## Preprint - Experiments

* [x] DINOv2 vs CLIP
* [x] Image classification control
* [ ] Image segmentation control

## Experiments - Understanding DINOv2 vs CLIP

* [x] Compute ImageNet-1K train activations for DINOv2 ViT-B/14
* [x] Compute ImageNet-1K train activations for CLIP ViT-B/16
* [x] Train SAE on patch-level activations of ImageNet-1K train from DINOv2 ViT-B/14
* [x] Train SAE on patch-level activations of ImageNet-1K train from CLIP ViT-B/16
* [x] Visualize features for DINOv2
* [x] Visualize features for CLIP
* [x] Find something neat.

## Experiments - Image Classification Control

* [x] Train SAE on [CLS] activations of ImageNet-1K train from CLIP ViT-B/16
* [x] Compute Caltech-101 train activations for CLIP ViT-B/16
* [x] Compute Caltech-101 test activations for CLIP ViT-B/16
* [x] Train linear probe for Caltech-101 classification
* [x] Calculate 99th percentile of feature activation for each feature.
* [x] Develop interactive Marimo dashboard
* [x] Find something neat.
* [ ] Calculate logit relationship

## Experiments - Image Segmentation Control

* [x] Train SAE on patch-level activations of ImageNet-1K train from DINOv2 ViT-B/14
* [x] Compute ADE20K train activations for DINOv2 ViT-B/14
* [x] Compute ADE20K validation activations for DINOv2 ViT-B/14
* [x] Train linear probe for ADE20K semantic segmentation (`checkpoints/contrib/semseg/lr_0_001__wd_0_001/model_step8000.pt`)
* [x] What percentage of patches meet the 90% threshold?
* [ ] Develop interactive Marimo dashboard
* [ ] Find something neat.
* [ ] Quantitative results
