# Logbook

This provides a set of notes, step-by-step, of my process developing this library.
In combination with code comments and git blame, it's probably the best way to understand *why* a decision was made, rather than *what* decision was made.

*Open science babbyyy!*

# 10/11/2024

Something in the post-training is not working.
I should run the post-training steps (analysis.py and generate_app_data.py) once for a known good checkpoint and once for the new checkpoint that used the dataloader and see if it's an issue with the post-training code or the checkpoint.
Honestly, I'm not sure which one I want to be broken.
Probbaly the post-training code?

# 10/17/2024

I trained a couple SAEs using both my updated codebase and the original codebase.
Like, the original codebase, without even ruff/isort added.
I did this because I figured there were four possible problems:

1. The updated training code is wrong.
2. The updated analysis.py file is wrong.
3. The updated get_app_data.py file is wrong.
4. The random seed was causing problems.

The key is that I now have several checkpoints, trained with the original codebase or with the udpated codebase.
I also have updated and original versions of the other scripts.
So I can take each piece through step by step and see where the mistake is.

| Checkpoint | Wandb | Training Code |
|---|---|---|
| 2dlebd60 | [samuelstevens/saev/wawwh1rj](https://wandb.ai/samuelstevens/saev/runs/wawwh1rj) | original |
| dd991rt3 | [samuelstevens/saev/g0sqbjux](https://wandb.ai/samuelstevens/saev/runs/g0sqbjux) | updated |


| Checkpoint | Updated Training? | Updated Analysis? | Updated App Data? | Worked? |
|---|---|---|--|---|
| 2dlebd60 | No | No | No | Yes |
| 2dlebd60 | No | No | Yes | No[^dataset-shuffle] |

[^dataset-shuffle]: This worked after I added the `.shuffle(seed=1)` line (commit [6319148](https://github.com/samuelstevens/saev/commit/6319148f269013721baf9da15a08e5b1ca0b6e32)) to the updated `generate_app_data.py` script. I think this is because in the original `vit_sae_analysis/dashboard_fns.py` there is a `shuffle(seed=seed)` and the default seed is 1.

So per this footnote[^dataset-shuffle] I think the `dataset.shuffle()` call in both `analysis.py` and `generate_app_data.py` needs to be the same.
I am going to update both the original code and the updated code to use `cfg.seed`.

# 10/19/2024

Same trial as before, but everywhere that we used `seed=1` is now `seed=cfg.seed`:

| Checkpoint | Updated Training? | Updated Analysis? | Updated App Data? | Worked? |
|---|---|---|--|---|
| 2dlebd60 | No | No | No | Yes |
| 2dlebd60 | No | No | Yes | Yes |

**Good!**
Now we can use the original analysis with the updated `generate_app_data.py` and it will work just fine.
So we can at least assume that `generate_app_data.py` is correct.
Now, let's figure out if there's something wrong with the `analysis.py` file.

| Checkpoint | Updated Training? | Updated Analysis? | Updated App Data? | Worked? |
|---|---|---|--|---|
| 2dlebd60 | No | Yes | No | No[^generate-bug] |
| 2dlebd60 | No | Yes | Yes | No |

[^generate-bug]: It threw an exception about some data not being available.

So here's the thing.
It's clearly my analysis that is wrong.
Even after removing the `dataset.shuffle` call in the updated `generate_app_data.py` file, the grids have some weird images.
Like it will be a mix of dogs and spiders in a single grid.
Why does that happen?
So I am going to do my best to eliminate as many differences as possible between the two scripts.

# 10/21/2024

I changed the activation store to behave *exactly* as the original code (shuffle before converting to an iterable dataset, commit [51f7947](https://github.com/samuelstevens/saev/commit/51f7947edcb19a7ca59dd3bd45d9869d090a91bf)).
Then the updated analysis and app data scripts work!
Now that I trust the updated analysis code, I can evaluate whether *my* pre-trained models are as good as the others.

| Checkpoint | Updated Training? | Updated Analysis? | Updated App Data? | Worked? |
|---|---|---|--|---|
| 2dlebd60 | No | Yes | Yes | **Yes** |
| dd991rt3 | Yes | Yes | Yes | **Yes** |

...and now we see that my updated training code works as well!
What a relief!

With this in mind, there are several minor changes I want to make before I do some BioCLIP and TreeOfLife runs:

1. Removing `transformer-lens` [done, commit [18612b7](https://github.com/samuelstevens/saev/commit/18612b75988c32ae8ab3db6656b44a442f3f7641)]
2. Removing HookedVisionTransformer [done, commit [c7ba7c7](https://github.com/samuelstevens/saev/commit/c7ba7c72c76472fd8cf2e7b2dc668d03a15b803d)]
3. OpenCLIP instead of huggingface `transformers` [done, commit [d362f64](https://github.com/samuelstevens/saev/commit/d362f64437b3599f56bb698136712d7590ee897b)]
4. Pre-computing ViT activations [done, commit [ee79f5b](https://github.com/samuelstevens/saev/commit/ee79f5b84186e655b2e5d485e972fe69bb73dd65)]

I'm going to do each of these independently using a set of runs as references.

# 10/22/2024

Removed HookedVisionTransformer (see above)
Checkpoint [v6jto37s](https://wandb.ai/samuelstevens/saev/runs/wwb20pa0) worked for training, analysis, and app data.

Testing an implementation using OpenCLIP instead of `transformers`.
Assuming it works (which seems likely given that the loss curve is identical), then I will pre-compute the activations, save them as a numpy array to disk, and memmap them during training rather than computing them.
I expect this to take a little bit because I had issues with shuffling and such in the analysis step earlier.
I think the best strategy is to work backwards.
The `generate_app_data.py` script doesn't need an activation store at all.
So I will start with the `analysis.py` script and add a new activations store class that meets the same interface as the original (maybe not for the constructor).
Then I will verify that the analysis script works correctly.

Only after that will I use the new class in training.
Working with the analysis script is a shorter feedback loop.

# 10/23/2024

OpenCLIP instead of transformers works (training, analysis, generate).
So now I am pre-computing activations.
I'm waiting on the activations to be saved (~3 hours).

CachedActivationsStore produced some duplicates in the analysis step.
Why is that?

For example, neuron 78 has the same image for image 6 and 7 (1-indexed, images 5 and 6 if zero-indexed).

Fixed it.
We no longer randomly sample batches; instead, we use a dataloader and `__getitem__`.

With training, however, the metrics no longer match the reference metrics.
Why is that?
We can find out by comparing to the original activations store.
Likely, we will need to build a custom data order using `np.random.default_rng(seed=cfg.seed)`.


# 10/24/2024

My strategy for calculating the mean activations only used 15 examples instead of 15 x 1024.
With 15 x 1024 examples, the b_dec is better initialized and it works exactly like before.

Now I have a complete training strategy.
The goal now is to train a SAE on BioCLIP's patch-level activations from TreeOfLife-10M.
How do we do this?

1. Train an SAE on ViT-L-14/openai image-level activations on *TreeOfLife-10M*.
2. Train an SAE on *BioCLIP* image-level activations on TreeOfLife-10M.
3. Train an SAE on BioCLIP *patch*-level activations on TreeOfLife-10M.

# 10/29/2024

I want to write a paper that does a bunch of introductory experiments for SAEs applied to vision.
Such a paper would be a good reference for new researchers hoping to train SAEs on vision models.
So I can make a set of experiments and slowly knock them off.
But I should think clearly about what specific graphs/figures I would want to make.
Then I can rank them in terms of difficulty, compute and reward (novelty, bits of information gained, etc) and work on the most important experiments first.

How do we know if an SAE is good?
This will enable us to say whether a recipe is good or bad.
So what does prior work do on automatically evaluating SAE quality?

# 10/30/2024

Here's a rough set of contributions + experiments:

We make the first comprehensive effort to train SAEs on vision models.

Contributions:

* Experimental findings.
* Code package to reproduce and extend.
* Trained SAE checkpoints + interactive explorer (web tool).

Experiments:

1. Activations: Compare [CLS] token activations vs patch-level activations.
2. Pretraining Task: Compare effects of supervised pre-training (classification), vision-only self-supervised (DINOv2, MAE) and vision-language (CLIP, SigLIP).
3. Pretraining Corpus: Compare various CLIP models (LAION, BioCLIP, PubmedCLIP, etc.)
4. Vision Architecture: Compare vision model architectures (ResNet, ViTs, ConvNeXt, etc.)
5. SAE Architecture: Compare SAE architectures and objectives (L1, TopK, Gated, JumpReLU, etc.)



The OpenAI paper (Scaling and evaluating sparse autoencoders) proposes four evaluation metrics:

1. Downstream loss: replace the original model representations with SAE reconstructions and calculate loss.
2. Probe loss: They train a 1D binary probe for specific, curated tasks. They pick features from SAEs that minimize loss on the task, and a low loss indicates that the SAE has learned the true feature used for the binary classification task.
3. Explanability:
4. Ablation sparsity:

Overall, I felt that these evaluation metrics are still not very good.
I think that we can use delta imagenet-1k linear probing accuracy as a measure of an SAE's performance, along with another dataset (iNat21? ToL-1M?) to verify that this approach is generalizable.
We also should include some qualitative study of the features to ensure that they are meaningful.
What will that qualitative study be?
Maybe a decision tree trained on SAE features to predict imagenet-1k classes?
We could use heuristics to pick out non-trivial SAE features; that is, we don't want features that are perfectly correlated with a class label.
Or we could use LAION as an example dataset.
ToL-10M was 37GB for one token.
LAION-400M would be 40x larger, so 1.4TB.
If we also stored patch-level activations, that would be 257x larger, or 360TB.
This is unfeasible.
But if we train on 1B tokens per autoencoder, that's 1B x 1024 floats x 4 bytes/float = ~4.1TB.
Which I think is not unreasonable.

In fact, it is very reasonable given our limits (100TB/user) on OSC.
Recall, however, that it's 4.1TB per vision backbone that I want to analyze.
So I can analyze about 23 backbones (realistically 20, given that we don't want to bump up against limits).

So what would I do?

1. Record 1B activations for DINOv2, OpenCLIP, and a pretrained ViT-L/14 that's hopefully trained on classification from reLAION datasets. This will let us first compare the 
2. Train a bunch of SAEs on these models. Explore different activations, L1 coefficients, etc.
3. Measure ImageNet performance using linear classification on reconstructed ImageNet activations.
