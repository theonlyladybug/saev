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

My strategy for calculating the mean activations only used 15 examples instead of 15 x 1024.
With 15 x 1024 examples, the b_dec is better initialized and it works exactly like before.
