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

1. Record 1B activations for DINOv2, OpenCLIP, and a pretrained ViT-L/14 that's hopefully trained on classification from reLAION datasets. This will let us first compare the effect of pretraining objective on SAE reconstruction, which I think is the most interesting.
2. Train a bunch of SAEs on these models. Explore different activations, L1 coefficients, etc.
3. Measure ImageNet performance using linear classification on reconstructed ImageNet activations.

What graphs and figures would I put in this paper?

* I want qualitative comparisons of features discovered across different objectives. One specific question is: do vision-only pretrained model learn abstract concepts like "love" or "taking care of their young"? I can also try to answer this question with specific, hand-crafted experiments.
* I want to see if particular vision architectures have significantly different MSE-L0 tradeoffs.
* Do different pre-training corpuses lead to meaningfully different features? Probably?

In some regard, I like the idea of a hand-crafted task with a linear probe to measure whether an SAE is learning a particular feature that can accurately predict a task.
Is there a neuron that reliably distinguishes between pictures of love and other pictures?
Such tasks would be naturally biased and arbitrary, but I think we can try to construct evals that lead to meaningful differences between models and their SAEs.
What are some examples of such tasks?

1. Love (abstract concept in ImageNet)
2. Groups of animals (abstract concept in ToL-10M)
3. Geographic areas/cultures - We know CLIP can do this already, but can DINOv2?
4. Taxonomic trees: is this a mammal, is this a bird, etc. Probably to a finer degree (is this a feline, is this a hawk, etc).

But again, if you can reliably make linear ImageNet performance go down using the SAE reconstructions, *probably* the SAE is of higher quality.
I would be very surprised if this is not the case.

This opens up questions like:

* At what layer do you get best ImageNet performance?
* Is this the same layer when using the model activations directly?
* Do you do better when training an SAE on all patches and all layers, or better to train patch-specific, layer-specific SAEs (similar to GemmaScope).

I forgot that I probably want to record activations for every layer and every patch.
For ViT-L-14 that's 24 layers, which means one ViT will fill up my storage completely.

# 10/31/2024 (Happy Halloween)

For ImageNet-1K, with 1.2M images, how big is a set of ViT-L/14 activations?
1.2M images x 24 layers x 257 patches x 1024 floats x 4 bytes/float = 30TB.
Still too big to put on my lab servers.
But with only 4 layers?
Then it's 5TB.
Still too big.
But I can debug these processes on the lab servers.

With a ViT-B/32, saving the last 3 layers, ImagetNet-1K is

1.2M x 3 x 50 x 768 x 4 bytes/float = 553GB

It seems that training is working well.
I can train on 100M patches in about 40m on an A6000, which is good because it's 10x more tokens than 10M and about 10x slower (40m vs 4m).

Now I need to make sure training still works when only using the [CLS] token.
Then I need to work on evaluation to get the entire pipeline works.

How do we evaluate?

1. Log sparsity histograms
2. Delta ImageNet-1K linear classification accuracy
3. Large-scale linear probing tasks using existing datasets
4. Hand-crafted linear probing tasks

# 11/01/2024

Notes on evaluation

[Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2-Small](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream#General_Advice_for_Training_SAEs)

> Since we don’t have good metrics for interpretability / reconstruction quality, it’s hard to know when we are actually optimizing what we care about. 

> On top of this, we’re trying to pick a good point on the pareto frontier between interpretability and reconstruction quality which is a hard thing to assess well.

> The main objective is to have your Sparse Autoencoder learn a population of sparse features (which are likely to be interpretable) without having some **dense features** (features which activate all the time and are likely uninterpretable) or too many **dead features** (features which never fire).

> Too Dense:  dense features will occur at a frequency > 1 / 100. Some dense-ish features are likely fine (such as a feature representing that a token begins with a space) but too many is likely an issue.

> Too Sparse: Dead features won’t be sampled so will turn up at log10(epsilon), for epsilon added to avoid logging 0 numbers. Too many of these mean you’re over penalizing with L1. 

This is only in the validation set. 
It's unlikely to have dead features in your entire training set (I think).

> Just-Right: Without too many dead or dense features, we see a distribution that has most mass between -5 or -4 and -3 log10 feature sparsity. The exact range can vary depending on the model / SAE size but the dense or dead features tend to stick out. 

This post suggests that the historgrams are one of the best ways to quickly determine whether an SAE is good or not.
We can automate this to some degree by counting the number of dead features and the number of dense features. 
A better SAE will have fewer dead features and fewer dense features than a worse SAE.
So that's a very easy and quick way to evaluate models.

# 11/02/2024

How exactly do we evaluate models?
Let's qualitatively compare several single-number metrics and see how well they correlate with the more in-depth analyses.

Single-number metrics:

* MSE + lambda * L1 (original training loss, but fix lambda).
* Number of dead fautres + lambda * number of dense features
* MSE + lambda L0

Complex metrics:

* Normalized ImageNet-1K linear classification accuracy
* Large-scale probe tasks

Qualititative metrics:

* Looking at the top-k image grids
* Hand-crafted probe tasks

How well do the probe tasks work?

I also think I don't have to re-train the linear classifiers.
I can just apply them directly to the SAE.

# 11/04/2024

With more than one worker, my iterable HF dataset is being shuffled.
This is very bad for me.
I need to make sure this doesn't happen, then (sadly) recalculate the image activations.

# 11/05/2024

My "clean" dataset using a ViT-B/32 has empty values at 62721167 and onwards.
The length of the dataset is 62777183.

The config is `DataLoad(shard_root='/local/scratch/stevens.994/cache/saev/4dc22752a94c350ea6045599290cfbc31e3ee96b213d485318e434362b3bbdda/', patches='patches', layer=-2)`
`dataset[62721166]` and `dataset[62721165]` are non-empty, but `dataset[62721167]` onwards is empty, which causes issues.

Time to figure out why.
It was an indexing issue.
It's fixed now.
I fixed this bug.

What's the priority?

1. Qualitatively an SAE trained on the CLS token using DinoV2 and a CLIP model.
2. Apply some specific hand-crafted evaluations to measure "abstract" features in DINOv2 space.

I want to uncover interesting patterns in DINOv2 features.
Does it also have superposition?
Or is it only due to language?

Also, how do we evaluate patch-level features qualitatively?
Should we look at images that have maximal patches?
Or is a mean a better use?
Can I highlight individual patches?


Now I want to pick out images that have patches that maximize individual SAE features.
So I need to look at every patch, then refer back to the original image index and patch index, then save these values.

Tomorrow:

Patch-level activations don't seem to be working. 
But this could be an outcome of using CLIP rather than DINO. 
So I should verify that CLS token visualizations still work with CLIP and pre-trained SAEs.
Then I should train a patch-level SAE with DINO activations on OSC.
Finally, I can try getting patch-level activations with DINO.

Why do the same images show up? 
Are the img indices actually changing from the first batch, or is it always 0-333?


# 11/06/2024

1. Why do the same images show up in different features? Are indices changing from loop to loop, or always [0, 334)?
2. Set the script up so it works with CLS tokens.
3. Verify that it works with older checkpoints.
4. Train both patch-level and CLS-level DINOv2 SAEs on OSC.
5. Change webapp.py script to support:
  1. Max activation over entire image (current)
  2. Max CLS activation (old)
  3. Max patch, then only show unique images out of top k (new)


# 11/07/2024

Potential outline

1. Qualitatively compare DINOv2 and CLIP using top-k grids. Demonstrate the different concepts that are learned using cherry picked examples.
2. Discuss methods to evaluate SAEs, cheap proxies for these evaluations, and which proxies are good approximations of more expensive evaluation methods.
3. Discuss training strategies that reliably improve these proxy metrics (hyperparams, different datasets, etc).
4. Use this analysis to compare DINOv2 vs CLIP, CLIP vs BioCLIP, training SAEs on ImageNet-1K, iNat21, ToL-10M and LAION-2B multilingual.

Time to train on the [CLS] token and the patch tokens for DIONv2.

# 11/08/2024

OSC is busy right now.
What experiments can I do on the lab servers?

1. Work on quantitative evaluations.
2. Write.
3. Work on hand-crafted tasks
4. Explore large-patch models like ViT-S/32

# 11/10/2024

Thinking about a preprint.

**What is the goal and why do people care?**

The goal is to produce a reliable training recipe for SAEs for ViTs.
People care because SAEs are a powerful tool for interpretability that are underused in vision.
Training SAEs is challenging, but there exists a simple recipe so that everyone can get started in this area.

**Who is your audience? Who will use this or build on this?**

Computer vision interpretability researchers.

**If you want it yourself to solve some problem, then that is a good sign. Explain why you want it.**

I want to find traits in images of living organisms.
I want to build a search engine using these patch-level traits.

**What is your hypothesis? How will you know if it's true?**

My hypothesis is that simple, cheap tricks like normalizing input vectors, using L1 magnitude in the sparsity term of the loss, etc will remove the need for more complex stuff like forcing W_dec to have unit norms, removing parallel gradients, ghost grads, etc.
I'll know if it's true if adding ghost grads to my recipe R is not meaningfully better than just my recipe R.
Same for removing parallel gradients, etc.

A second hypothesis is that DINOv2 learns fine-grained visual patterns, while CLIP learns larger, semantic concepts.
I hypothesize that DINOv2 will be better for recovering morphological traits within species, and CLIP will be better across more diverse clades.
I will know if this is true by comparing scores on Broden.

**What is the problem/impediment that means it has not been done yet?**

No one is interested.
Existing interpretability methods like GradCAM and such seem to be "good enough".
There's not really an impediment to applying SAEs to ViTs, there just isn't as drastic a need.

**Nugget — what is your key insight that makes it doable? This is probably the single most important thing in a good paper.**

The nugget is that sparse autoencoders uncover features that are not easily recoverable by gradient based interpretability or clustering methods.

> This isn't a good nugget. Keep thinking about it.

**What is your elevator pitch? That is, if you meet a senior scientist in an elevator and she asks you what your paper is about, how do you describe it in 3 sentences or fewer.**

We train sparse autoencoders on vision transformers and find new patterns in the features learned through different pre-training objectivs and data distributions.

**Teaser — what would a teaser image show to explain this core idea?**



**What are the key previous works and what do they get wrong? All papers have limitations. Find these and they point the way forward.**

**How will you evaluate your method quantitatively?**

**What is your “demo”? That is, how do you show that the idea works.**

**What are the key risks?**

**Do you have all the data you need?**


---

The answers to some of these questions imply that I need to compare to GradCAM and standard interpretability methods.
First, I need to convince computer vision interpretability folks that our method has merit compared to other methods.
Second, I need to show that GradCAM specifically isn't good enough for the types of things I want to do.
On the other hand, that sounds boring and slow.
Why do I care about SAEs over GradCAM?
Because I want to find other images that have similar features at a patch level.
Can I do that with the raw activations?
Like just a clustering method?


I should expand on this:

> The goal is to produce a reliable training recipe for SAEs for ViTs.
> People care because SAEs are a powerful tool for interpretability that are underused in vision.
> Training SAEs is challenging, but there exists a simple recipe so that everyone can get started in this area.

How do you know they're a powerful tool?
What evidence do you have?

I can use SAEs to produce interpretable classification trees.

Why is training SAEs challenging?
What's wrong with the simplest possible recipe?

I know that training SAEs for vision is challenging because the default recipe doesn't work for CLIP for ImageNet-1K.
But my recipe does.

How do you know the default recipe "doesn't work"?

* No interpretable features -> many features are dense/dead, not many features corespond with Broden concepts
* 


**What do I want?**

I want a simple, reliable recipe for training sparse autoencoders on vision transformer patch activations for any pre-trained vision transformer and for any dataset of activations.

**Why do I want that?**

With such a recipe, I can train a sparse autoencoder for any ViT on any dataset.
With sparse autoencoders, I can

1. Train interpretable classifiers (decision trees, concept bottleneck models).
2. Discover differences in the learned features of pre-trained vision models.
3. Find morphological traits in living organism datasets that are consistent across species.

**How do I know I have a reliable recipe?**

If I have a recipe that works on DINOv2, CLIP, and BioCLIP across ImageNet-1K and iNat21 (mini), where

* Larger SAEs outperform smaller SAEs on the reconstruction-sparsity tradeoff
* larger SAEs outperform smaller SAEs on our downstream measures

Then I am confident that I have a reliable recipe.

**Why doesn't it exist?**

I don't know.
It doesn't seem that hard.

What about from a perspective farther out?


# 11/11/2024

Wider models aren't doing better on the MSE-LO tradeoff.

From the GemmaScope paper:

> Holding all else equal, wide SAEs learn more latent directions and provide better reconstruction fidelity at a given level of sparsity than narrow SAEs.

OpenAI's scaling paper also shows this.

So there's probably something wrong with my setup.
Let's start with the dumb baseline again and verify that each additional step of complexity leads to an improved model.

Performance:

I can take the idea from [this post](https://community.wandb.ai/t/is-it-possible-to-log-to-multiple-runs-simultaneously/4387/2) and train many models in parallel so that I can amortize the cost of loading a batch over many training runs.
I did this.

## Checkpoint Notes

| WandB ID | Model | Dataset | Remove Parallel Grads | Normalize W_dec | Re-Init b_dec | Shard Root |
|---|---|---|---|---|---|---|
| h52suuax | CLIP | iNat21 | True | True | True | 029ca197e |
| 9d8r1qhs | CLIP | iNat21 | False | False | False | 029ca197e |
| cesfj6kj | DINOv2 | iNat21 | False | False | False | b8e0fc701 |
| c57ddw4o | CLIP | ImageNet-1K | False | False | False | f46d3e7a3 |
| 1lm0m9d2 | CLIP | ImageNet-1K | True | True | True | f46d3e7a3 |
| lfesqa63 | DINOv2 | ImageNet-1K | False | False | False | 3c824621e |

For iNat21
```sh
CKPT=cesfj6kj CKPT_DATA=b8e0fc701b95ffe84a99d1beeb57a16c2d8aa462c888e114f8e22658bf601fb4 CUDA_VISIBLE_DEVICES=1 \
  uv run main.py webapp \
  --ckpt checkpoints/$CKPT/sae.pt \
  --dump-to /research/nfs_su_809/workspace/stevens.994/saev/webapp/$CKPT \
  --data.shard-root /local/scratch/stevens.994/cache/saev/$CKPT_DATA \
  --data.patches patches --data.layer -2 --top-k 64 --n-workers 32 --sort-by patch images:inat21-dataset --images.root /research/nfs_su_809/workspace/stevens.994/datasets/inat21/ --images.split train_mini
```

For ImageNet-1K
```sh
CKPT=1lm0m9d2 CKPT_DATA=f46d3e7a3223a50f1423ad03305cda971fac4512f03264392dd719c8f2381cab CUDA_VISIBLE_DEVICES=3 \
  uv run main.py webapp \
  --ckpt checkpoints/$CKPT/sae.pt \
  --dump-to /research/nfs_su_809/workspace/stevens.994/saev/webapp/$CKPT \
  --data.shard-root /local/scratch/stevens.994/cache/saev/$CKPT_DATA \
  --data.patches patches --data.layer -2 --top-k 64 --n-workers 32 --sort-by patch images:imagenet-dataset
```

## Notes on Checkpoint `cesfj6kj`

DINOv2, iNat21 train_mini split.

* 235 is bovine horns. 3300 is also bovine horns but a different kind; larger animals like bison and cows.
* 402 is bird heads
* 561 is bird wings
* 103, 575 712 are flower anthers
* 741 is ungulate noses
* 850 is bird wing *tips*
* 1196 has porcupine spines and antelope horns, but 1203 is sea urchin spines.
* 1638, 2709 have timestamps (artifacts of data)
* 1665 is reptile and amphibian noses
* 1823 is fish caudal (tail) fin
* 1900 is flower petals
* 2010, 2635 are bird tail
* 2094 is owl heads
* 2193 is insect legs
* 2247 is spiral shells
* 2358 is bird feeders
* 2537, 3568 are split images
* 2606 is protective shells, like porcupines, echidnas and armadillos.
* 2635 is bird tails
* 2746 is mushrool stalks
* 2770 is legs but only for Aphonopelma (tarantula genus)
* 2948 is callouts in the images
* 3007, 3691 are phoenix (palm tree genus)
* 3009 is Trochilidae (hummingbird family)
* 3093 is birds with black markings on the eyes <- TRAIT?
* 3218 is bird wings with white stripes on wing
* 3226 is rings on a finger (human jewelery)
* 3332 is cacti spines
* 3410 is Echinacea purpurea (but both images are labeled as animals because they contain insects).
* 3476 is human hair
* 3652 is coins (used for scale)
* 3664 is a bird wing pattern (not sure what it's called, but very distinctive).
* 3738 is a long bird beak. All sandpipers and pelicans.
* 3786 seems to be a thin white stripe at the front edge of bird wings.
* 3897 is a moth body
* 3988 is Ramphastos (toucan genus) bills (there are definitely other neurons for this feature)
* 4748 might be legs of ungulates.
* 6995 looks like dorsal fins of fish.
* 10863 seems to activate on fur.
* 10992 might be fish pectoral fins



## Notes on Checkpoint `h52suuax`

CLIP, iNat21 train_mini split, *with CLIP tricks

* 647 is mammal feet; covers rodents, carnivores, marsupials, *and* hyraxes
* 674 is stripes across zebras, fish, *and* butterflies.
* 1128 is camoflage across spiders, insects and frogs.
* 1246 is only on flowering plants --> is this reliable??
* 1282 is spines across sea urchins and porcupines

* 6826 is the spines of Erinaceus roumanicus (specific hedgehog species)
* 6819 is the spikes of Asphodelaceae family, which has the following description on Wikipedia: "The flowers (the inflorescence) are typically borne on a leafless stalk (scape) which arises from a basal rosette of leaves."
* 6371 is both dorsal and anal fins
* 6293 is birds with white markings on their head

## Notes on Checkpoint `lfesqa63`

DINO, ImageNet-1K train without tricks

* 128 is round things (donuts, tubas, trumpets)
* 218 and 1931 are fluffly pink things
* 435 is the round trumpet bell, but from the side.
* 461 is the crossed legs of folding tables, folding chairs, folding ironing boards, etc.
* 626 is window screens, which have a very distinct texture.
* 687 is flower bouqets, typically held, but I expect that to be a result of ImageNet, not the SAE.
* 819 is tank treads from a 45 degree angle, but 83 is tank treads straight from the side. **I think DINOv2 doesn't learn object rotation invariances because of a lack of labels.** 3600 is also tank treads, from a different angle again.
* 882 is animal hind (all from the side view).
* 932 is a flower pattern
* 951 is open animal mouths
* 1075 is flags (all american--ImageNet or SAE?)
* 1172 is metal wings (often from "pickelhaube")
* 1235 is the specific "knee" joint of birds with long legs
* 1328 is the Coca-Cola logo
* 1368 is ram/bighorn sheep curved horns.
* 1572 is "cursive"
* 1888 is kingsnakes; 2417 is kingsnake and other snakes
* 10143 is human ears
* 10312 is human shoulders (hoodie, lab coat, black and white)
* 10778 is masks (ski mask, knight helmet, burka)
* 10858 is dog litters. But is this all young, or specifically dog littles because they look the same?
* 2264 is animal ears (cats, fox, lynx)
* 2389 is the dung ball
* 2504 is echidna spines
* 2791 is the spiky bit below the petals in a cardoon flower.
* 2940 is the hood of a cobra (3892)
* 3090 is the legs of a tarantula
* 3428 is antenna of leaf beetles
* 3604 is generic text on shiny packaging
* 3901 is specifically the *top* of the treads
* 3919 is generic characters -> no OCR!
* 3956 is whisker (cat otter, lync, etc)
* 10966 is sneakers
* 11170 is bird feet
* 11183 is gauges
* 11248 is cat whiskers (tigers, cats, leopards)
* 11264 is onions on burgers.
* 11492 is rice (specifically with other food? Visual or semantic?)
* 11565 is just generic "black text on white background", but not reading the symbols
* 11650 is jack o lanterns. Compare to CLIP's "october" feature
* 11744 is striped rodent ears (possom, badger, lemur)
* 11851 is wing tips in birds of prey (again)

## Notes on Checkpoint `1lm0m9d2`

CLIP, ImageNet-1K train, *with* SAE tricks

* 113 is "cloud" (text and symbol)
* 188 is god and bible
* 299 is dog tongues
* 315 is ungulate legs/hooves
* 350 is bald heads
* 376 is "October" with just text. Does it work for Halloween stuff too?
* 407 is "Support" text.
* 425 might an example of needing a register.
* 537 is grated cheese on top of food
* 594 is photography (OCR and concepts)
* 599 is teeth (drawing and photographs)
* 613 is snow dog tails
* 768 is images that are split (multiple shots in one image)
* 889 is linoleum tile (flooring)
* 927 is Russia (OCR, symbols, concepts like U-boat, locomotive)
* 949 is "royal"

What patterns am I seeing now?

* CLIP groups text, symbols, drawings and images together in the semantic space. This is really useful for things like LMMs.
* DINOv2 identifies traits more often than CLIP when looking at iNat21. Like, 5x as often




I'm writing a submission to CVPR. The premise is that we apply sparse autoencoders to vision models to interpret their internal representations.

My current outline is

1. Introduction: we want to interpret foundation vision models and see examples of the concepts being represented. SAEs are the best way to do this. We apply SAEs to DINOv2 and CLIP vision models and find some neat stuff.
2. Related work.
3. How we train the SAEs (technical details, easy to write)
4. Findings
    1. CLIP learns abstract semantic relationships, DINOv2 doesn't.
    2. DINOv2 identifies morphological traits in animals much more often than CLIP.
    3. Training an SAE on one datset transfers to a new dataset.
5. Conclusion & Future Work

# 11/14/2024

Anthropic discusses ways to search for particular features [here](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#searching).
In general, it seems using combinations of positive/negative examples and filtering by "fire/no fire" is good, rather than the LDA-based classifier I have now.
