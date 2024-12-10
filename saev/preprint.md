# Notes for Preprint

I'm writing a submission to ICML. The premise is that we apply sparse autoencoders to vision models like DINOv2 and CLIP to interpret and control their internal representations. 

## Outline

1. Introduction
    1.1. Understanding requires intervention - we must test hypotheses through controlled experiments (scientific method)
    1.2. Current methods provide only understanding or only control, never both
    1.3 Understanding and controlling vision models requires three key capabilities: the ability to identify human-interpretable features (like 'fur' or 'wheels'), reliable ways to manipulate these features, and compatibility with existing models. 
    1.4 Current methods fail to meet all three requirements - they either discover features that can't be manipulated, enable manipulations that aren't interpretable, or require expensive model retraining.
    1.5. SAEs from NLP provide unified solution: interpretable features that can be precisely controlled
    1.6. Contributions: SAE for vision model, new understanding of differences in vision models, multiple control examples across tasks

2. Background & Related Work
    2.1. Foundation model interpretability
    2.2. Model editing 
    2.3. Sparse representations
    2.4. SAEs in language models

3. Method
    3.1. SAE architecture and training
    3.2. Feature intervention framework
        3.2.1. Train a (or use an existing pre-trained) task-specific head: make predictions based on [CLS] activations, use an LLM to generate captions based on an image and a prompt, etc.
        3.2.2. Manipulate the vision transformer activations using the SAE-proposed features and compare outputs before/after intervention.
    3.4. Evaluation metrics

4. Understanding Results
    4.1. Pre-Training Modality Affects Learned Features - DINOv2 vs CLIP
    4.2. Pre-Training Distritbuion Affects Learned Features - CLIP vs BioCLIP

5. Control Results
    5.1. Image Classification Control
    5.2. Semantic Segmentation Control
        * Intro explaining? Unknown
        * Technical description of training linear semseg head on DINOv2 features.
        * Description of how we automatically find ADE20K class features in SAE latent space
        * Qualitative results (cherry picked examples, full-width figure)
        * Quantitative results (single-column table)
    5.3. Image Generation Control
    5.4. Vision-Language Control

6. Discussion
    6.1. Limitations
    6.2. Societal implications
    6.3. Future work

I also want to build some interactive dashboards and tools to demonstrate that this stuff works.

1. I want my current PCA dashboard with UMAP instead
2. Given a linear classifier of semantic segmentation features, I want to manipulate the features in a given patch, and apply the suppression to all patches to see the live changes on the segmentation mask.
3. After training an SAE on a CLS token, I can then train a linear classifier on the CLS token with ImageNet-1K, and manipulate the features directly.
4. Given a small vision-language model like Phi-3.5 or Moondream, I want to manipulate the vision embeddings (suppressing or adding one or more features) and then see how the top 5 responses change in response to the user input (non-zero temperature).
5. Given a zero-shot CLIP or SigLIP classifier, you can add subtract features from all patches, then see how the classification changes

---

With respect to writing, we want to frame everything as Goal->Problem->Solution. In general, I want you to be skeptical and challenging of arguments that are not supported by evidence.

Some questions that come up that are not in the outline yet:

Q: Am you using standard SAEs or have you adopted the architecture?

A: I am using ReLU SAEs with an L1 sparsity term and I have constrained the columns of W_dec to be unit norm to prevent shrinkage. We are not using sigmoid or tanh activations because of prior work from Anthropic exploring the use of these activation functions, finding them to produce worse features than ReLU.

Q: What datasets are you using?

A: I am using ImageNet-1K for training and testing. I am extending it to iNat2021 (train-mini, 500K images) to demonstrate that results hold beyond ImageNet. 

We're going to work together on writing this paper, so I want to give you an opportunity to ask any questions you might have.

It can be helpful to think about this project from the perspective of a top machine learning researcher, like Lucas Beyer, Yann LeCun, or Francois Chollet. What would they think about this project? What criticisms would they have? What parts would be novel or exciting to them?



