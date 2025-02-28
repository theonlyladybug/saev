# Inference Instructions

Briefly, you need to:

1. Download a checkpoint.
2. Get the code.
3. Load the checkpoint.
4. Get activations.

Details are below.

## Download a Checkpoint

First, download an SAE checkpoint from the [Huggingface collection](https://huggingface.co/collections/osunlp/sae-v-67ab8c4fdf179d117db28195).

For instance, you can choose the SAE trained on OpenAI's CLIP ViT-B/16 with ImageNet-1K activations [here](https://huggingface.co/osunlp/SAE_CLIP_24K_ViT-B-16_IN1K).

You can use `wget` if you want:

```sh
wget https://huggingface.co/osunlp/SAE_CLIP_24K_ViT-B-16_IN1K/resolve/main/sae.pt
```

## Get the Code

The easiest way to do this is to clone the code:

```
git clone https://github.com/OSU-NLP-Group/SAE-V
```

You can also install the package from git if you use uv (not sure about pip or cuda):

```sh
uv add git+https://github.com/OSU-NLP-Group/SAE-V
```

Or clone it and install it as an editable with pip, lik `pip install -e .` in your virtual environment.

Then you can do things like `from saev import ...`.

.. note::
  If you struggle to get `saev` installed, open an issue on [GitHub](https://github.com/OSU-NLP-Group/SAE-V) and I will figure out how to make it easier.

## Load the Checkpoint

```py
import saev.nn

sae = saev.nn.load("PATH_TO_YOUR_SAE_CKPT.pt")
```

Now you have a pretrained SAE.

## Get Activations

[TODO: document this process]
