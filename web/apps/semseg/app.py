import os.path

import beartype
import gradio as gr
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

import saev.activations
import saev.config
import saev.nn
import saev.visuals

####################
# Helper Functions #
####################


@beartype.beartype
def load_tensor(path: str) -> Tensor:
    return torch.load(path, weights_only=True, map_location="cpu")


####################
# Global Variables #
####################


acts_dataset = saev.activations.Dataset(
    saev.config.DataLoad(
        shard_root="/local/scratch/stevens.994/cache/saev/1864947033ca8b8a171a482644a948a6e6489e3249469373c78dfeeb0a75bcd4",
        scale_mean=True,
        scale_norm=True,
    )
)

max_frequency = 1e-2

ckpt = "oebd6e6i"

sae_ckpt_fpath = f"/home/stevens.994/projects/saev/checkpoints/{ckpt}/sae.pt"
sae = saev.nn.load(sae_ckpt_fpath)
sae.eval()

ckpt_data_root = (
    f"/research/nfs_su_809/workspace/stevens.994/saev/features/{ckpt}/sort_by_patch"
)

top_img_i = load_tensor(os.path.join(ckpt_data_root, "top_img_i.pt"))
top_values = load_tensor(os.path.join(ckpt_data_root, "top_values.pt"))
sparsity = load_tensor(os.path.join(ckpt_data_root, "sparsity.pt"))


mask = torch.ones((sae.cfg.d_sae), dtype=bool)
mask = mask & (sparsity < max_frequency)
print(mask.float().sum())

in1k_dataset = saev.activations.get_dataset(
    saev.config.ImagenetDataset(),
    img_transform=v2.Compose([
        v2.Resize(size=(512, 512)),
        v2.CenterCrop(size=(448, 448)),
    ]),
)


@torch.inference_mode
def get_sae_examples(image_i: int, cell_i: int) -> list[Image.Image]:
    """
    Given a particular cell, returns some highlighted images showing what feature fires most on this cell.
    """
    i = image_i * acts_dataset.metadata.n_patches_per_img + cell_i
    example = acts_dataset[i]

    print(image_i, cell_i)

    assert example["image_i"] == image_i
    assert example["patch_i"] == cell_i

    vit_act = example["act"]

    _, f_x, _ = sae(vit_act[None, :])

    latents = torch.argsort(f_x, descending=True).cpu()
    top_latent = latents[mask[latents]][0].item()
    print("Best feature:", top_latent)

    elems, seen_i_im = [], set()

    for i_im, values_p in zip(top_img_i[top_latent].tolist(), top_values[top_latent]):
        if i_im in seen_i_im:
            continue

        example = in1k_dataset[i_im]
        elems.append(
            saev.visuals.GridElement(example["image"], example["label"], values_p)
        )
        seen_i_im.add(i_im)

    # How to scale values.
    upper = None
    if top_values[top_latent].numel() > 0:
        upper = top_values[top_latent].max().item()

    images = [saev.visuals.make_img(elem, upper=upper) for elem in elems] * 4
    return images[:4]


with gr.Blocks() as demo:
    image_picker = gr.Number(label="Validation Example")
    cell_picker = gr.Number(label="Image Patch")
    get_sae_examples_btn = gr.Button(value="Get SAE Examples")

    sae_example_images = [gr.Image() for _ in range(4)]

    get_sae_examples_btn.click(
        get_sae_examples,
        inputs=[image_picker, cell_picker],
        outputs=sae_example_images,
        api_name="get-sae-examples",
    )

demo.launch()
