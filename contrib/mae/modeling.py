import os
import re

import beartype
import torch
import tyro
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import saev.helpers


@beartype.beartype
class MaskedAutoencoder(torch.nn.Module):
    def __init__(
        self,
        *,
        d_encoder: int,
        d_hidden_encoder: int,
        n_heads_encoder: int,
        n_layers_encoder: int,
        d_decoder: int,
        d_hidden_decoder: int,
        n_heads_decoder: int,
        n_layers_decoder: int,
        image_size_px: tuple[int, int],
        patch_size_px: tuple[int, int],
        ln_eps: float,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            d=d_encoder,
            d_hidden=d_hidden_encoder,
            n_heads=n_heads_encoder,
            n_layers=n_layers_encoder,
            image_size_px=image_size_px,
            patch_size_px=patch_size_px,
            ln_eps=ln_eps,
        )
        self.decoder = Decoder(
            d_in=d_encoder,
            d=d_decoder,
            d_hidden=d_hidden_decoder,
            n_layers=n_layers_decoder,
            image_size_px=image_size_px,
            patch_size_px=patch_size_px,
            ln_eps=ln_eps,
        )


configs = {
    "facebook/vit-mae-base": dict(
        d_encoder=768,
        d_hidden_encoder=3072,
        n_heads_encoder=12,
        n_layers_encoder=12,
        d_decoder=512,
        d_hidden_decoder=2048,
        n_heads_decoder=2,
        n_layers_decoder=8,
        image_size_px=(224, 224),
        patch_size_px=(16, 16),
        ln_eps=1e-12,
    ),
}


@beartype.beartype
@torch.inference_mode
def load_mae(ckpt: str, *, chunk_size_kb: int = 1024) -> MaskedAutoencoder:
    """
    Loads a pre-trained MAE ViT from disk.
    If it's not on disk, downloads the checkpoint from huggingface and then loads it into the `MaskedAutoencoder` class.
    """
    if ckpt not in configs:
        raise ValueError(f"Checkpoint {ckpt} not supported.")

    model_dname = f"saev--{ckpt.replace('/', '--')}"
    model_dpath = os.path.join(saev.helpers.get_cache_dir(), model_dname)
    model_fpath = os.path.join(model_dpath, "pytorch_model.bin")

    if not os.path.isfile(model_fpath):
        import requests
        import tqdm

        os.makedirs(model_dpath, exist_ok=True)

        url = f"https://huggingface.co/{ckpt}/resolve/main/pytorch_model.bin"
        chunk_size = int(chunk_size_kb * 1024)
        r = requests.get(url, stream=True)
        r.raise_for_status()

        n_bytes = int(r.headers["content-length"])

        with open(model_fpath, "wb") as fd:
            for chunk in tqdm.tqdm(
                r.iter_content(chunk_size=chunk_size),
                total=n_bytes / chunk_size,
                unit="b",
                unit_scale=1,
                unit_divisor=1024,
                desc="Downloading dataset",
            ):
                fd.write(chunk)
        print(f"Downloaded model: {model_fpath}.")

    state_dict = torch.load(model_fpath, weights_only=True, mmap=True)
    updated_state_dict = {}
    weight_map = {
        # Encoder attention
        "vit.encoder.layer.{}.attention.attention.query.weight": "vit.encoder.layers.{}.attention.query.weight",
        "vit.encoder.layer.{}.attention.attention.query.bias": "vit.encoder.layers.{}.attention.query.bias",
        "vit.encoder.layer.{}.attention.attention.value.weight": "vit.encoder.layers.{}.attention.value.weight",
        "vit.encoder.layer.{}.attention.attention.value.bias": "vit.encoder.layers.{}.attention.value.bias",
        "vit.encoder.layer.{}.attention.attention.key.weight": "vit.encoder.layers.{}.attention.key.weight",
        "vit.encoder.layer.{}.attention.attention.key.bias": "vit.encoder.layers.{}.attention.key.bias",
        "vit.encoder.layer.{}.attention.output.dense.weight": "vit.encoder.layers.{}.attention.output.weight",
        "vit.encoder.layer.{}.attention.output.dense.bias": "vit.encoder.layers.{}.attention.output.bias",
        # Encoder layernorms
        "vit.encoder.layer.{}.layernorm_before.weight": "vit.encoder.layers.{}.layernorm1.weight",
        "vit.encoder.layer.{}.layernorm_before.bias": "vit.encoder.layers.{}.layernorm1.bias",
        "vit.encoder.layer.{}.layernorm_after.weight": "vit.encoder.layers.{}.layernorm2.weight",
        "vit.encoder.layer.{}.layernorm_after.bias": "vit.encoder.layers.{}.layernorm2.bias",
        # Encoder FFNN
        "vit.encoder.layer.{}.intermediate.dense.weight": "vit.encoder.layers.{}.ffnn.linear1.weight",
        "vit.encoder.layer.{}.intermediate.dense.bias": "vit.encoder.layers.{}.ffnn.linear1.bias",
        "vit.encoder.layer.{}.output.dense.weight": "vit.encoder.layers.{}.ffnn.linear2.weight",
        "vit.encoder.layer.{}.output.dense.bias": "vit.encoder.layers.{}.ffnn.linear2.bias",
        # Decoder attention
        "decoder.decoder_layers.{}.attention.attention.query.weight": "decoder.layers.{}.attention.query.weight",
        "decoder.decoder_layers.{}.attention.attention.query.bias": "decoder.layers.{}.attention.query.bias",
        "decoder.decoder_layers.{}.attention.attention.key.weight": "decoder.layers.{}.attention.key.weight",
        "decoder.decoder_layers.{}.attention.attention.key.bias": "decoder.layers.{}.attention.key.bias",
        "decoder.decoder_layers.{}.attention.attention.value.weight": "decoder.layers.{}.attention.value.weight",
        "decoder.decoder_layers.{}.attention.attention.value.bias": "decoder.layers.{}.attention.value.bias",
        "decoder.decoder_layers.{}.attention.output.dense.weight": "decoder.layers.{}.attention.output.weight",
        "decoder.decoder_layers.{}.attention.output.dense.bias": "decoder.layers.{}.attention.output.bias",
        # Decoder layernorms
        "decoder.decoder_layers.{}.layernorm_before.weight": "decoder.layers.{}.layernorm1.weight",
        "decoder.decoder_layers.{}.layernorm_before.bias": "decoder.layers.{}.layernorm1.bias",
        "decoder.decoder_layers.{}.layernorm_after.weight": "decoder.layers.{}.layernorm2.weight",
        "decoder.decoder_layers.{}.layernorm_after.bias": "decoder.layers.{}.layernorm2.bias",
        # Decoder FFNN
        "decoder.decoder_layers.{}.intermediate.dense.weight": "decoder.layers.{}.ffnn.linear1.weight",
        "decoder.decoder_layers.{}.intermediate.dense.bias": "decoder.layers.{}.ffnn.linear1.bias",
        "decoder.decoder_layers.{}.output.dense.weight": "decoder.layers.{}.ffnn.linear2.weight",
        "decoder.decoder_layers.{}.output.dense.bias": "decoder.layers.{}.ffnn.linear2.bias",
        # General decoder stuff
        "decoder.decoder_embed.weight": "decoder.embd.weight",
        "decoder.decoder_embed.bias": "decoder.embd.bias",
        "decoder.decoder_norm.weight": "decoder.layernorm.weight",
        "decoder.decoder_norm.bias": "decoder.layernorm.bias",
        "decoder.decoder_pred.weight": "decoder.head.weight",
        "decoder.decoder_pred.bias": "decoder.head.bias",
        "decoder.decoder_pos_embed": "decoder.pos_embd",
    }
    for orig_key, value in state_dict.items():
        match = re.search(r"\.(\d+)\.", orig_key)
        if "layer" in orig_key and match:
            layer = match.group(1)
            abstract_key = orig_key.replace(f".{layer}.", ".{}.")
            new_key = weight_map.get(abstract_key, orig_key).replace("{}", layer)
        else:
            new_key = weight_map.get(orig_key, orig_key)

        updated_state_dict[new_key] = value

    model = MaskedAutoencoder(**configs[ckpt])
    model.load_state_dict(updated_state_dict, strict=True)
    return model


@jaxtyped(typechecker=beartype.beartype)
class VisionTransformer(torch.nn.Module):
    def __init__(
        self,
        *,
        d: int,
        d_hidden: int,
        n_heads: int,
        n_layers: int,
        image_size_px: tuple[int, int],
        patch_size_px: tuple[int, int],
        ln_eps: float,
    ):
        super().__init__()
        self.embeddings = Embeddings(
            d=d, image_size_px=image_size_px, patch_size_px=patch_size_px
        )
        self.encoder = Encoder(
            d=d,
            d_hidden=d_hidden,
            n_heads=n_heads,
            n_layers=n_layers,
            ln_eps=ln_eps,
        )
        self.layernorm = torch.nn.LayerNorm(d, eps=ln_eps)


@jaxtyped(typechecker=beartype.beartype)
class Encoder(torch.nn.Module):
    def __init__(
        self,
        *,
        d: int,
        d_hidden: int,
        n_heads: int,
        n_layers: int,
        ln_eps: float,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d=d, d_hidden=d_hidden, ln_eps=ln_eps)
            for _ in range(n_layers)
        ])

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self):
        breakpoint()


@jaxtyped(typechecker=beartype.beartype)
class TransformerBlock(torch.nn.Module):
    def __init__(self, *, d: int, d_hidden: int, ln_eps: float):
        super().__init__()
        self.attention = Attention(d=d)
        self.ffnn = Feedforward(d=d, d_hidden=d_hidden)
        self.layernorm1 = torch.nn.LayerNorm(d, eps=ln_eps)
        self.layernorm2 = torch.nn.LayerNorm(d, eps=ln_eps)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        x_ = self.attention(self.layernorm1(x))

        x = x_ + x

        x_ = self.ffnn(self.layernorm2(x))
        return x_ + x


@jaxtyped(typechecker=beartype.beartype)
class Attention(torch.nn.Module):
    def __init__(self, *, d: int):
        super().__init__()
        self.query = torch.nn.Linear(d, d)
        self.key = torch.nn.Linear(d, d)
        self.value = torch.nn.Linear(d, d)
        self.output = torch.nn.Linear(d, d)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self):
        breakpoint()


@jaxtyped(typechecker=beartype.beartype)
class Feedforward(torch.nn.Module):
    def __init__(self, *, d: int, d_hidden: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(d, d_hidden)
        self.linear2 = torch.nn.Linear(d_hidden, d)


@jaxtyped(typechecker=beartype.beartype)
class Embeddings(torch.nn.Module):
    def __init__(
        self,
        *,
        d: int,
        image_size_px: tuple[int, int],
        patch_size_px: tuple[int, int],
    ):
        super().__init__()

        image_w_px, image_h_px = image_size_px
        patch_w_px, patch_h_px = patch_size_px
        n_patches = (image_w_px // patch_w_px) * (image_h_px // patch_h_px)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d))
        self.position_embeddings = torch.nn.Parameter(
            torch.zeros(1, n_patches + 1, d), requires_grad=False
        )
        self.patch_embeddings = PatchEmbeddings(d=d, patch_size_px=patch_size_px)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, pixel_values: Float[Tensor, "batch 3 height width"]):
        breakpoint()


@jaxtyped(typechecker=beartype.beartype)
class PatchEmbeddings(torch.nn.Module):
    def __init__(self, d: int, patch_size_px: tuple[int, int]):
        super().__init__()
        self.projection = torch.nn.Conv2d(
            3, d, kernel_size=patch_size_px, stride=patch_size_px
        )

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, pixels: Int[Tensor, "batch 3 height width"]):
        breakpoint()


@jaxtyped(typechecker=beartype.beartype)
class Decoder(torch.nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        d_hidden: int,
        n_layers: int,
        patch_size_px: tuple[int, int],
        image_size_px: tuple[int, int],
        ln_eps: float,
    ):
        super().__init__()

        image_w_px, image_h_px = image_size_px
        patch_w_px, patch_h_px = patch_size_px
        n_patches = (image_w_px // patch_w_px) * (image_h_px // patch_h_px)

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, d))

        self.embd = torch.nn.Linear(d_in, d)

        self.pos_embd = torch.nn.Parameter(
            torch.zeros(1, n_patches + 1, d), requires_grad=False
        )

        self.layers = torch.nn.ModuleList([
            TransformerBlock(d=d, d_hidden=d_hidden, ln_eps=ln_eps)
            for _ in range(n_layers)
        ])

        self.layernorm = torch.nn.LayerNorm(d, eps=ln_eps)
        self.head = torch.nn.Linear(d, patch_w_px * patch_h_px * 3)


if __name__ == "__main__":
    print(tyro.cli(load_mae))
