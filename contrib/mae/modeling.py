import os
import re
import typing

import beartype
import einops
import torch
import tyro
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import saev.helpers


@beartype.beartype
class MaskedAutoencoder(torch.nn.Module):
    class Output(typing.TypedDict):
        latents: Float[Tensor, "batch n d"]
        decoded: Float[Tensor, "batch n d"]
        ids_restore: Int[Tensor, "batch n"]

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
        mask_ratio: float,
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
            mask_ratio=mask_ratio,
            ln_eps=ln_eps,
        )
        self.decoder = Decoder(
            d_in=d_encoder,
            d=d_decoder,
            d_hidden=d_hidden_decoder,
            n_layers=n_layers_decoder,
            n_heads=n_heads_decoder,
            image_size_px=image_size_px,
            patch_size_px=patch_size_px,
            ln_eps=ln_eps,
        )

    def forward(
        self,
        x_B3WH: Float[Tensor, "batch 3 width height"],
        noise_BN: Float[Tensor, "batch n"] | None = None,
    ) -> Output:
        encoded = self.vit(x_B3WH, noise_BN=noise_BN)
        decoded_BND = self.decoder(encoded["x_BMD"], encoded["ids_restore_BN"])

        return self.Output(
            latents=encoded["x_BMD"],
            decoded=decoded_BND,
            ids_restore=encoded["ids_restore_BN"],
        )


configs = {
    "facebook/vit-mae-base": dict(
        d_encoder=768,
        d_hidden_encoder=3072,
        n_heads_encoder=12,
        n_layers_encoder=12,
        d_decoder=512,
        d_hidden_decoder=2048,
        n_heads_decoder=16,
        n_layers_decoder=8,
        image_size_px=(224, 224),
        patch_size_px=(16, 16),
        mask_ratio=0.75,
        ln_eps=1e-12,
    ),
}

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


@beartype.beartype
@torch.no_grad
def load_ckpt(ckpt: str, *, chunk_size_kb: int = 1024) -> MaskedAutoencoder:
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

    state_dict = torch.load(model_fpath, weights_only=True)
    updated_state_dict = {}
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
    model.eval()
    return model


@jaxtyped(typechecker=beartype.beartype)
def random_masking(
    x_BND: Float[Tensor, "batch n d"],
    mask_ratio: float,
    noise_BN: Float[Tensor, "batch n"] | None = None,
) -> tuple[
    Float[Tensor, "batch m d"], Float[Tensor, "batch n"], Int[Tensor, "batch n"]
]:
    """
    Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsorting random noise.
    """
    batch_size, n, d = x_BND.shape
    n_keep = int(n * (1 - mask_ratio))

    if noise_BN is None:
        # noise in [0, 1]
        torch.manual_seed(42)
        noise_BN = torch.rand(batch_size, n, device=x_BND.device)

    # sort noise for each sample
    # ascend: small is keep, large is remove
    ids_shuffled_BN = torch.argsort(noise_BN, dim=1).to(x_BND.device)
    ids_restore_BN = torch.argsort(ids_shuffled_BN, dim=1).to(x_BND.device)

    # keep the first subset
    ids_keep_BM = ids_shuffled_BN[:, :n_keep]
    ids_keep_BMD = ids_keep_BM[..., None].repeat(1, 1, d)
    x_BMD = torch.gather(x_BND, dim=1, index=ids_keep_BMD)

    # generate the binary mask: 0 is keep, 1 is remove
    mask_BN = torch.ones([batch_size, n], device=x_BND.device)
    mask_BN[:, :n_keep] = 0
    # unshuffle to get the binary mask
    mask_BN = torch.gather(mask_BN, dim=1, index=ids_restore_BN)

    return x_BMD, mask_BN, ids_restore_BN


@jaxtyped(typechecker=beartype.beartype)
class VisionTransformer(torch.nn.Module):
    class Output(typing.TypedDict):
        x_BMD: Float[Tensor, "batch m d"]
        ids_restore_BN: Int[Tensor, "batch n"]

    def __init__(
        self,
        *,
        d: int,
        d_hidden: int,
        n_heads: int,
        n_layers: int,
        image_size_px: tuple[int, int],
        patch_size_px: tuple[int, int],
        mask_ratio: float,
        ln_eps: float,
    ):
        super().__init__()
        self.embeddings = Embeddings(
            d=d,
            image_size_px=image_size_px,
            patch_size_px=patch_size_px,
            mask_ratio=mask_ratio,
        )
        self.encoder = Encoder(
            d=d,
            d_hidden=d_hidden,
            n_heads=n_heads,
            n_layers=n_layers,
            ln_eps=ln_eps,
        )
        self.layernorm = torch.nn.LayerNorm(d, eps=ln_eps)

    def forward(
        self,
        x_B3WH: Float[Tensor, "batch 3 width height"],
        noise_BN: Float[Tensor, "batch n"] | None = None,
    ) -> Float[Tensor, "batch ..."]:
        embedded = self.embeddings(x_B3WH, noise_BN=noise_BN)
        x_BMD = self.encoder(embedded["x_BMD"])
        x_BMD = self.layernorm(x_BMD)

        return self.Output(x_BMD=x_BMD, ids_restore_BN=embedded["ids_restore_BN"])


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
            TransformerBlock(d=d, d_hidden=d_hidden, n_heads=n_heads, ln_eps=ln_eps)
            for _ in range(n_layers)
        ])

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x_BMD: Float[Tensor, "batch m d"]) -> Float[Tensor, "batch m d"]:
        for layer in self.layers:
            x_BMD = layer(x_BMD)
        return x_BMD


@jaxtyped(typechecker=beartype.beartype)
class TransformerBlock(torch.nn.Module):
    def __init__(self, *, d: int, d_hidden: int, n_heads: int, ln_eps: float):
        super().__init__()
        self.attention = Attention(d=d, n_heads=n_heads)
        self.ffnn = Feedforward(d=d, d_hidden=d_hidden)
        self.layernorm1 = torch.nn.LayerNorm(d, eps=ln_eps)
        self.layernorm2 = torch.nn.LayerNorm(d, eps=ln_eps)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x: Float[Tensor, "batch n d"]) -> Float[Tensor, "batch n d"]:
        x_ = self.attention(self.layernorm1(x))

        x = x_ + x

        x_ = self.ffnn(self.layernorm2(x))
        return x_ + x


@jaxtyped(typechecker=beartype.beartype)
class Attention(torch.nn.Module):
    def __init__(self, *, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0, f"n_heads={n_heads} must evenly divide d={d}"

        self.n_heads = n_heads

        self.query = torch.nn.Linear(d, d)
        self.key = torch.nn.Linear(d, d)
        self.value = torch.nn.Linear(d, d)
        self.output = torch.nn.Linear(d, d)

    @jaxtyped(typechecker=beartype.beartype)
    def split(
        self, x_BND: Float[Tensor, "batch n d"]
    ) -> Float[Tensor, "batch n_heads n d_head"]:
        return einops.rearrange(
            x_BND,
            "batch n (n_heads d_head) -> batch n_heads n d_head",
            n_heads=self.n_heads,
        )

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x_BND: Float[Tensor, "batch n d"]) -> Float[Tensor, "batch n d"]:
        q_BHNd = self.split(self.query(x_BND))
        k_BHNd = self.split(self.key(x_BND))
        v_BHNd = self.split(self.value(x_BND))

        x_BHNd = torch.nn.functional.scaled_dot_product_attention(
            q_BHNd, k_BHNd, v_BHNd, dropout_p=0.0, is_causal=False, scale=None
        )

        x_BND = einops.rearrange(
            x_BHNd, "batch n_heads n d_head -> batch n (n_heads d_head)"
        ).contiguous()

        x_BND = self.output(x_BND)
        return x_BND


@jaxtyped(typechecker=beartype.beartype)
class Feedforward(torch.nn.Module):
    def __init__(self, *, d: int, d_hidden: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(d, d_hidden)
        self.linear2 = torch.nn.Linear(d_hidden, d)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x_BND: Float[Tensor, "batch n d"]) -> Float[Tensor, "batch n d"]:
        x_BNF = self.linear1(x_BND)
        x_BNF = torch.nn.functional.gelu(x_BNF)
        x_BND = self.linear2(x_BNF)
        return x_BND


@jaxtyped(typechecker=beartype.beartype)
class Embeddings(torch.nn.Module):
    class Output(typing.TypedDict):
        x_BMD: Float[Tensor, "batch m d"]
        mask_BN: Float[Tensor, "batch n"]
        ids_restore_BN: Int[Tensor, "batch n"]

    def __init__(
        self,
        *,
        d: int,
        image_size_px: tuple[int, int],
        patch_size_px: tuple[int, int],
        mask_ratio: float,
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

        self.mask_ratio = mask_ratio

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self,
        x_BCWH: Float[Tensor, "batch 3 height width"],
        noise_BN: Float[Tensor, "batch n"] | None = None,
    ) -> Output:
        batch_size, _, _, _ = x_BCWH.shape

        x_BND = self.patch_embeddings(x_BCWH) + self.position_embeddings[:, 1:, :]

        x_BMD, mask_BN, ids_restore_BN = random_masking(
            x_BND, self.mask_ratio, noise_BN=noise_BN
        )

        cls_x_11D = self.cls_token + self.position_embeddings[:, :1, :]

        cls_x_B1D = cls_x_11D.expand(batch_size, -1, -1)
        return self.Output(
            x_BMD=torch.cat((cls_x_B1D, x_BMD), dim=1),
            mask_BN=mask_BN,
            ids_restore_BN=ids_restore_BN,
        )


# CHECKED AGAINST REF---WORKS
@jaxtyped(typechecker=beartype.beartype)
class PatchEmbeddings(torch.nn.Module):
    def __init__(self, d: int, patch_size_px: tuple[int, int]):
        super().__init__()
        self.projection = torch.nn.Conv2d(
            3, d, kernel_size=patch_size_px, stride=patch_size_px
        )

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self, x_BCWH: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch n_patches d"]:
        return einops.rearrange(self.projection(x_BCWH), "batch d w h -> batch (w h) d")


@jaxtyped(typechecker=beartype.beartype)
class Decoder(torch.nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        d_hidden: int,
        n_layers: int,
        n_heads: int,
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
            TransformerBlock(d=d, d_hidden=d_hidden, n_heads=n_heads, ln_eps=ln_eps)
            for _ in range(n_layers)
        ])

        self.layernorm = torch.nn.LayerNorm(d, eps=ln_eps)
        self.head = torch.nn.Linear(d, patch_w_px * patch_h_px * 3)

    def forward(
        self,
        x_BMD: Float[Tensor, "batch m d_in"],
        ids_restore_BN: Int[Tensor, "batch n"],
    ) -> Float[Tensor, "batch n patch_pixels"]:
        batch_size, m, _ = x_BMD.shape
        _, n = ids_restore_BN.shape

        # Linear projection from encoder dimension to decoder dimension
        # CHECKED AGAINST REF---WORKS
        x_BMD = self.embd(x_BMD)

        _, _, d_decoder = x_BMD.shape

        # Add the mask tokens back
        n_mask_tokens = n + 1 - m
        masks_BOD = self.mask_token.repeat(batch_size, n_mask_tokens, 1)
        x_BND = torch.cat([x_BMD[:, 1:, :], masks_BOD], dim=1)  # no cls token

        index_BND = ids_restore_BN[..., None].repeat(1, 1, d_decoder).to(x_BND.device)
        x_BND = torch.gather(x_BND, dim=1, index=index_BND)

        x_BND = torch.cat([x_BMD[:, :1, :], x_BND], dim=1)  # append cls token

        # Add positional embeddings again.
        x_BND = x_BND + self.pos_embd

        for layer in self.layers:
            x_BND = layer(x_BND)

        x_BND = self.layernorm(x_BND)
        logits_BNP = self.head(x_BND)

        # Remove cls token
        logits_BNP = logits_BNP[:, 1:, :]

        return logits_BNP


def _compare_to_ref_impl(ckpt: str, dataset_root: str):
    """
    Compare against HF's implementation with:

    ```sh
    uv run python -m contrib.mae.modeling --ckpt facebook/vit-mae-base --dataset-root /nfs/$USERS/datasets/flowers102/train/
    ```
    """
    import torchvision.datasets
    import transformers
    from torchvision.transforms import v2

    ref_model = transformers.ViTMAEForPreTraining.from_pretrained(ckpt)
    my_model = load_ckpt(ckpt)

    # Same values from the official script.
    transform = v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.CenterCrop(size=(224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_root, transform=transform)

    batch_B3WH = torch.stack([dataset[i][0] for i in range(4)])

    torch.manual_seed(42)
    noise_BN = torch.rand(4, 196)

    ref_vit_out = ref_model.vit(batch_B3WH, noise=noise_BN)
    my_vit_out = my_model.vit(batch_B3WH, noise_BN=noise_BN)
    assert (ref_vit_out.last_hidden_state == my_vit_out["x_BMD"]).all()
    assert (ref_vit_out["ids_restore"] == my_vit_out["ids_restore_BN"]).all()

    ref_out = ref_model(batch_B3WH, noise=noise_BN, output_hidden_states=True)
    my_out = my_model(batch_B3WH, noise_BN=noise_BN)

    assert (ref_out["logits"] == my_out["decoded"]).all()


if __name__ == "__main__":
    tyro.cli(_compare_to_ref_impl)
