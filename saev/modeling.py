"""
modeling is the main module for the saev package and contains all the important non-config classes.

It's fine for this package to be slow to import (see `saev.config` for a discussion of import times).
"""

import hashlib
import io
import json
import logging
import os
import typing

import beartype
import einops
import numpy as np
import torch
import wids
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from . import config, helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


###########
# HELPERS #
###########


@beartype.beartype
def get_cache_dir() -> str:
    """
    Get cache directory from environment variables, defaulting to the current working directory (.)

    Returns:
        A path to a cache directory (might not exist yet).
    """
    cache_dir = ""
    for var in ("SAEV_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


###############
# RecordedVit #
###############
# Depends on Config so has to come after it.


@jaxtyped(typechecker=beartype.beartype)
class RecordedVit(torch.nn.Module):
    n_layers: int
    model: torch.nn.Module
    _storage: Float[Tensor, "batch n_layers 1 dim"] | None

    def __init__(self, cfg: config.Config):
        super().__init__()

        import open_clip

        if cfg.model.startswith("hf-hub:"):
            clip, _img_transform = open_clip.create_model_from_pretrained(cfg.model)
        else:
            arch, ckpt = cfg.model.split("/")
            clip, _img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=get_cache_dir()
            )

        self.model = clip.visual
        self.model.proj = None
        self.model.output_tokens = True  # type: ignore
        self._img_transform = _img_transform

        self._storage = None
        self._i = 0

        self.n_layers = 0
        for layer in self.model.transformer.resblocks:
            layer.register_forward_hook(self.hook)
            self.n_layers += 1

        self.logger = logging.getLogger("recorder")

    def make_img_transform(self):
        return self._img_transform

    def forward(
        self, *args, **kwargs
    ) -> tuple[typing.Any, Float[Tensor, "batch n_layers 1 dim"]]:
        self.reset()
        output = self.model(*args, **kwargs)
        return output, self.activations

    def hook(
        self, module, args: tuple, output: Float[Tensor, "batch n_layers dim"]
    ) -> None:
        if self._storage is None:
            batch, _, dim = output.shape
            self._storage = torch.zeros(
                (batch, self.n_layers, 1, dim), device=output.device
            )

        if self._storage[:, self._i, 0, :].shape != output[:, 0, :].shape:
            batch, _, dim = output.shape

            old_batch, _, _, old_dim = self._storage.shape
            msg = "Output shape does not match storage shape: (batch) %d != %d or (dim) %d != %d"
            self.logger.warning(msg, old_batch, batch, old_dim, dim)

            self._storage = torch.zeros(
                (batch, self.n_layers, 1, dim), device=output.device
            )

        # Image token only.
        self._storage[:, self._i, 0, :] = output[:, 0, :]
        self._i += 1

    def reset(self):
        self._i = 0

    @property
    def activations(self) -> Float[Tensor, "batch n_layers 1 dim"]:
        if self._storage is None:
            raise RuntimeError("First call model()")
        return torch.clone(self._storage).cpu()


##########################
# CachedActivationsStore #
##########################
# Depends on Config and RecordedViT so has to come after them.


@jaxtyped(typechecker=beartype.beartype)
class CachedActivationsStore(torch.utils.data.Dataset):
    cfg: config.Config
    shape: tuple[int, int]

    def __init__(
        self, cfg: config.Config, vit: RecordedVit | None, *, on_missing: str = "error"
    ):
        self.cfg = cfg
        self._acts_filepath = get_acts_filepath(cfg)

        # Try to load the activations from disk. If they are missing, do something based on what 'on_missing' is.
        if os.path.isfile(self._acts_filepath):
            # Don't need to do anything if the file exists.
            pass
        elif on_missing == "error":
            raise RuntimeError(f"Activations are not saved at '{self._acts_filepath}'.")
        elif on_missing == "make":
            if vit is None:
                raise RuntimeError("Need a ViT in order to dump activations.")
            dump_acts(cfg, vit)
        else:
            raise ValueError(f"Invalid value '{on_missing}' for arg 'on_missing'.")

        self.shape = (cfg.data.n_imgs, cfg.d_vit)
        # TODO
        # self.labels = torch.tensor(dataset["label"])
        self.labels = None

        self._acts = np.memmap(
            self._acts_filepath, mode="r", dtype=np.float32, shape=self.shape
        )

    def __getitem__(self, i: int) -> tuple[Float[Tensor, " d_model"], Int[Tensor, ""]]:
        # TODO: also works with numpy arrays. How can we document this behavior in the typesystem?
        return torch.from_numpy(self._acts[i]), torch.tensor(i)

    def __len__(self) -> int:
        length, _ = self.shape
        return length


@beartype.beartype
def get_acts_filepath(cfg: config.Config) -> str:
    """
    Return the activations filepath based on the relevant values of a config.

    Args:
        cfg: Config for experiment.

    Returns:
        Filepath to where activations should be dumped/loaded from.
    """
    cfg_str = (
        str(cfg.image_width)
        + str(cfg.image_height)
        + cfg.model
        + cfg.module_name
        + str(cfg.block_layer)
        + str(cfg.data)
    )
    acts_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()
    return os.path.join(get_cache_dir(), "saev-acts", acts_hash, "acts.bin")


@beartype.beartype
def get_imagenet_dataloader(
    cfg: config.Config, preprocess
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for Imagenet loaded from Huggingface.

    Args:
        cfg: Config.
        preprocess: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches, `'index'` keys containing original dataset indices and `'label'` keys containing label batches.
    """
    assert isinstance(cfg.data, config.Imagenet)

    import datasets

    dataset = datasets.load_dataset(
        cfg.data.name, split="train", trust_remote_code=True
    )

    @beartype.beartype
    def add_index(example, indices: list[int]):
        example["index"] = indices
        return example

    @beartype.beartype
    def map_fn(example: dict[str, list]):
        example["image"] = [preprocess(img) for img in example["image"]]
        return example

    dataset = (
        dataset.map(add_index, with_indices=True, batched=True)
        .to_iterable_dataset(num_shards=cfg.n_workers or 8)
        .map(map_fn, batched=True, batch_size=32)
        .with_format("torch")
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.vit_batch_size,
        drop_last=False,
        num_workers=cfg.n_workers,
        persistent_workers=cfg.n_workers > 0,
        shuffle=False,
        pin_memory=False,
    )
    return dataloader


@beartype.beartype
def get_tol_dataloader(cfg: config.Config, preprocess) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for the TreeOfLife-10M dataset.

    Currently does not include a true index or label in the loaded examples.

    Args:
        cfg: Config.
        preprocess: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches.
    """
    assert isinstance(cfg.data, config.TreeOfLife)

    def transform(sample: dict):
        return {"image": preprocess(sample[".jpg"]), "index": sample["__key__"]}

    dataset = wids.ShardListDataset(cfg.data.metadata).add_transform(transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.vit_batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        persistent_workers=False,
    )

    return dataloader


@beartype.beartype
def dump_acts(cfg: config.Config, vit: RecordedVit):
    # Make filepath.
    filepath = get_acts_filepath(cfg)
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)

    # Get dataloader
    preprocess = vit.make_img_transform()
    if isinstance(cfg.data, config.Imagenet):
        dataloader = get_imagenet_dataloader(cfg, preprocess)
    elif isinstance(cfg.data, config.TreeOfLife):
        dataloader = get_tol_dataloader(cfg, preprocess)
    else:
        typing.assert_never(cfg.data)

    # Make memmap'ed file.
    acts = np.memmap(
        filepath, mode="w+", dtype=np.float32, shape=(cfg.data.n_imgs, cfg.d_vit)
    )

    n_batches = cfg.data.n_imgs // cfg.vit_batch_size + 1

    vit = vit.to(cfg.device)

    # Calculate and write ViT activations.
    with torch.inference_mode():
        for i, batch in helpers.progress(enumerate(dataloader), total=n_batches):
            images = batch.pop("image").to(cfg.device)
            index = np.arange(
                i * cfg.vit_batch_size, i * cfg.vit_batch_size + len(images)
            )
            _, cache = vit(images)
            acts[index] = cache[:, cfg.block_layer, 0, :]

    acts.flush()


#####################
# SparseAutoencoder #
#####################
# Depends on Config and CachedActivationsStore so has to come after them.


@jaxtyped(typechecker=beartype.beartype)
def get_sae_batches(
    cfg: config.Config, acts_store: CachedActivationsStore
) -> Float[Tensor, "reinit_size d_model"]:
    """
    Get a batch of vit activations to re-initialize the SAE.

    Args:
        cfg: Config.
        acts_store: Activation store.
    """
    examples = []
    perm = np.random.default_rng(seed=cfg.seed).permutation(len(acts_store))
    perm = perm[: cfg.reinit_size]

    examples, _ = acts_store[perm]

    return examples


@beartype.beartype
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    l1_coeff: float
    use_ghost_grads: bool

    def __init__(self, d_vit: int, d_sae: int, l1_coeff: float, use_ghost_grads: bool):
        super().__init__()

        self.l1_coeff = l1_coeff
        self.use_ghost_grads = use_ghost_grads

        # Initialize the weights.
        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_vit, d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_vit))
        )

        with torch.no_grad():
            # Anthropic normalizes this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = torch.nn.Parameter(torch.zeros(d_vit))

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x: Float[Tensor, "batch d_model"], dead_neuron_mask=None):
        # Remove encoder bias as per Anthropic
        h_pre = (
            einops.einsum(
                x - self.b_dec, self.W_enc, "... d_vit, d_vit d_sae -> ... d_sae"
            )
            + self.b_enc
        )
        f_x = torch.nn.functional.relu(h_pre)

        x_hat = (
            einops.einsum(f_x, self.W_dec, "... d_sae, d_sae d_vit -> ... d_vit")
            + self.b_dec
        )

        # add config for whether l2 is normalized:
        mse_loss = (
            torch.pow((x_hat - x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        ghost_loss = torch.tensor(0.0, dtype=mse_loss.dtype, device=mse_loss.device)
        # gate on config and training so evals is not slowed down.
        if self.use_ghost_grads and self.training and dead_neuron_mask.sum() > 0:
            assert dead_neuron_mask is not None

            # ghost protocol

            # 1.
            residual = x - x_hat
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(h_pre[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out *= norm_scaling_factor[:, None].detach()

            # 3.
            ghost_loss = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (ghost_loss + 1e-6)).detach()
            ghost_loss *= mse_rescaling_factor

        ghost_loss = ghost_loss.mean()
        mse_loss = mse_loss.mean()
        sparsity = torch.abs(f_x).sum(dim=1).mean(dim=(0,))
        l1_loss = self.l1_coeff * sparsity
        loss = mse_loss + l1_loss + ghost_loss

        return x_hat, f_x, loss, mse_loss, l1_loss, ghost_loss

    @torch.no_grad()
    def init_b_dec(self, cfg: config.Config, acts_store: CachedActivationsStore):
        previous_b_dec = self.b_dec.clone().cpu()

        all_activations = get_sae_batches(cfg, acts_store).detach().cpu()
        mean = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - mean, dim=-1)

        print(f"Prev dist: {previous_distances.median(0).values.mean().item()}")
        print(f"New dist: {distances.median(0).values.mean().item()}")

        self.b_dec.data = mean.to(self.b_dec.dtype).to(self.b_dec.device)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_vit) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_vit, d_sae d_vit -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_vit -> d_sae d_vit",
        )


@beartype.beartype
def dump(cfg: config.Config, sae: SparseAutoencoder, run_id: str):
    filepath = f"{cfg.ckpt_path}/{run_id}/sae.pt"

    sae_kwargs = dict(
        d_vit=cfg.d_vit,
        d_sae=cfg.d_sae,
        l1_coeff=cfg.l1_coeff,
        use_ghost_grads=cfg.use_ghost_grads,
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as fd:
        kwargs_str = json.dumps(sae_kwargs)
        fd.write((kwargs_str + "\n").encode("utf-8"))
        torch.save(sae.state_dict(), fd)


@beartype.beartype
def load(cfg: config.Config, run_id: str) -> SparseAutoencoder:
    filepath = f"{cfg.ckpt_path}/{run_id}/sae.pt"

    with open(filepath, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())

    model = SparseAutoencoder(**kwargs)
    state_dict = torch.load(buffer, weights_only=True, map_location=cfg.device)
    model.load_state_dict(state_dict)
    return model


###########
# Session #
###########
# Depends on Config, RecordedVit, SparseAutoencoder and CachedActivationsStore so has to come after them.


@beartype.beartype
class Session(typing.NamedTuple):
    """
    Session is a group of instances of the main classes for saev experiments.
    """

    cfg: config.Config
    vit: RecordedVit
    sae: SparseAutoencoder
    acts_store: CachedActivationsStore

    @classmethod
    def from_cfg(cls, cfg: config.Config, *, on_missing: str = "error") -> "Session":
        """
        Load a new session from an existing config.

        Args:
            cfg: Confg.
            on_missing: passed to the `CachedActivationsStore`.

        Returns:
            A session with a config, ViT, SAE and activation store.
        """
        vit = RecordedVit(cfg)
        vit.eval()
        for parameter in vit.model.parameters():
            parameter.requires_grad_(False)
        vit.to(cfg.device)

        sae = SparseAutoencoder(cfg.d_vit, cfg.d_sae, cfg.l1_coeff, cfg.use_ghost_grads)
        acts_store = CachedActivationsStore(cfg, vit, on_missing=on_missing)

        return cls(cfg, vit, sae, acts_store)

    @classmethod
    def from_disk(cls, path) -> "Session":
        if torch.cuda.is_available():
            cfg = torch.load(path, weights_only=False)["cfg"]
        else:
            cfg = torch.load(path, map_location="cpu", weights_only=False)["cfg"]

        cfg, vit, _, acts_store = cls.from_cfg(cfg)
        sae = SparseAutoencoder.load_from_pretrained(path)
        return cls(cfg, vit, sae, acts_store)
