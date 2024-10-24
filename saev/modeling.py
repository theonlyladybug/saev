import dataclasses
import gzip
import hashlib
import logging
import os
import pickle
import typing

import beartype
import datasets
import einops
import numpy as np
import open_clip
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import wandb

from . import helpers

###########
# HELPERS #
###########


@beartype.beartype
def get_cache_dir() -> str:
    """Get cache directory from environment variables, defaulting to the current working directory (.)"""
    cache_dir = ""
    for var in ("SAEV_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


##########
# Config #
##########


@beartype.beartype
@dataclasses.dataclass
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    # Data Generating Function (Model + Training Distibuion)
    image_width: int = 224
    image_height: int = 224
    model: str = "ViT-L-14/openai"
    module_name: str = "resid"
    block_layer: int = -2
    dataset_path: str = "ILSVRC/imagenet-1k"
    n_workers: int = 8

    # SAE Parameters
    d_in: int = 1024

    # Activation Store Parameters
    n_epochs: int = 3
    n_batches_in_store: int = 15
    vit_batch_size: int = 1024

    # SAE Parameters
    expansion_factor: int = 64

    # Training Parameters
    l1_coefficient: float = 0.00008
    lr: float = 0.0004
    lr_warm_up_steps: int = 500
    batch_size: int = 1024

    # Resampling protocol args
    use_ghost_grads: bool = True
    feature_sampling_window: int = 64
    resample_batches: int = 32
    feature_reinit_scale: float = 0.2
    dead_feature_window: int = 64
    dead_feature_estimation_method: str = "no_fire"
    dead_feature_threshold: float = 1e-6

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "saev"
    wandb_log_freq: int = 10

    # Misc
    device: str = "cuda"
    seed: int = 42
    dtype: torch.dtype = torch.float32
    checkpoint_path: str = "checkpoints"

    @property
    def store_size(self) -> int:
        return self.n_batches_in_store * self.batch_size

    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion_factor

    @property
    def run_name(self) -> str:
        return (
            f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-epochs-{self.n_epochs}"
        )

    def __post_init__(self):
        unique_id = wandb.util.generate_id()
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"


###############
# RecordedVit #
###############
# Depends on Config so has to come after it.


@jaxtyped(typechecker=beartype.beartype)
class RecordedVit(torch.nn.Module):
    n_layers: int
    model: torch.nn.Module

    def __init__(self, cfg: Config):
        super().__init__()

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
    cfg: Config
    shape: tuple[int, int]

    def __init__(self, cfg: Config, vit: RecordedVit, *, on_missing: str = "error"):
        self.cfg = cfg
        self._acts_filepath = get_acts_filepath(cfg)

        # Try to load the activations from disk. If they are missing, do something based on what 'on_missing' is.
        if os.path.isfile(self._acts_filepath):
            # Don't need to do anything if the file exists.
            pass
        elif on_missing == "error":
            raise RuntimeError(f"Activations are not saved at '{self._acts_filepath}'.")
        elif on_missing == "make":
            save_acts(cfg, vit)
        else:
            raise ValueError(f"Invalid value '{on_missing}' for arg 'on_missing'.")

        # Need the original dataset to calculate the length and save the class labels.
        dataset = datasets.load_dataset(
            cfg.dataset_path, split="train", trust_remote_code=True
        )

        self.shape = (len(dataset), cfg.d_in)
        self.labels = torch.tensor(dataset["label"])
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
def get_acts_filepath(cfg: Config) -> str:
    cfg_str = (
        str(cfg.image_width)
        + str(cfg.image_height)
        + cfg.model
        + cfg.module_name
        + str(cfg.block_layer)
        + cfg.dataset_path
    )
    acts_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()
    return os.path.join(get_cache_dir(), "saev-acts", acts_hash, "acts.bin")


@beartype.beartype
def save_acts(cfg: Config, vit: RecordedVit):
    # Make filepath.
    filepath = get_acts_filepath(cfg)
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)

    # Get dataloader
    dataset = datasets.load_dataset(
        cfg.dataset_path, split="train", trust_remote_code=True
    )

    preprocess = vit.make_img_transform()

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
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.n_workers,
        persistent_workers=cfg.n_workers > 0,
        shuffle=False,
        pin_memory=False,
    )
    n_batches = len(dataset) // cfg.vit_batch_size + 1

    # Make memmap'ed file.
    acts = np.memmap(
        filepath, mode="w+", dtype=np.float32, shape=(len(dataset), cfg.d_in)
    )

    # Calculate and write ViT activations.
    with torch.inference_mode():
        for batch in helpers.progress(dataloader, total=n_batches):
            index = batch.pop("index")
            images = batch.pop("image").to(cfg.device)
            _, cache = vit(images)
            acts[index] = cache[:, cfg.block_layer, 0, :]
    acts.flush()


#####################
# SparseAutoencoder #
#####################
# Depends on Config and CachedActivationsStore so has to come after them.


@jaxtyped(typechecker=beartype.beartype)
def get_sae_batches(
    cfg: Config, acts_store: CachedActivationsStore
) -> Float[Tensor, "store_size d_model"]:
    """
    Get a batch of vit activations
    """
    examples = []
    perm = np.random.default_rng(seed=cfg.seed).permutation(len(acts_store))
    perm = perm[: cfg.store_size]

    examples, _ = acts_store[perm]

    return examples


@beartype.beartype
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if not isinstance(cfg.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {cfg.d_in=}; {type(cfg.d_in)=}"
            )

        self.cfg = cfg
        self.l1_coefficient = cfg.l1_coefficient
        self.dtype = cfg.dtype
        self.device = cfg.device

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            )
        )
        self.b_enc = torch.nn.Parameter(
            torch.zeros(cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg.d_sae, cfg.d_in, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            # Anthropic normalizes this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = torch.nn.Parameter(
            torch.zeros(cfg.d_in, dtype=self.dtype, device=self.device)
        )

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x: Float[Tensor, "batch d_model"], dead_neuron_mask=None):
        # move x to correct dtype
        x = x.to(self.dtype)

        # Remove encoder bias as per Anthropic
        h_pre = (
            einops.einsum(
                x - self.b_dec, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
            )
            + self.b_enc
        )
        f_x = torch.nn.functional.relu(h_pre)

        x_hat = (
            einops.einsum(f_x, self.W_dec, "... d_sae, d_sae d_in -> ... d_in")
            + self.b_dec
        )

        # add config for whether l2 is normalized:
        mse_loss = (
            torch.pow((x_hat - x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        # gate on config and training so evals is not slowed down.
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask.sum() > 0:
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
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            # 3.
            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        mse_loss = mse_loss.mean()
        sparsity = torch.abs(f_x).sum(dim=1).mean(dim=(0,))
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid

        return x_hat, f_x, loss, mse_loss, l1_loss, mse_loss_ghost_resid

    @torch.no_grad()
    def initialize_b_dec(self, acts_store: CachedActivationsStore):
        previous_b_dec = self.b_dec.clone().cpu()
        assert isinstance(acts_store, CachedActivationsStore)
        all_activations = get_sae_batches(self.cfg, acts_store).detach().cpu()

        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def save_model(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}

        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        print(f"Saved model to {path}")

    @classmethod
    def load_from_pretrained(cls, path: str):
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                state_dict = torch.load(path, weights_only=False)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")

        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pkl.gz file: {e}"
                )
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz"
            )

        # Ensure the loaded state contains both 'cfg' and 'state_dict'
        if "cfg" not in state_dict or "state_dict" not in state_dict:
            raise ValueError(
                "The loaded state dictionary must contain 'cfg' and 'state_dict' keys"
            )

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        assert isinstance(self.cfg, Config)
        return f"sparse_autoencoder_{self.cfg.model}_{self.cfg.block_layer}_{self.cfg.module_name}_{self.cfg.d_sae}"


###########
# Session #
###########
# Depends on Config, RecordedVit, SparseAutoencoder and CachedActivationsStore so has to come after them.


@beartype.beartype
class Session(typing.NamedTuple):
    vit: RecordedVit
    sae: SparseAutoencoder
    acts_store: CachedActivationsStore

    @classmethod
    def from_cfg(cls, cfg: Config) -> "Session":
        vit = RecordedVit(cfg)
        vit.eval()
        for parameter in vit.model.parameters():
            parameter.requires_grad_(False)
        vit.to(cfg.device)

        sae = SparseAutoencoder(cfg)
        acts_store = CachedActivationsStore(cfg, vit, on_missing="error")

        return cls(vit, sae, acts_store)

    @classmethod
    def from_disk(cls, path) -> "Session":
        if torch.cuda.is_available():
            cfg = torch.load(path, weights_only=False)["cfg"]
        else:
            cfg = torch.load(path, map_location="cpu", weights_only=False)["cfg"]

        vit, _, acts_store = cls.from_cfg(cfg)
        sae = SparseAutoencoder.load_from_pretrained(path)
        return cls(vit, sae, acts_store)
