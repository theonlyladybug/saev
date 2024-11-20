"""
To save lots of activations, we want to do things in parallel, with lots of slurm jobs, and save multiple files, rather than just one.

This module handles that additional complexity.

Conceptually, activations are either thought of as

1. A single [n_imgs x n_layers x (n_patches + 1), d_vit] tensor. This is a *dataset*
2. Multiple [n_imgs_per_shard, n_layers, (n_patches + 1), d_vit] tensors. This is a set of sharded activations.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
import shutil
import typing

import beartype
import numpy as np
import torch
import torchvision.datasets
import wids
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor

from . import config, helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


@jaxtyped(typechecker=beartype.beartype)
class VitRecorder(torch.nn.Module):
    cfg: config.Activations
    _storage: Float[Tensor, "batch n_layers all_patches dim"] | None
    _i: int

    def __init__(
        self, cfg: config.Activations, patches: slice = slice(None, None, None)
    ):
        super().__init__()

        self.cfg = cfg
        self.patches = patches
        self._storage = None
        self._i = 0
        self.logger = logging.getLogger(f"recorder({cfg.model_org}:{cfg.model_ckpt})")

    def register(self, modules: list[torch.nn.Module]):
        for i in self.cfg.layers:
            modules[i].register_forward_hook(self.hook)
        return self

    def hook(
        self, module, args: tuple, output: Float[Tensor, "batch n_layers dim"]
    ) -> None:
        if self._storage is None:
            batch, _, dim = output.shape
            self._storage = self._empty_storage(batch, dim, output.device)

        if self._storage[:, self._i, 0, :].shape != output[:, 0, :].shape:
            batch, _, dim = output.shape

            old_batch, _, _, old_dim = self._storage.shape
            msg = "Output shape does not match storage shape: (batch) %d != %d or (dim) %d != %d"
            self.logger.warning(msg, old_batch, batch, old_dim, dim)

            self._storage = self._empty_storage(batch, dim, output.device)

        self._storage[:, self._i] = output[:, self.patches, :].detach()
        self._i += 1

    def _empty_storage(self, batch: int, dim: int, device: torch.device):
        n_patches_per_img = self.cfg.n_patches_per_img
        if self.cfg.cls_token:
            n_patches_per_img += 1

        return torch.zeros(
            (batch, len(self.cfg.layers), n_patches_per_img, dim), device=device
        )

    def reset(self):
        self._i = 0

    @property
    def activations(self) -> Float[Tensor, "batch n_layers all_patches dim"]:
        if self._storage is None:
            raise RuntimeError("First call model()")
        return self._storage.cpu()


@jaxtyped(typechecker=beartype.beartype)
class Clip(torch.nn.Module):
    def __init__(self, cfg: config.Activations):
        super().__init__()

        assert cfg.model_org == "clip"

        import open_clip

        if cfg.model_ckpt.startswith("hf-hub:"):
            clip, self._img_transform = open_clip.create_model_from_pretrained(
                cfg.model_ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = cfg.model_ckpt.split("/")
            clip, self._img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )

        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model

        assert not isinstance(self.model, open_clip.timm_model.TimmModel)
        self.recorder = VitRecorder(cfg).register(self.model.transformer.resblocks)

    def make_img_transform(self):
        return self._img_transform

    def forward(self, batch: Float[Tensor, "batch 3 width height"]):
        self.recorder.reset()
        result = self.model(batch)
        return result, self.recorder.activations


@jaxtyped(typechecker=beartype.beartype)
class Siglip(torch.nn.Module):
    def __init__(self, cfg: config.Activations):
        super().__init__()
        assert cfg.model_org == "siglip"

        import open_clip

        if cfg.model_ckpt.startswith("hf-hub:"):
            clip, self._img_transform = open_clip.create_model_from_pretrained(
                cfg.model_ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = cfg.model_ckpt.split("/")
            clip, self._img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )

        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model

        assert isinstance(self.model, open_clip.timm_model.TimmModel)
        self.recorder = VitRecorder(cfg).register(self.model.trunk.blocks)

    def make_img_transform(self):
        return self._img_transform

    def forward(self, batch: Float[Tensor, "batch 3 width height"]):
        self.recorder.reset()
        result = self.model(batch)
        return result, self.recorder.activations


@jaxtyped(typechecker=beartype.beartype)
class TimmVit(torch.nn.Module):
    def __init__(self, cfg: config.Activations):
        super().__init__()
        assert cfg.model_org == "timm"
        import timm

        err_msg = "You are trying to load a non-ViT checkpoint; the `img_encode()` method assumes `model.forward_features()` will return features with shape (batch, n_patches, dim) which is not true for non-ViT checkpoints."
        assert "vit" in cfg.model_ckpt, err_msg
        self.model = timm.create_model(cfg.model_ckpt, pretrained=True)

        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self._img_transform = timm.data.create_transform(**data_cfg, is_training=False)

        self.recorder = VitRecorder(cfg).register(self.model.blocks)

    def make_img_transform(self):
        return self._img_transform

    def forward(self, batch: Float[Tensor, "batch 3 width height"]):
        self.recorder.reset()

        patches = self.model.forward_features(batch)
        # Use [CLS] token if it exists for img representation, otherwise do a maxpool
        if self.model.num_prefix_tokens > 0:
            img = patches[:, 0, ...]
        else:
            img = patches.max(axis=1).values

        # Return only the [CLS] token and the patches.
        patches = patches[:, self.model.num_prefix_tokens :, ...]

        return torch.cat((img[:, None, :], patches), axis=1), self.recorder.activations


@jaxtyped(typechecker=beartype.beartype)
class DinoV2(torch.nn.Module):
    def __init__(self, cfg: config.Activations):
        super().__init__()

        assert cfg.model_org == "dinov2"

        self.model = torch.hub.load("facebookresearch/dinov2", cfg.model_ckpt)

        n_reg = self.model.num_register_tokens
        patches = torch.cat((
            torch.tensor([0]),  # CLS token
            torch.arange(n_reg + 1, n_reg + 1 + cfg.n_patches_per_img),  # patches
        ))

        self.recorder = VitRecorder(cfg, patches).register(self.model.blocks)

    def make_img_transform(self):
        from torchvision.transforms import v2

        return v2.Compose([
            v2.Resize(size=256),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])

    def forward(self, batch: Float[Tensor, "batch 3 width height"]):
        self.recorder.reset()

        dct = self.model.forward_features(batch)

        features = torch.cat(
            (dct["x_norm_clstoken"][:, None, :], dct["x_norm_patchtokens"]), axis=1
        )
        return features, self.recorder.activations


@beartype.beartype
def make_vit(cfg: config.Activations):
    if cfg.model_org == "timm":
        return TimmVit(cfg)
    elif cfg.model_org == "clip":
        return Clip(cfg)
    elif cfg.model_org == "siglip":
        return Siglip(cfg)
    elif cfg.model_org == "dinov2":
        return DinoV2(cfg)
    else:
        typing.assert_never(cfg.model_org)


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    Dataset of activations from disk.
    """

    class Example(typing.NamedTuple):
        """Individual example."""

        vit_acts: Float[Tensor, " d_vit"]
        img_i: Int[Tensor, ""]
        patch_i: Int[Tensor, ""]

    cfg: config.DataLoad
    """Configuration; set via CLI args."""
    metadata: "Metadata"
    """Activations metadata; automatically loaded from disk."""
    layer_index: int
    """Layer index into the shards if we are choosing a specific layer."""
    scalar: float
    """Normalizing scalar such that ||x / scalar ||_2 ~= sqrt(d_vit)."""
    act_mean: Float[Tensor, " d_vit"]
    """Mean activation."""

    def __init__(self, cfg: config.DataLoad):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shard_root):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shard_root}'.")

        metadata_fpath = os.path.join(self.cfg.shard_root, "metadata.json")
        self.metadata = Metadata.load(metadata_fpath)

        # Pick a really big number so that if you accidentally use this when you shouldn't, you get an out of bounds IndexError.
        self.layer_index = 1_000_000
        if isinstance(self.cfg.layer, int):
            err_msg = f"Non-exact matches for .layer field not supported; {self.cfg.layer} not in {self.metadata.layers}."
            assert self.cfg.layer in self.metadata.layers, err_msg
            self.layer_index = self.metadata.layers.index(self.cfg.layer)

        # Premptively set these values so that preprocess() doesn't freak out.
        self.scalar = 1.0
        self.act_mean = torch.zeros(self.d_vit)

        # If neither of these are true, we can skip this work.
        if self.cfg.scale_mean or self.cfg.scale_norm:
            # Load a random subset of samples to calculate the mean activation and mean L2 norm.
            perm = np.random.default_rng(seed=42).permutation(len(self))
            perm = perm[: cfg.n_random_samples]

            samples, _, _ = zip(*[
                self[p.item()]
                for p in helpers.progress(
                    perm, every=25_000, desc="examples to calc means"
                )
            ])
            samples = torch.stack(samples)
            if samples.abs().max() > 1e3:
                raise ValueError(
                    "You found an abnormally large activation {example.abs().max().item():.5f} that will mess up your L2 mean."
                )

            # Activation mean
            if self.cfg.scale_mean:
                self.act_mean = samples.mean(axis=0)
                if (self.act_mean > 1e3).any():
                    raise ValueError(
                        "You found an abnormally large activation that is messing up your activation mean."
                    )

            # Norm
            if self.cfg.scale_norm:
                l2_mean = torch.linalg.norm(samples - self.act_mean, axis=1).mean()
                if l2_mean > 1e3:
                    raise ValueError(
                        "You found an abnormally large activation that is messing up your L2 mean."
                    )

                self.scalar = l2_mean / math.sqrt(self.d_vit)

    def transform(self, act: Float[np.ndarray, " d_vit"]) -> Float[Tensor, " d_vit"]:
        """
        Apply a scalar normalization so the mean squared L2 norm is same as d_vit. This is from 'Scaling Monosemanticity':

        > As a preprocessing step we apply a scalar normalization to the model activations so their average squared L2 norm is the residual stream dimension

        So we divide by self.scalar which is the datasets (approximate) L2 mean before normalization divided by sqrt(d_vit).
        """
        act = torch.from_numpy(act.copy())
        act = act.clamp(-self.cfg.clamp, self.cfg.clamp)
        return (act - self.act_mean) / self.scalar

    @property
    def d_vit(self) -> int:
        """Dimension of the underlying vision transformer's embedding space."""
        return self.metadata.d_vit

    @jaxtyped(typechecker=beartype.beartype)
    def __getitem__(self, i: int) -> Example:
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", int()):
                img_act = self.get_img_patches(i)
                # Select layer's cls token.
                act = img_act[self.layer_index, 0, :]
                return self.Example(
                    self.transform(act), torch.tensor(i), torch.tensor(-1)
                )
            case ("cls", "meanpool"):
                img_act = self.get_img_patches(i)
                # Select cls tokens from across all layers
                cls_act = img_act[:, 0, :]
                # Meanpool over the layers
                act = cls_act.mean(axis=0)
                return self.Example(
                    self.transform(act), torch.tensor(i), torch.tensor(-1)
                )
            case ("meanpool", int()):
                img_act = self.get_img_patches(i)
                # Select layer's patches.
                layer_act = img_act[self.layer_index, 1:, :]
                # Meanpool over the patches
                act = layer_act.mean(axis=0)
                return self.Example(
                    self.transform(act), torch.tensor(i), torch.tensor(-1)
                )
            case ("meanpool", "meanpool"):
                img_act = self.get_img_patches(i)
                # Select all layer's patches.
                act = img_act[:, 1:, :]
                # Meanpool over the layers and patches
                act = act.mean(axis=(0, 1))
                return self.Example(
                    self.transform(act), torch.tensor(i), torch.tensor(-1)
                )
            case ("patches", int()):
                n_imgs_per_shard = (
                    self.metadata.n_patches_per_shard
                    // len(self.metadata.layers)
                    // (self.metadata.n_patches_per_img + 1)
                )
                n_examples_per_shard = (
                    n_imgs_per_shard * self.metadata.n_patches_per_img
                )

                shard = i // n_examples_per_shard
                pos = i % n_examples_per_shard

                acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
                shape = (
                    n_imgs_per_shard,
                    len(self.metadata.layers),
                    self.metadata.n_patches_per_img + 1,
                    self.metadata.d_vit,
                )
                acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
                # Choose the layer and the non-CLS tokens.
                acts = acts[:, self.layer_index, 1:]

                # Choose a patch among n and the patches.
                act = acts[
                    pos // self.metadata.n_patches_per_img,
                    pos % self.metadata.n_patches_per_img,
                ]
                return self.Example(
                    self.transform(act),
                    # What image is this?
                    torch.tensor(i // self.metadata.n_patches_per_img),
                    torch.tensor(i % self.metadata.n_patches_per_img),
                )
            case _:
                print((self.cfg.patches, self.cfg.layer))
                typing.assert_never((self.cfg.patches, self.cfg.layer))

    def get_shard_patches(self):
        raise NotImplementedError()

    def get_img_patches(
        self, i: int
    ) -> Float[np.ndarray, "n_layers all_patches d_vit"]:
        n_imgs_per_shard = (
            self.metadata.n_patches_per_shard
            // len(self.metadata.layers)
            // (self.metadata.n_patches_per_img + 1)
        )
        shard = i // n_imgs_per_shard
        pos = i % n_imgs_per_shard
        acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
        shape = (
            n_imgs_per_shard,
            len(self.metadata.layers),
            self.metadata.n_patches_per_img + 1,
            self.metadata.d_vit,
        )
        acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
        # Note that this is not yet copied!
        return acts[pos]

    def __len__(self) -> int:
        """
        Dataset length depends on `patches` and `layer`.
        """
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", "all"):
                # Return a CLS token from a random image and random layer.
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("cls", int()):
                # Return a CLS token from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("cls", "meanpool"):
                # Return a CLS token from a random image and meanpool over all layers.
                return self.metadata.n_imgs
            case ("meanpool", "all"):
                # Return the meanpool of all patches from a random image and random layer.
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("meanpool", int()):
                # Return the meanpool of all patches from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("meanpool", "meanpool"):
                # Return the meanpool of all patches from a random image and meanpool over all layers.
                return self.metadata.n_imgs
            case ("patches", int()):
                # Return a patch from a random image, fixed layer, and random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("patches", "meanpool"):
                # Return a patch from a random image, meanpooled over all layers, and a random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("patches", "all"):
                # Return a patch from a random image, random layer and random patch.
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * self.metadata.n_patches_per_img
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))


@beartype.beartype
def setup(cfg: config.Activations):
    """
    Run dataset-specific setup. These setup functions can assume they are the only job running, but they should be idempotent; they should be safe (and ideally cheap) to run multiple times in a row.
    """
    if isinstance(cfg.data, config.ImagenetDataset):
        setup_imagenet(cfg)
    elif isinstance(cfg.data, config.TreeOfLifeDataset):
        setup_tol(cfg)
    elif isinstance(cfg.data, config.LaionDataset):
        setup_laion(cfg)
    elif isinstance(cfg.data, config.ImageFolderDataset):
        setup_imagefolder(cfg)
    elif isinstance(cfg.data, config.BrodenDataset):
        setup_broden(cfg)
    else:
        typing.assert_never(cfg.data)


@beartype.beartype
def setup_imagenet(cfg: config.Activations):
    assert isinstance(cfg.data, config.ImagenetDataset)


@beartype.beartype
def setup_tol(cfg: config.Activations):
    assert isinstance(cfg.data, config.TreeOfLifeDataset)


@beartype.beartype
def setup_laion(cfg: config.Activations):
    """
    Do setup for LAION dataloader.
    """
    assert isinstance(cfg.data, config.LaionDataset)

    import datasets
    import img2dataset
    import submitit

    logger = logging.getLogger("laion")

    # 1. Download cfg.data.n_imgs data urls.

    # Check if URL list exists.
    n_urls = 0
    if os.path.isfile(cfg.data.url_list_filepath):

        def blocks(files, size=65536):
            while True:
                b = files.read(size)
                if not b:
                    break
                yield b

        with open(cfg.data.url_list_filepath, "r") as fd:
            n_urls = sum(bl.count("\n") for bl in blocks(fd))

    # We use -1 just in case there's something wrong with our n_urls count.
    dumped_urls = n_urls >= cfg.data.n_imgs - 1

    # If we don't have all the image urls written, need to dump to a file.
    if not dumped_urls:
        logger.info("Dumping URLs to '%s'.", cfg.data.url_list_filepath)

        if os.path.isfile(cfg.data.url_list_filepath):
            logger.warning(
                "Overwriting existing list of %d URLs because we want %d URLs.",
                n_urls,
                cfg.data.n_imgs,
            )

        dataset = (
            datasets.load_dataset(cfg.data.name, streaming=True, split="train")
            .shuffle(cfg.seed)
            .filter(
                lambda example: example["status"] == "success"
                and example["height"] >= 256
                and example["width"] >= 256
            )
            .take(cfg.data.n_imgs)
        )

        with open(cfg.data.url_list_filepath, "w") as fd:
            for example in helpers.progress(
                dataset, every=5_000, desc="Writing URLs", total=cfg.data.n_imgs
            ):
                fd.write(f'{{"url": "{example["url"]}", "key": "{example["key"]}"}}\n')

    # 2. Download the images to a webdatset format using img2dataset
    # TODO: check whether images are downloaded. Read all the _stats.json files and see if we have all 10M.
    imgs_downloaded = False

    if not imgs_downloaded:

        def download(n_processes: int, n_threads: int):
            assert isinstance(cfg.data, config.LaionDataset)

            if os.path.exists(cfg.data.tar_dir):
                shutil.rmtree(cfg.data.tar_dir)

            img2dataset.download(
                url_list=cfg.data.url_list_filepath,
                input_format="jsonl",
                image_size=256,
                output_folder=cfg.data.tar_dir,
                processes_count=n_processes,
                thread_count=n_threads,
                resize_mode="keep_ratio",
                encode_quality=100,
                encode_format="webp",
                output_format="webdataset",
                oom_shard_count=6,
                ignore_ssl=not cfg.ssl,
            )

        if cfg.slurm:
            executor = submitit.SlurmExecutor(folder=cfg.log_to)
            executor.update_parameters(
                time=12 * 60,
                partition="cpuonly",
                cpus_per_task=64,
                stderr_to_stdout=True,
                account=cfg.slurm_acct,
            )
            job = executor.submit(download, 64, 256)
            job.result()
        else:
            download(8, 32)


@beartype.beartype
def setup_broden(cfg: config.Activations):
    assert isinstance(cfg.data, config.BrodenDataset)

    # logger = logging.getLogger("broden-setup")
    # url = "http://netdissect.csail.mit.edu/data/broden1_224.zip"
    if os.path.isfile(os.path.join(cfg.data.root, "index.csv")):
        return
    os.makedirs(cfg.data.root, exist_ok=True)
    breakpoint()


@beartype.beartype
def setup_imagefolder(cfg: config.Activations):
    assert isinstance(cfg.data, config.ImageFolderDataset)
    breakpoint()


@beartype.beartype
def get_dataloader(cfg: config.Activations, preprocess):
    """
    Gets the dataloader for the current experiment; delegates dataloader construction to dataset-specific functions.

    Args:
        cfg: Experiment config.
        preprocess: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches.
    """
    if isinstance(cfg.data, config.ImagenetDataset):
        dataloader = get_imagenet_dataloader(cfg, preprocess)
    elif isinstance(cfg.data, config.TreeOfLifeDataset):
        dataloader = get_tol_dataloader(cfg, preprocess)
    elif isinstance(cfg.data, config.LaionDataset):
        dataloader = get_laion_dataloader(cfg, preprocess)
    elif isinstance(cfg.data, config.ImageFolderDataset):
        dataloader = get_imagefolder_dataloader(cfg, preprocess)
    elif isinstance(cfg.data, config.BrodenDataset):
        dataloader = get_broden_dataloader(cfg, preprocess)
    else:
        typing.assert_never(cfg.data)

    return dataloader


@beartype.beartype
def get_laion_dataloader(
    cfg: config.Activations, preprocess
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for a subset of the LAION datasets.

    This requires several steps:

    1. Download list of image URLs
    2. Use img2dataset to download these images to webdataset format.
    3. Create a dataloader from these webdataset tar files.

    So that we don't have to redo any of these steps, we check on the existence of various files to check if this stuff is done already.
    """
    # 3. Create a webdataset loader over these images.
    # TODO: somehow we need to know which images are in the dataloader. Like, a way to actually go back to the original image. The HF dataset has a "key" column that is likely unique.
    breakpoint()


@beartype.beartype
def get_tol_dataloader(
    cfg: config.Activations, preprocess
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for the TreeOfLife-10M dataset.

    Currently does not include a true index or label in the loaded examples.

    Args:
        cfg: Config for loading activations.
        preprocess: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches.
    """
    assert isinstance(cfg.data, config.TreeOfLifeDataset)

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


class PreprocessedImageNet(torch.utils.data.Dataset):
    def __init__(self, cfg: config.Activations, preprocess):
        import datasets

        assert isinstance(cfg.data, config.ImagenetDataset)

        self.hf_dataset = datasets.load_dataset(
            cfg.data.name, split=cfg.data.split, trust_remote_code=True
        )

        self.preprocess = preprocess

    def __getitem__(self, i):
        example = self.hf_dataset[i]
        example["index"] = i
        example["image"] = self.preprocess(example["image"].convert("RGB"))
        return example

    def __len__(self) -> int:
        return len(self.hf_dataset)


@beartype.beartype
def get_imagenet_dataloader(
    cfg: config.Activations, preprocess
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for Imagenet loaded from Huggingface.

    Args:
        cfg: Config.
        preprocess: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches, `'index'` keys containing original dataset indices and `'label'` keys containing label batches.
    """
    assert isinstance(cfg.data, config.ImagenetDataset)

    dataloader = torch.utils.data.DataLoader(
        dataset=PreprocessedImageNet(cfg, preprocess),
        batch_size=cfg.vit_batch_size,
        drop_last=False,
        num_workers=cfg.n_workers,
        persistent_workers=cfg.n_workers > 0,
        shuffle=False,
        pin_memory=False,
    )
    return dataloader


class TransformedImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Args:
            index: Index

        Returns:
            dict with keys 'image', 'index', 'target' and 'label'.
        """
        breakpoint()
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"image": sample, "target": target, "index": index}


@beartype.beartype
def get_imagefolder_dataloader(
    cfg: config.Activations, preprocess
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for an ImageFolder.

    Args:
        cfg: Config.
        preprocess: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches, `'index'` keys containing original dataset indices and `'label'` keys containing label batches.
    """
    assert isinstance(cfg.data, config.ImageFolderDataset)

    dataset = TransformedImageFolder(cfg.data.root, preprocess)
    breakpoint()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.vit_batch_size,
        drop_last=False,
        num_workers=cfg.n_workers,
        persistent_workers=cfg.n_workers > 0,
        shuffle=False,
        pin_memory=True,
    )

    return dataloader


@beartype.beartype
class PreprocessedBroden(torch.utils.data.Dataset):
    def __init__(self, cfg: config.BrodenDataset, transform):
        import csv

        self.cfg = cfg
        self.transform = transform

        self.samples = []

        with open(os.path.join(cfg.root, "index.csv")) as fd:
            for row in csv.DictReader(fd):
                self.samples.append(row["image"])

    def __getitem__(self, i):
        fpath = os.path.join(self.cfg.root, "images", self.samples[i])
        with open(fpath, "rb") as fd:
            img = Image.open(fd).convert("RGB")
        img = self.transform(img)
        return {"image": img, "index": i}

    def __len__(self) -> int:
        return len(self.samples)


@beartype.beartype
def get_broden_dataloader(
    cfg: config.Activations, preprocess
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for Broden dataset.

    Args:
        cfg: Config.
        preprocess: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches and `'index'` keys containing original dataset indices.
    """
    assert isinstance(cfg.data, config.BrodenDataset)

    dataset = PreprocessedBroden(cfg.data, preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.vit_batch_size,
        drop_last=False,
        num_workers=cfg.n_workers,
        persistent_workers=cfg.n_workers > 0,
        shuffle=False,
        pin_memory=True,
    )

    return dataloader


@beartype.beartype
def dump(cfg: config.Activations):
    """
    Args:
        cfg: Config for activations.
    """
    logger = logging.getLogger("dump")

    if not cfg.ssl:
        logger.warning("Ignoring SSL certs. Try not to do this!")
        # https://github.com/openai/whisper/discussions/734#discussioncomment-4491761
        # Ideally we don't have to disable SSL but we are only downloading weights.
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    # Run any setup steps.
    setup(cfg)

    # Actually record activations.
    if cfg.slurm:
        import submitit

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=24 * 60,
            partition="gpu",
            gpus_per_node=1,
            cpus_per_task=cfg.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )

        job = executor.submit(worker_fn, cfg)
        logger.info("Running job '%s'.", job.job_id)
        job.result()

    else:
        worker_fn(cfg)


@beartype.beartype
def worker_fn(cfg: config.Activations):
    """
    Args:
        cfg: Config for activations.
    """

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    logger = logging.getLogger("dump")

    vit = make_vit(cfg)
    dataloader = get_dataloader(cfg, vit.make_img_transform())

    writer = ShardWriter(cfg)

    n_batches = cfg.data.n_imgs // cfg.vit_batch_size + 1
    logger.info("Dumping %d batches of %d examples.", n_batches, cfg.vit_batch_size)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        cfg = dataclasses.replace(cfg, device="cpu")

    vit = vit.to(cfg.device)
    # vit = torch.compile(vit)

    i = 0
    # Calculate and write ViT activations.
    with torch.inference_mode():
        for batch in helpers.progress(dataloader, total=n_batches):
            images = batch.pop("image").to(cfg.device)
            # cache has shape [batch size, n layers, n patches + 1, d vit]
            out, cache = vit(images)
            del out

            writer[i : i + len(cache)] = cache
            i += len(cache)

    writer.flush()


@beartype.beartype
class ShardWriter:
    """
    ShardWriter is a stateful object that handles sharded activation writing to disk.
    """

    root: str
    shape: tuple[int, int, int, int]
    shard: int
    acts_path: str
    acts: Float[np.ndarray, "n_imgs_per_shard n_layers all_patches d_vit"] | None
    filled: int

    def __init__(self, cfg: config.Activations):
        self.logger = logging.getLogger("shard-writer")

        self.root = get_acts_dir(cfg)

        n_patches_per_img = cfg.n_patches_per_img
        if cfg.cls_token:
            n_patches_per_img += 1
        self.n_imgs_per_shard = (
            cfg.n_patches_per_shard // len(cfg.layers) // n_patches_per_img
        )
        self.shape = (
            self.n_imgs_per_shard,
            len(cfg.layers),
            n_patches_per_img,
            cfg.d_vit,
        )

        self.shard = -1
        self.acts = None
        self.next_shard()

    @jaxtyped(typechecker=beartype.beartype)
    def __setitem__(
        self, i: slice, val: Float[Tensor, "_ n_layers all_patches d_vit"]
    ) -> None:
        assert i.step is None
        a, b = i.start, i.stop
        assert len(val) == b - a

        offset = self.n_imgs_per_shard * self.shard

        if b >= offset + self.n_imgs_per_shard:
            # We have run out of space in this mmap'ed file. Let's fill it as much as we can.
            n_fit = offset + self.n_imgs_per_shard - a
            self.acts[a - offset : a - offset + n_fit] = val[:n_fit]
            self.filled = a - offset + n_fit

            self.next_shard()

            # Recursively call __setitem__ in case we need *another* shard
            self[a + n_fit : b] = val[n_fit:]
        else:
            msg = f"0 <= {a} - {offset} <= {offset} + {self.n_imgs_per_shard}"
            assert 0 <= a - offset <= offset + self.n_imgs_per_shard, msg
            msg = f"0 <= {b} - {offset} <= {offset} + {self.n_imgs_per_shard}"
            assert 0 <= b - offset <= offset + self.n_imgs_per_shard, msg
            self.acts[a - offset : b - offset] = val
            self.filled = b - offset

    def flush(self) -> None:
        if self.acts is not None:
            self.acts.flush()

        self.acts = None

    def next_shard(self) -> None:
        self.flush()

        self.shard += 1
        self._count = 0
        self.acts_path = os.path.join(self.root, f"acts{self.shard:06}.bin")
        self.acts = np.memmap(
            self.acts_path, mode="w+", dtype=np.float32, shape=self.shape
        )
        self.filled = 0

        self.logger.info("Opened shard '%s'.", self.acts_path)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Metadata:
    model_org: str
    model_ckpt: str
    layers: tuple[int, ...]
    n_patches_per_img: int
    cls_token: bool
    d_vit: int
    seed: int
    n_imgs: int
    n_patches_per_shard: int
    data: str

    @classmethod
    def from_cfg(cls, cfg: config.Activations) -> "Metadata":
        return cls(
            cfg.model_org,
            cfg.model_ckpt,
            tuple(cfg.layers),
            cfg.n_patches_per_img,
            cfg.cls_token,
            cfg.d_vit,
            cfg.seed,
            cfg.data.n_imgs,
            cfg.n_patches_per_shard,
            str(cfg.data),
        )

    @classmethod
    def load(cls, fpath) -> "Metadata":
        with open(fpath) as fd:
            dct = json.load(fd)
        dct["layers"] = tuple(dct.pop("layers"))
        return cls(**dct)

    def dump(self, fpath):
        with open(fpath, "w") as fd:
            json.dump(dataclasses.asdict(self), fd, indent=4)

    @property
    def hash(self) -> str:
        cfg_str = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()


@beartype.beartype
def get_acts_dir(cfg: config.Activations) -> str:
    """
    Return the activations filepath based on the relevant values of a config.
    Also saves a metadata.json file to that directory for human reference.

    Args:
        cfg: Config for experiment.

    Returns:
        Directory to where activations should be dumped/loaded from.
    """
    metadata = Metadata.from_cfg(cfg)

    acts_dir = os.path.join(cfg.dump_to, metadata.hash)
    os.makedirs(acts_dir, exist_ok=True)

    metadata.dump(os.path.join(acts_dir, "metadata.json"))

    return acts_dir
