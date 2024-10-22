import beartype
import typing
import torch

from .activations_store import ActivationsStore
from .config import Config
from .sparse_autoencoder import SparseAutoencoder
from . import vits


@beartype.beartype
class Session(typing.NamedTuple):
    vit: vits.RecordedVit
    sae: SparseAutoencoder
    acts_store: ActivationsStore

    @classmethod
    def from_cfg(cls, cfg: Config) -> "Session":
        vit = vits.RecordedVit.from_cfg(cfg)
        vit.eval()
        for parameter in vit.model.parameters():
            parameter.requires_grad_(False)
        vit.to(cfg.device)

        sae = SparseAutoencoder(cfg)
        acts_store = ActivationsStore(cfg, vit)

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
