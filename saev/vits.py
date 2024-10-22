import typing

import torch
from jaxtyping import Float, jaxtyped
import beartype
from torch import Tensor
import transformers  # CLIPModel, CLIPProcessor
import logging
from .config import Config


@jaxtyped(typechecker=beartype.beartype)
class RecordedVit(torch.nn.Module):
    n_layers: int
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._storage = None
        self._i = 0

        self.n_layers = 0
        for layer in model.vision_model.encoder.layers:
            layer.register_forward_hook(self.hook)
            self.n_layers += 1

        self.logger = logging.getLogger("recorder")
        self.model = model

    @classmethod
    def from_cfg(cls, cfg: Config) -> "RecordedVit":
        model = transformers.CLIPModel.from_pretrained(cfg.model_name)
        return cls(model)

    def forward(
        self, *args, **kwargs
    ) -> tuple[typing.Any, Float[Tensor, "batch n_layers 1 dim"]]:
        self.reset()
        output = self.model(*args, **kwargs)
        return output, self.activations

    def hook(self, module, args, output) -> None:
        # For some reason, output is a length-1 tuple.
        output = output[0]

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
