import beartype
import torch

import saev.config

from . import config

n_classes = 151


@beartype.beartype
def main(cfg: config.Train):
    train_dataset = Dataset(cfg.train_acts, cfg.train_imgs)
    val_dataset = Dataset(cfg.val_acts, cfg.val_imgs)

    model = torch.nn.Linear(train_dataset.d_vit, n_classes)
    optim = torch.optim.AdamW(
        model.parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.n_epochs):
        model.train()
        for batch in train_dataloader:
            breakpoint()

        model.eval()
        for batch in val_dataloader:
            breakpoint()


@beartype.beartype
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, acts_cfg: saev.config.DataLoad, imgs_cfg: saev.config.Ade20kDataset
    ):
        breakpoint()

    @property
    def d_vit(self) -> int:
        breakpoint()
