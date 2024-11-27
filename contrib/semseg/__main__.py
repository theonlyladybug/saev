import typing

import beartype
import tyro

from . import config


@beartype.beartype
def train(cfg: typing.Annotated[config.Train, tyro.conf.arg(name="")]):
    from . import training

    training.main(cfg)


@beartype.beartype
def visuals():
    print("Not implemented.")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "train": train,
        "visuals": visuals,
    })
