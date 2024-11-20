import typing

import tyro

from . import config


def dump_topk(cfg: typing.Annotated[config.Topk, tyro.conf.arg(name="")]):
    from .dump_topk import main

    main(cfg)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "dump-topk": dump_topk,
        "nothing": lambda: print("dummy."),
    })
