import typing

import tyro

import saev


def activations(cfg: typing.Annotated[saev.Config, tyro.conf.arg(name="")]):
    """Save ViT activations for use later on."""
    import saev.modeling

    vit = saev.modeling.RecordedVit(cfg)
    saev.modeling.save_acts(cfg, vit)


def train(cfg: typing.Annotated[saev.Config, tyro.conf.arg(name="")]):
    import saev.training

    saev.training.main(cfg)


# @beartype.beartype
# def main(cmd: Cmd, /, cfg: saev.Config):
#     elif cmd == "analysis":
#         import saev.analysis

#         saev.analysis.main(cfg)
#     elif cmd == "webapp":
#         print("To view the webapp, run:")
#         print()
#         print("  marimo run webapp.py")
#     else:
#         typing.assert_never(cmd)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "activations": activations,
        "train": train,
    })
