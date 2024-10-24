import logging
import typing

import tyro

import saev

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("main")


def activations(
    cfg: typing.Annotated[saev.Config, tyro.conf.arg(name="")],
    disable_ssl: bool = False,
):
    """
    Save ViT activations for use later on.

    Args:
        cfg: Configuration for experiment.
        disable_ssl: Whether to ignore ssl when downloading models and such.
    """
    if disable_ssl:
        logger.warning("Ignoring SSL certs. Try not to do this!")
        # https://github.com/openai/whisper/discussions/734#discussioncomment-4491761
        # Ideally we don't have to disable SSL but we are only downloading weights.
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    import saev.modeling

    vit = saev.modeling.RecordedVit(cfg)
    logger.info("Loaded ViT '%s'.", cfg.model)
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
    logger.info("Done.")
