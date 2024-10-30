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

    import submitit

    if cfg.slurm:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=30 * 60,  # 30 hours
            partition="gpu",
            gpus_per_node=1,
            cpus_per_task=16,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    def fn(cfg: saev.Config):
        import saev.modeling

        saev.modeling.dump_acts(cfg, saev.modeling.RecordedVit(cfg))

    job = executor.submit(fn, cfg)
    job.result()


def train(cfg: typing.Annotated[saev.Config, tyro.conf.arg(name="")]):
    import submitit

    if cfg.slurm:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=30,
            partition="debug",
            gpus_per_node=1,
            cpus_per_task=12,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    def fn(cfg: saev.Config):
        import saev.training

        saev.training.main(cfg)

    job = executor.submit(fn, cfg)
    job.result()


def analysis(
    cfg: typing.Annotated[saev.Config, tyro.conf.arg(name="")],
    run_id: str,
    top_k: int = 16,
    root: str = "data",
):
    import submitit

    if cfg.slurm:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=30,
            partition="debug",
            gpus_per_node=1,
            cpus_per_task=12,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    def fn():
        import saev.analysis

        saev.analysis.main(cfg, run_id, top_k=top_k, root=root)

    job = executor.submit(fn)
    job.result()


def webapp(
    cfg: typing.Annotated[saev.Config, tyro.conf.arg(name="")],
    load_from: str,
    dump_to: str = "web",
):
    import saev.webapp

    saev.webapp.main(cfg, load_from, dump_to)

    print()
    print("To view the webapp, run:")
    print()
    print("    marimo run webapp.py")
    print()


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "activations": activations,
        "train": train,
        "analysis": analysis,
        "webapp": webapp,
    })
    logger.info("Done.")
