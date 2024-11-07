import dataclasses
import logging
import tomllib
import typing

import tyro

import saev

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("main")


def activations(cfg: typing.Annotated[saev.ActivationsConfig, tyro.conf.arg(name="")]):
    """
    Save ViT activations for use later on.

    Args:
        cfg: Configuration for activations.
    """
    import saev.activations

    saev.activations.dump(cfg)


def sweep(cfg: typing.Annotated[saev.TrainConfig, tyro.conf.arg(name="")], sweep: str):
    """
    Run a grid search over a set of hyperparameters.

    Args:
        cfg: Baseline config for training an SAE.
        sweep: Path to .toml file defining the sweep parameters.
    """
    import submitit

    import saev.sweep
    import saev.training

    with open(sweep, "rb") as fd:
        dcts = list(saev.sweep.expand(tomllib.load(fd)))
    logger.info("Sweep has %d experiments.", len(dcts))

    sweep_cfgs, errs = [], []
    for d, dct in enumerate(dcts):
        try:
            sweep_cfgs.append(dataclasses.replace(cfg, **dct, seed=cfg.seed + d))
        except Exception as err:
            errs.append(str(err))

    if cfg.slurm:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=60,
            partition="preemptible",
            gpus_per_node=1,
            cpus_per_task=cfg.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    jobs = executor.map_array(saev.training.train, sweep_cfgs)
    for i, result in enumerate(submitit.helpers.as_completed(jobs)):
        exp_id = result.result()
        logger.info("Finished task %s (%d/%d)", exp_id, i + 1, len(jobs))


def train(cfg: typing.Annotated[saev.TrainConfig, tyro.conf.arg(name="")]):
    def fn():
        import saev.training

        saev.training.main(cfg)

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

    job = executor.submit(fn)
    job.result()


def evaluate(cfg: typing.Annotated[saev.EvaluateConfig, tyro.conf.arg(name="")]):
    def run_histograms():
        import saev.histograms

        return saev.histograms.evaluate(cfg.histograms)

    def run_imagenet1k():
        import saev.imagenet1k

        return saev.imagenet1k.evaluate(cfg.imagenet)

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

    jobs = []
    # jobs.append(executor.submit(run_histograms))
    jobs.append(executor.submit(run_imagenet1k))
    for job in jobs:
        job.result()


def webapp(cfg: typing.Annotated[saev.WebappConfig, tyro.conf.arg(name="")]):
    import saev.webapp

    saev.webapp.main(cfg)

    print()
    print("To view the webapp, run:")
    print()
    print("    uv run marimo edit webapp.py")
    print()


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "activations": activations,
        "sweep": sweep,
        "evaluate": evaluate,
        "webapp": webapp,
    })
    logger.info("Done.")
