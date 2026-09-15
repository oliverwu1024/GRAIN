"""Training entrypoint.

    python train.py -c config.example.yaml

Constant Adam(1e-4), batch 1024, float64 throughout, no schedule, clipping or
mixed precision. Every window of every prior series is emitted in file order
through a 5000-element shuffle buffer, and the epoch is declared as a fixed
`steps_per_epoch` over that endless stream.

There is no benchmark evaluation and no checkpoint gate: one checkpoint is
overwritten after every epoch, unconditionally.
"""

import argparse
import datetime
import os
from typing import Any, Dict

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.fabric.plugins.environments import LightningEnvironment
import numpy as np
import torch
import yaml
from gluonts.env import env

from constants import (
    BATCH_SIZE,
    FLOAT_DTYPE,
    LEARNING_RATE,
    MAX_HISTORY,
    QUANTILE_LEVELS,
    SEED,
    SHUFFLE_BUFFER,
    STEPS_PER_EPOCH,
    TARGET_LEN,
)
from dataset import PriorDataset
from estimator import GRAINEstimator


def srun_launched() -> bool:
    """True only for a real multi-task `srun` launch, where one task is one rank."""
    ntasks = str(os.environ.get("SLURM_NTASKS_PER_NODE", "")).split("(")[0]
    return ntasks.isdigit() and int(ntasks) > 1


def resolve_num_devices(cfg: Dict[str, Any]) -> int:
    """Turn the `devices` config value into a concrete count.

    Under srun this must match --ntasks-per-node, since each task is one rank.
    A mismatch hangs, or trains on the wrong number of shards.
    """
    devices = cfg.get("devices", 1)
    if isinstance(devices, int) and devices > 0:
        num = devices
    else:
        available = torch.cuda.device_count()
        num = available if available > 0 else 1

    ntasks = os.environ.get("SLURM_NTASKS_PER_NODE")
    if ntasks:
        ntasks_n = int(str(ntasks).split("(")[0])
        if ntasks_n > 1 and ntasks_n != num:
            raise ValueError(
                f"devices={num} in the config but SLURM --ntasks-per-node="
                f"{ntasks_n}. Under srun each task is one rank, so these must "
                f"match. Fix the config or the sbatch directive."
            )
    return num


def resolve_run_dir(cfg: Dict[str, Any]) -> str:
    """One run directory shared by every rank.

    Lightning relaunches this script per rank under DDP, and under srun every rank
    starts at once, so a per-process timestamp would give each its own directory.
    The job id is the same across tasks, so prefer it.
    """
    name = cfg.get("model_save_name", "grain_yearly")
    run_id = os.environ.get("GRAIN_RUN_ID")
    if run_id is None:
        job = os.environ.get("SLURM_JOB_ID")
        stamp = (
            f"slurm{job}" if job
            else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        run_id = f"{name}.{stamp}"
        os.environ["GRAIN_RUN_ID"] = run_id
    out_dir = os.path.join(cfg.get("output_dir", "runs"), run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    torch.set_default_dtype(FLOAT_DTYPE)
    seed = cfg.get("seed", SEED)
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)

    prior = PriorDataset(cfg["prior_values"], cfg["prior_meta"])
    out_dir = resolve_run_dir(cfg)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    print(f"prior: {len(prior):,} series")
    print(f"run:   {out_dir}")

    # `batch_size` in the config is the global batch. Lightning DDP feeds it to
    # each rank, so divide it here; otherwise asking for 4 GPUs would quadruple
    # the effective batch instead of just running faster.
    num_devices = resolve_num_devices(cfg)
    global_batch = cfg.get("batch_size", BATCH_SIZE)
    if global_batch % num_devices:
        raise ValueError(
            f"batch_size {global_batch} is not divisible by devices {num_devices}; "
            f"pick a global batch that splits evenly across ranks"
        )
    per_rank_batch = global_batch // num_devices
    print(f"devices: {num_devices}   global batch: {global_batch}   "
          f"per-rank batch: {per_rank_batch}")

    steps = cfg.get("steps_per_epoch", STEPS_PER_EPOCH)
    epochs = cfg.get("epochs", 8000)
    print(f"steps/epoch: {steps}   epochs: {epochs}   "
          f"checkpoint: {os.path.join(ckpt_dir, 'model.ckpt')} (rewritten each epoch)")

    estimator = GRAINEstimator(
        prediction_length=cfg.get("prediction_length", TARGET_LEN),
        max_prediction_length=cfg.get("max_prediction_length", TARGET_LEN),
        context_length=cfg.get("context_length", MAX_HISTORY),
        lr=cfg.get("learning_rate", LEARNING_RATE),
        batch_size=per_rank_batch,
        num_batches_per_epoch=steps,
        shuffle_buffer=cfg.get("shuffle_buffer", SHUFFLE_BUFFER),
        model_kwargs={
            "quantile_levels": tuple(cfg.get("quantile_levels", QUANTILE_LEVELS)),
            "horizon_weights": cfg.get("horizon_weights"),
        },
    )

    # Drive the trainer directly rather than through `estimator.train()`, which
    # installs its own best-only ModelCheckpoint on train_loss and reloads from it
    # at the end. Here nothing is selected on a metric.
    transformation = estimator.create_transformation()
    module = estimator.create_lightning_module()
    with env._let(max_idle_transforms=max(len(prior), 100)):
        loader = estimator.create_training_data_loader(
            transformation.apply(prior, is_train=True), module
        )

    # One file, overwritten every epoch. monitor=None means "keep the latest",
    # not "keep the best".
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="model",
        every_n_epochs=1,
        save_top_k=1,
        monitor=None,
    )
    trainer_kwargs = dict(
        max_epochs=epochs,
        accelerator=cfg.get("accelerator", "auto"),
        devices=num_devices,
        strategy=cfg.get("strategy", "auto"),
        default_root_dir=out_dir,
        enable_progress_bar=cfg.get("progress_bar", True),
        gradient_clip_val=cfg.get("gradient_clip_val", None),
        callbacks=[checkpoint],
        logger=CSVLogger(out_dir, name="", version="", flush_logs_every_n_steps=steps),
    )
    if not srun_launched():
        # Lightning treats any set SLURM_NTASKS as a cluster launch, including
        # inside an interactive allocation, where it then rejects `--ntasks=N`
        # and demands `--ntasks-per-node`. That check only makes sense for a real
        # multi-task srun launch.
        trainer_kwargs["plugins"] = [LightningEnvironment()]

    pl.Trainer(**trainer_kwargs).fit(model=module, train_dataloaders=loader)
    print("done")


if __name__ == "__main__":
    main()
