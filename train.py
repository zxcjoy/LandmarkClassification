import argparse
import os

import pandas as pd
import pytorch_lightning as pl
import yaml
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import *
from sklearn.model_selection import StratifiedKFold

from dataset import LitDataModule
from model import LitModule



def train(
    cfg,
    fold,
    dataframe,
):
    pl.seed_everything(cfg.seed)

    with open(os.path.join(cfg.data_root, "style_map.yaml"), "r") as f:
        coarse_mapping = yaml.safe_load(f)
    with open(os.path.join(cfg.data_root, "landmark_map.yaml"), "r") as f:
        fine_mapping = yaml.safe_load(f)

    datamodule = LitDataModule(
        fold,
        cfg.batch_size,
        cfg.num_workers,
        cfg.image_size,
        dataframe,
        coarse_mapping,
        fine_mapping,
        cfg.transform,
        tuple(cfg.mean),
        tuple(cfg.std),
    )
    datamodule.setup()

    module = LitModule(
        pretrained=cfg.pretrained,
        drop_rate=cfg.drop_rate,
        coarse_num_classes=len(coarse_mapping),
        fine_num_classes=len(fine_mapping),
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        step_size=cfg.step_size,
        gamma=cfg.gamma,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        monitor="val_f1",
        mode="max",
        verbose="True",
    )

    trainer = pl.Trainer(
        callbacks=[model_checkpoint],
        benchmark=True,
        deterministic=True,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.epoch,
        precision=cfg.precision,
        log_every_n_steps=1,
        logger=WandbLogger("lightning_logs", "effnet-b0", project = "effnet-b0"),
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scripts.")
    parser.add_argument("-cfg", "--config", default="config/train.yaml", type=str)
    parser.add_argument("-csv", "--csv-file", default="data/dataset.csv", type=str)
    parser.add_argument(
        "-ckpt_dir", "--checkpoint-dir", default="checkpoints", type=str
    )
    parser.add_argument("-ckpt_file", "--checkpoint-file", type=str)

    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.create(vars(args)))

    dataframe = pd.read_csv("data/dataset.csv")

    for fold, (train_idx, valid_idx) in enumerate(
        StratifiedKFold(n_splits=cfg.K_fold, random_state=cfg.seed, shuffle=True).split(
            X=dataframe, y=dataframe["style"].values
        )
    ):
        dataframe.loc[valid_idx, "fold"] = fold

    for i in range(cfg.K_fold):
        train(cfg, i, dataframe)
