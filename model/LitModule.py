from typing import Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchmetrics import F1Score
from model import EffNet_B0


class LitModule(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
        drop_rate=0.0,
        coarse_num_classes=6,
        fine_num_classes=30,
        learning_rate=1e-3,
        weight_decay=1e-5,
        step_size=10,
        gamma=0.5,
    ):
        super(LitModule, self).__init__()
        self.pretrained = pretrained
        self.drop_rate = drop_rate
        self.coarse_num_classes = coarse_num_classes
        self.fine_num_classes = fine_num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        self.model = EffNet_B0(
            self.coarse_num_classes,
            self.fine_num_classes,
            pretrained=self.pretrained,
            dropout=self.drop_rate,
        )

        self.coarse_loss_fn = nn.CrossEntropyLoss()
        self.fine_loss_fn = nn.CrossEntropyLoss()

        self.coarse_f1_fn = F1Score(
            task="multiclass", num_classes=self.coarse_num_classes, average="macro"
        )
        self.fine_f1_fn = F1Score(
            task="multiclass", num_classes=self.fine_num_classes, average="macro"
        )

        self.save_hyperparameters()

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(X["image"])

    def configure_optimizers(self):
        # Setup the optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        style, landmark = self(batch)

        coarse_loss = self.coarse_loss_fn(style, batch["style"])
        fine_loss = self.fine_loss_fn(landmark, batch["landmark"])
        self.log(f"{step}_coarse_loss", coarse_loss, sync_dist=True)
        self.log(f"{step}_fine_loss", fine_loss, sync_dist=True)

        loss = 0.5 * coarse_loss + 0.5 * fine_loss
        self.log(f"{step}_loss", loss, sync_dist=True, prog_bar=True)

        coarse_f1 = self.coarse_f1_fn(nn.Softmax(1)(style), batch["style"])
        fine_f1 = self.fine_f1_fn(nn.Softmax(1)(landmark), batch["landmark"])
        self.log(f"{step}_coarse_f1", coarse_f1, sync_dist=True)
        self.log(f"{step}_fine_f1", fine_f1, sync_dist=True)

        f1 = 0.5 * coarse_f1 + 0.5 * fine_f1
        self.log(f"{step}_f1", f1, sync_dist=True, prog_bar=True)

        return loss
