from typing import Dict, Optional

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset import landmarksDataset


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fold,
        batch_size,
        num_workers,
        image_size,
        dataframe: pd.DataFrame,
        coarse_mapping: Dict,
        fine_mapping: Dict,
        transform: bool = False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super(LitDataModule, self).__init__()
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataframe = dataframe
        self.coarse_mapping = coarse_mapping
        self.fine_mapping = fine_mapping
        self.transform = transform
        self.mean = mean
        self.std = std

        self.save_hyperparameters()

        if self.transform:
            self.train_transform = A.Compose(
                [
                    # A.SmallestMaxSize(max_size=1024),
                    A.HorizontalFlip(p=0.5),
                    # A.VerticalFlip(p=0.5),
                    A.Rotate(p=0.5),
                    A.RandomRotate90(p=0.5),
                    # A.RandomFog(p=0.5),
                    # A.RandomRain(p=0.5),
                    # A.RandomShadow(p=0.5),
                    # A.RandomSnow(p=0.5),
                    # A.RandomSunFlare(p=0.5),
                    # A.GaussNoise(p=0.5),
                    # A.ShiftScaleRotate(
                    #     shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                    # ),
                    # A.RGBShift(
                    #     r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5
                    # ),
                    # A.ColorJitter(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    # A.RandomResizedCrop(512, 512, p=0.5),
                    A.ZoomBlur(p=0.6),
                    A.Normalize(mean=self.mean, std=self.std),
                    A.Resize(self.image_size, self.image_size),
                    ToTensorV2(),
                ]
            )

            self.val_transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=1024),
                    A.Normalize(mean=self.mean, std=self.std),
                    A.Resize(self.image_size, self.image_size),
                    ToTensorV2(),
                ]
            )
        else:
            self.train_transform = None
            self.val_transform = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_df = self.dataframe[self.dataframe["fold"] != self.fold].reset_index()
            val_df = self.dataframe[self.dataframe["fold"] == self.fold].reset_index()
            self.train_dataset = landmarksDataset(
                train_df,
                self.image_size,
                self.coarse_mapping,
                self.fine_mapping,
                self.train_transform,
                mean=self.mean,
                std=self.std,
            )
            self.val_dataset = landmarksDataset(
                val_df,
                self.image_size,
                self.coarse_mapping,
                self.fine_mapping,
                self.val_transform,
                mean=self.mean,
                std=self.std,
            )
        if stage == "test" or stage is None:
            self.test_dataset = landmarksDataset(
                self.dataframe,
                self.image_size,
                self.coarse_mapping,
                self.fine_mapping,
                mean=self.mean,
                std=self.std,
            )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True, val=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(
        self, dataset: landmarksDataset, train: bool = False, val: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if train and val else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if train and val else False,
        )
