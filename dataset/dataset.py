from typing import Dict, Optional

import albumentations as A
import numpy as np
import pandas as pd
import pillow_avif
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class landmarksDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int,
        coarse_mapping: Dict,
        fine_mapping: Dict,
        transform: Optional[None] = None,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super(landmarksDataset, self).__init__()
        self.dataframe = dataframe
        self.image_size = image_size
        self.coarse_mapping = coarse_mapping
        self.fine_mapping = fine_mapping
        self.transform = transform
        self.rev_coarse_mapping = {v: k for k, v in self.coarse_mapping.items()}
        self.rev_fine_mapping = {v: k for k, v in self.fine_mapping.items()}
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.dataframe["image_path"][idx]).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            stand = A.Compose(
                [
                    A.Normalize(mean=self.mean, std=self.std),
                    A.Resize(self.image_size, self.image_size),
                    ToTensorV2(),
                ]
            )
            image = stand(image=image)["image"]
        coarse_label = self.rev_coarse_mapping[self.dataframe["style"][idx]]
        fine_label = self.rev_fine_mapping[self.dataframe["landmark"][idx]]
        return {"image": image, "style": coarse_label, "landmark": fine_label}


if __name__ == "__main__":
    with open("data/style_map.yaml", "r") as stream:
        coarse_map = yaml.safe_load(stream)
    with open("data/landmark_map.yaml", "r") as stream:
        fine_map = yaml.safe_load(stream)
    dataset = landmarksDataset(
        pd.read_csv("data/dataset.csv"), 224, coarse_map, fine_map
    )
    print(dataset[0]["image"].shape)
    print(dataset[0]["style"])
    print(dataset[0]["landmark"])
