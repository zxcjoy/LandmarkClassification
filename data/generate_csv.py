import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATASET_ROOT_DIR = Path("data/Landmarks-v1_0")

dataframe = pd.DataFrame(
    [
        {
            "image_path": os.path.join(root, f),
            "style": os.path.join(root, f).split("/")[2],
            "landmark": os.path.join(root, f).split("/")[-2],
        }
        for root, _, files in os.walk(DATASET_ROOT_DIR)
        for f in files
        if f != ".DS_Store"
    ]
)

style_map = {idx: item for idx, item in enumerate(np.sort(dataframe["style"].unique()))}

landmark_map = {
    idx: item
    for idx, item in enumerate(
        np.sort(np.append(dataframe["landmark"].unique(), ["other"]))
    )
}

with open("data/style_map.yaml", "w") as f:
    yaml.dump(style_map, f)
with open("data/landmark_map.yaml", "w") as f:
    yaml.dump(landmark_map, f)

dataframe.to_csv("data/dataset.csv", index=False)
