import glob
import os
import pandas as pd
import numpy as np
import pathlib
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import ToTensor
from pickle import load


class BronchoDataset(data.Dataset):
    def __init__(
        self, root_folder, image_root, train=True, augment=True, target_size=(256, 256)
    ):
        super().__init__()
        self.root_folder = root_folder
        self.image_root = image_root
        self.train = train
        self.augment = augment
        self.target_size = target_size

        self.items = glob.glob(os.path.join(self.root_folder, "*.csv"))

        self.position_label_names = ["shift_x", "shift_y", "shift_z"]
        self.rotation_label_names = ["qx", "qy", "qz"]
        parent_dir = pathlib.Path(__file__).parent.absolute()
        self.scaler = load(open(os.path.join(parent_dir, "scaler.pkl"), "rb"))
        stats = load(open(os.path.join(parent_dir, "statistics.pkl"), "rb"))
        self.image_mean, self.image_std = stats["mean"], stats["std"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        dataframe = pd.read_csv(self.items[index])
        labels = dataframe.loc[:, self.position_label_names + self.rotation_label_names].to_numpy()
        std_labels = self.scaler.transform(labels)
        position_labels = std_labels[:, :len(self.position_label_names)]
        rotation_labels = std_labels[:, len(self.position_label_names):]
        images_paths = [
            os.path.join(self.image_root, row["patient"], row["filename"])
            for _, row in dataframe.iterrows()
        ]
        images = torch.stack([self.test_image_transforms()(Image.open(i)) for i in images_paths])
        return {
            "images": images,
            "pos_labels": position_labels,
            "rot_labels": rotation_labels,
        }

    def test_image_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.target_size),
                transforms.Normalize(self.image_mean, self.image_std),
            ]
        )

    def train_image_transforms(self):
        # ColorJitter
        # RandomAffine?? without rotations
        # GaussianBlur
        # RandomAdjustSharpness
        # RandomAutocontrast
        # RandomEqualize
        # merge them with RandomApply
        pass
