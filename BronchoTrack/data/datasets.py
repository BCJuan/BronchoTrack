import glob
import os
import pandas as pd
import numpy as np
import pathlib
import torch
from typing import Optional
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import ToTensor
from pickle import load
import pytorch_lightning as pl


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
        position_labels = std_labels[1:, :len(self.position_label_names)]
        rotation_labels = std_labels[1:, len(self.position_label_names):]
        images_paths = [
            os.path.join(self.image_root, row["patient"], row["filename"])
            for _, row in dataframe.iterrows()
        ]
        images = torch.stack([self.test_image_transforms()(Image.open(i)) for i in images_paths])
        return {
            "images": images,
            "pos_labels": torch.tensor(position_labels, dtype=torch.float32),
            "rot_labels": torch.tensor(rotation_labels, dtype=torch.float32)
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


class BronchoDataModule(pl.LightningDataModule):

    def __init__(self, root_folder, image_root, batch_size, target_size=(256, 256)):
        super().__init__()
        self.root_folder = root_folder
        self.image_root = image_root
        self.batch_size = batch_size
        self.target_size = target_size
        self.trainpath = os.path.join(root_folder, "train")
        self.testpath = os.path.join(root_folder, "test")
        self.valpath = os.path.join(root_folder, "val")
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_set = BronchoDataset(self.trainpath, self.image_root, train=True, target_size=self.target_size)
        self.test_set = BronchoDataset(self.testpath, self.image_root, train=False, target_size=self.target_size)
        self.val_set = BronchoDataset(self.valpath, self.image_root, train=False, target_size=self.target_size)

    def train_dataloader(self):
        mnist_train = data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=8)
        return mnist_train

    def test_dataloader(self):
        mnist_test = data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=8)
        return mnist_test

    def val_dataloader(self):
        mnist_val = data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=8)
        return mnist_val