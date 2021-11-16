import glob
import os
from types import new_class
import pandas as pd
import numpy as np
import pathlib
import torch
from typing import Optional
from torch.utils import data
from torchvision import transforms
from PIL import Image
from pickle import load
import pytorch_lightning as pl


class BronchoDataset(data.Dataset):
    def __init__(
        self, root_folder, image_root, train=True, target_size=(256, 256), augment=False, length=15
    ):
        super().__init__()
        self.root_folder = root_folder
        self.image_root = image_root
        self.train = train
        self.target_size = target_size
        self.augment = augment
        self.length = length
        self.items = glob.glob(os.path.join(self.root_folder, "*.csv"))

        self.position_label_names = ["pos_x_dif", "pos_y_dif", "pos_z_dif"]
        self.rotation_label_names = ["Rx_dif", "Ry_dif", "Rz_dif"]
        parent_dir = pathlib.Path(__file__).parent.absolute()
        self.scaler = load(open(os.path.join(parent_dir, "scaler.pkl"), "rb"))
        stats = load(open(os.path.join(parent_dir, "statistics.pkl"), "rb"))
        self.image_mean, self.image_std = stats["mean"], stats["std"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        dataframe = pd.read_csv(self.items[index])
        if self.train:
            begin_index = np.random.randint(1, len(dataframe) - self.length)
            labels = dataframe.loc[(begin_index + 1):(begin_index + self.length), self.position_label_names + self.rotation_label_names].to_numpy()
            # includes lobe as seond path folder
            images_paths = [
                os.path.join(self.image_root, row["patient"].strip(), row["filename"].split("_")[-4], row["filename"])
                for _, row in dataframe.loc[begin_index:(begin_index + self.length), :].iterrows()]
        else:
            labels = dataframe.loc[1:, self.position_label_names + self.rotation_label_names].to_numpy()
            # includes lobe as seond path folder
            images_paths = [
                os.path.join(self.image_root, row["patient"].strip(), row["filename"].split("_")[-4], row["filename"])
                for _, row in dataframe.iterrows()]
        std_labels = self.scaler.transform(labels)
        # selects labels
        position_labels = std_labels[:, :len(self.position_label_names)]
        rotation_labels = std_labels[:, -len(self.rotation_label_names):]

        if self.train:
            if self.augment:
                t = self.train_image_transforms()
            else:
                t = self.test_image_transforms()
        else:
            t = self.test_image_transforms()
        images = torch.stack([t(Image.open(i)) for i in images_paths])
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
        return transforms.Compose(
            [
                TrainTransform(),
                transforms.ToTensor(),
                transforms.Resize(self.target_size),
                transforms.Normalize(self.image_mean, self.image_std),
            ]
        )


class TrainTransform(object):

    def __init__(self):
        self.cjitter = transforms.ColorJitter(brightness=.4, hue=.2, saturation=0.3, contrast=0.3)
        self.gblur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 6))
        self.posterizer = transforms.RandomPosterize(bits=6, p=0.4)
        self.solarizer = transforms.RandomSolarize(threshold=250, p=0.2)
        self.sharpness_adjuster = transforms.RandomAdjustSharpness(sharpness_factor=7, p=0.4)
        self.autocontraster = transforms.RandomAutocontrast(p=0.5)
        self.equalizer = transforms.RandomEqualize(p=0.5)

    def __call__(self, image):
        r = np.random.rand(1)
        if r > 0.5:
            image = self.cjitter(image)
        r = np.random.rand(1)
        if r > 0.5:
            image = self.gblur(image)
        image = self.posterizer(image)
        image = self.solarizer(image)
        image = self.sharpness_adjuster(image)
        image = self.autocontraster(image)
        image = self.equalizer(image)
        return image


class BronchoDataModule(pl.LightningDataModule):

    def __init__(self, root_folder, image_root, batch_size, target_size=(256, 256), augment=False):
        super().__init__()
        self.root_folder = root_folder
        self.image_root = image_root
        self.batch_size = batch_size
        self.target_size = target_size
        self.trainpath = os.path.join(root_folder, "train")
        self.testpath = os.path.join(root_folder, "test")
        self.valpath = os.path.join(root_folder, "val")
        self.batch_size = batch_size
        self.augment = augment

    def setup(self, stage: Optional[str] = None):
        self.train_set = BronchoDataset(self.trainpath, self.image_root, train=True, target_size=self.target_size, augment=self.augment)
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
