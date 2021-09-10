import glob
import os
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms, io


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

        self.position_mean = [0.0, 0.0, 0.0]
        self.position_std = [1.0, 1.0, 1.0]
        self.rotation_mean = [0.0, 0.0, 0.0]
        self.rotation_std = [1.0, 1.0, 1.0]
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (1.0, 1.0, 1.0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        dataframe = pd.read_csv(self.items[index])
        position_labels = dataframe.loc[:, self.position_label_names]
        rotation_labels = dataframe.loc[:, self.rotation_label_names]
        images_paths = [
            os.path.join(self.image_root, row["patient"], row["filename"])
            for _, row in dataframe.iterrows()
        ]
        images = torch.stack([self.test_image_transforms()(io.read_image(i).float()/255) for i in images_paths])
        return {
            "images": images,
            "pos_labels": position_labels,
            "rot_labels": rotation_labels,
        }

    def test_image_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.Normalize(self.image_mean, self.image_std),
            ]
        )

    def label_transforms(self):
        # Normalize
        pass

    def train_image_transforms(self):
        # ColorJitter
        # RandomAffine?? without rotations
        # GaussianBlur
        # RandomAdjustSharpness
        # RandomAutocontrast
        # RandomEqualize
        # merge them with RandomApply
        pass
