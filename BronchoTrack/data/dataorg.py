import abc
import os
import shutil
import pathlib
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pickle import dump
from PIL import Image
from tqdm import tqdm


class BronchoOrganizer(abc.ABC):
    def __init__(self, origin_root, new_root, split=True, split_size=10, clean=True, cutdown=None):
        """[summary]

        Args:
            origin_root ([type]): [description]
            new_root ([type]): [description]
            split_size (int, optional): Size by which split the sequence.
                Defaults to 10.
        """
        self.origin_root = origin_root
        self.new_root = new_root
        self.split_size = split_size
        self.split = split
        self.cutdown = cutdown
        if clean:
            self._clean()

        os.makedirs(new_root, exist_ok=True)
        for subset in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.new_root, subset), exist_ok=True)

        self.lobes = ["bl", "br", "ul", "ur"]
        self.levels = [0]  # should be 1, 2, 3
        self.columns_use = ["patient", "shift_x", "shift_y", "shift_z",
                            "Rx", "Ry", "Rz", "Rx_base", "Ry_base", "Rz_base",
                            "filename", "base_filename"]

    def create_csvs(self):
        csvfiles = glob.glob(os.path.join(self.origin_root, "*CPAP.csv"))
        for csv in tqdm(csvfiles, total=len(csvfiles)):
            sample_df = pd.read_csv(csv)
            for lobe in self.lobes:
                lobe_df = sample_df[sample_df["lobe"] == lobe].copy().reset_index()
                lobe_df["isStatusChanged"] = lobe_df["base_filename"].shift(
                    1, fill_value=lobe_df["base_filename"].head(1)) != lobe_df["base_filename"]
                change_indexes = lobe_df.loc[lobe_df["isStatusChanged"], :].index
                previous_index = 0
                for j, index in enumerate(change_indexes):
                    level_lobe_df = lobe_df.iloc[previous_index:index, :].copy().reset_index()  
                    if self.split:
                        self._splitting(level_lobe_df, csv, lobe, j)
                    else:
                        self._no_splitting(level_lobe_df, csv, lobe, j)

    def _splitting(self, level_lobe_df, csv, lobe, level):
        if self.cutdown:
            len_split = round(len(level_lobe_df)*self.cutdown) // self.split_size
        else:
            len_split = len(level_lobe_df) // self.split_size
        
        for i in range(len_split):
            seq_level_lobe_df = level_lobe_df.loc[
                (i * self.split_size):((i + 1) * self.split_size), self.columns_use,
            ].copy().reset_index()

            if len(seq_level_lobe_df) > self.split_size:
                extension_name = (
                    os.path.splitext(os.path.basename(csv))[0]
                    + "_"
                    + lobe
                    + "_"
                    + str(level)
                    + "_"
                    + str(i)
                    + ".csv"
                )

                if seq_level_lobe_df.loc[0, "patient"] != "LENS_P18_14_01_2016_INSP_CPAP":
                    destname = os.path.join(self.new_root, "train", extension_name)
                else:
                    if lobe == "ur" or lobe == "bl":
                        destname = os.path.join(self.new_root, "val", extension_name)
                    else:
                        destname = os.path.join(self.new_root, "test", extension_name)
                seq_level_lobe_df.to_csv(destname)

    def _no_splitting(self, level_lobe_df, csv, lobe, level):
        seq_level_lobe_df = level_lobe_df.loc[:, self.columns_use].copy().reset_index()
        extension_name = (
            os.path.splitext(os.path.basename(csv))[0]
            + "_"
            + lobe
            + "_"
            + str(level)
            + ".csv"
        )
        if seq_level_lobe_df.loc[0, "patient"] != "LENS_P18_14_01_2016_INSP_CPAP":
            destname = os.path.join(self.new_root, "train", extension_name)
        else:
            if lobe == "ur" or lobe == "bl":
                destname = os.path.join(self.new_root, "val", extension_name)
            else:
                destname = os.path.join(self.new_root, "test", extension_name)
        seq_level_lobe_df.to_csv(destname)

    def _clean(self):
        shutil.rmtree(self.new_root)

    def compute_statistics(self):
        self.compute_label_statistics()
        self.image_statistis()

    def compute_label_statistics(self):
        csvfiles = glob.glob(os.path.join(self.origin_root, "*CPAP.csv"))
        scaler = StandardScaler()
        for csv in csvfiles:
            if os.path.samefile(csv, os.path.join(self.origin_root, "LENS_P18_14_01_2016_INSP_CPAP.csv")):
                sample_df = pd.read_csv(csv).reindex(self.columns_use[1:-2], axis=1)
                for new_column in self.columns_use[4:7]:
                    sample_df[new_column + "_dif"] = \
                        sample_df.apply(lambda x: compute_angle_difference(
                            x[new_column], x[new_column + "_base"]), axis=1)
                sample_df = sample_df.reindex(self.columns_use[1:4] + self.columns_use[-5:-2], axis=1)
                scaler.partial_fit(sample_df.to_numpy())
        parent_dir = pathlib.Path(__file__).parent.absolute()
        dump(scaler, open(os.path.join(parent_dir, "scaler.pkl"), "wb"))

    def image_statistis(self):
        n_images = 0
        for folder in os.listdir(self.origin_root):
            if folder != "LENS_P18_14_01_2016_INSP_CPAP":
                folderpath = os.path.join(self.origin_root, folder)
                if os.path.isdir(folderpath):
                    for subfolder in os.listdir(folderpath):
                        subfolder_path = os.path.join(folderpath, subfolder)
                        if os.path.isdir(subfolder_path):
                            print(subfolder_path)
                            n_images += len(os.listdir(subfolder_path))
        print(n_images)
        means = []
        for folder in tqdm(os.listdir(self.origin_root), total=len(os.listdir(self.origin_root))):
            if folder != "LENS_P18_14_01_2016_INSP_CPAP":
                folderpath = os.path.join(self.origin_root, folder)
                if os.path.isdir(folderpath):
                    for subfolder in os.listdir(folderpath):
                        subfolder_path = os.path.join(folderpath, subfolder)
                        if os.path.isdir(subfolder_path):
                            for image in os.listdir(subfolder_path):
                                imagepath = os.path.join(subfolder_path, image)
                                im = Image.open(imagepath)
                                means.append((np.sum(im, axis=(0, 1))/(im.size[0]*im.size[1]))/n_images)
        avg_mean = np.sum(np.stack(means), axis=0)
        stds = []
        for folder in tqdm(os.listdir(self.origin_root), total=len(os.listdir(self.origin_root))):
            if folder != "LENS_P18_14_01_2016_INSP_CPAP":
                folderpath = os.path.join(self.origin_root, folder)
                if os.path.isdir(folderpath):
                    for subfolder in os.listdir(folderpath):
                        subfolder_path = os.path.join(folderpath, subfolder)
                        if os.path.isdir(subfolder_path):
                            for image in os.listdir(subfolder_path):
                                imagepath = os.path.join(subfolder_path, image)
                                im = (Image.open(imagepath) - avg_mean)**2
                                stds.append((np.sum(im, axis=(0, 1))/(im.shape[0]*im.shape[1]))/n_images)
        avg_std = np.sqrt(np.sum(np.stack(stds), axis=0))
        statistics_file_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "statistics.pkl"
        )
        stats = {"mean": avg_mean, "std": avg_std}
        dump(stats, open(statistics_file_path, "wb"))


def compute_angle_difference(new, base):
    if new > 0:
        new = -new
    return new - base
