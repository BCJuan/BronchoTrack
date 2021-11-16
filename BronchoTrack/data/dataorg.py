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
    def __init__(self, origin_root, new_root, n_trajectories=75, clean=True):
        """[summary]

        Args:
            origin_root ([type]): [description]
            new_root ([type]): [description]
            split_size (int, optional): Size by which split the sequence.
                Defaults to 10.
        """
        self.origin_root = origin_root
        self.new_root = new_root
        self.n_trajectories = n_trajectories
        self.split_length = 876
        if clean:
            self._clean()

        os.makedirs(new_root, exist_ok=True)
        for subset in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.new_root, subset), exist_ok=True)

        self.lobes = ["bl", "br", "ul", "ur"]
        self.levels = [0]  # should be 1, 2, 3
        self.columns_use = ["patient", "pos_x", "pos_y", "pos_z",
                            "Rx", "Ry", "Rz",
                            "filename", "base_filename"]
        self.position_label_names = ["pos_x_dif", "pos_y_dif", "pos_z_dif"]
        self.rotation_label_names = ["Rx_dif", "Ry_dif", "Rz_dif"]

    def create_csvs(self):
        csvfiles = glob.glob(os.path.join(self.origin_root, "*CPAP.csv"))
        for csv in tqdm(csvfiles, total=len(csvfiles)):
            sample_df = pd.read_csv(csv)
            for lobe in self.lobes:
                lobe_df = sample_df[sample_df["lobe"] == lobe].copy().reset_index()
                lobe_df["isStatusChanged"] = lobe_df["base_filename"].shift(
                    1, fill_value=lobe_df["base_filename"].head(1)) != lobe_df["base_filename"]
                change_indexes = np.insert(
                    lobe_df.loc[lobe_df["isStatusChanged"], :].index.values, 0, 0)
                self._splitting(lobe_df, csv, lobe, change_indexes)

    def _splitting(self, level_lobe_df, csv, lobe, change_indexes):
        # trajectories_indexes_relative = np.random.randint(0, self.split_length, size=(self.n_trajectories, len(change_indexes)))
        trajectories_indexes_relative = np.random.randint(0, self.split_length, size=self.n_trajectories)
        for i, j in enumerate(trajectories_indexes_relative):
            trajectory_indexes = change_indexes + j
            seq_level_lobe_df = level_lobe_df.loc[
                trajectory_indexes, self.columns_use,
            ].copy().reset_index()

            extension_name = (
                os.path.splitext(os.path.basename(csv))[0]
                + "_"
                + lobe
                + "_"
                + str(i)
                + ".csv"
            )
            for column in self.columns_use[1:7]:
                seq_level_lobe_df[column + "_dif"] = seq_level_lobe_df[column].diff(1).bfill()         
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
        # self.image_statistis()

    def compute_label_statistics(self):
        csvfiles = glob.glob(os.path.join(self.new_root, "train", "*.csv"))
        scaler = StandardScaler()
        for csv in csvfiles:
            sample_df = pd.read_csv(csv)
            sample_df = sample_df.reindex(self.position_label_names + self.rotation_label_names, axis=1)
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
                            n_images += len(os.listdir(subfolder_path))
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
