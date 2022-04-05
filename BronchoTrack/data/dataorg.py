import abc
import os
import shutil
import pathlib
import glob
from numpy import random
from numpy.core.defchararray import index
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pickle import dump
from PIL import Image
from tqdm import tqdm


class BronchoOrganizer(abc.ABC):
    def __init__(self, origin_root, new_root, n_trajectories=75, clean=True, test_pacient="P18", only_val=False,
                 intra_patient=False, length=2, rotate_patient=False, save_indexes=False):
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
        self.test_pacient = test_pacient
        self.only_val = only_val
        self.intra_patient = intra_patient
        self.length = length
        self.rotate_patient = rotate_patient
        self.save_indexes = save_indexes
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
                change_indexes = lobe_df.loc[lobe_df["isStatusChanged"], :].index.values

                if self.intra_patient:
                    self._splitting_intra_patient(lobe_df, csv, lobe, change_indexes)
                else:
                    self._splitting_only_val(lobe_df, csv, lobe, change_indexes, lobe_df.loc[0, "patient"])

    def _splitting_only_val(self, level_lobe_df, csv, lobe, change_indexes, patient):
        if self.rotate_patient:
            trajectories_indexes_relative = np.load("indexes.npy")    
        else:
            trajectories_indexes_relative = np.random.randint(0, self.split_length, size=self.n_trajectories)
        if self.save_indexes:
            np.save("indexes.npy", trajectories_indexes_relative)
        if str(self.test_pacient) in patient:
            self._split_save(trajectories_indexes_relative, change_indexes, level_lobe_df, csv, lobe, "val")
            self._no_split_save(trajectories_indexes_relative, change_indexes, level_lobe_df, csv, lobe, "test")
        else:
            self._split_save(trajectories_indexes_relative, change_indexes, level_lobe_df, csv, lobe, "train")

    def _splitting_intra_patient(self, level_lobe_df, csv, lobe, change_indexes):
        trajectories_indexes_relative = np.random.randint(0, self.split_length, size=self.n_trajectories)
        # trajectories_indexes_relative = np.random.randint(0, self.split_length, size=(self.n_trajectories, len(change_indexes)))
        # index_to_0 = np.random.randint(0, 2, size=(self.n_trajectories, len(change_indexes)))
        # trajectories_indexes_relative = index_to_0*trajectories_indexes_relative
        trj_train, trj_val = train_test_split(trajectories_indexes_relative, test_size=0.2, random_state=42, shuffle=True)
        if self.only_val:
            trj_test = trj_val
        else:
            trj_val, trj_test = train_test_split(trj_val, test_size=0.5, random_state=42, shuffle=True)
        self._no_split_save(trj_test, change_indexes, level_lobe_df, csv, lobe, "test")
        self._split_save(trj_val, change_indexes, level_lobe_df, csv, lobe, "val")
        self._split_save(trj_train, change_indexes, level_lobe_df, csv, lobe, "train")    

    def _no_split_save(self, trajectory, change_indexes, level_lobe_df, csv, lobe, folder):
        for i, j in enumerate(trajectory):
            trajectory_indexes = change_indexes + j
            seq_level_lobe_df = level_lobe_df.loc[
                trajectory_indexes, self.columns_use,
            ].copy().reset_index()
            extension_name = (os.path.splitext(os.path.basename(csv))[0] + "_" + lobe + "_" + str(i) + ".csv")
            for column in self.columns_use[1:7]:
                seq_level_lobe_df[column + "_dif"] = seq_level_lobe_df[column].diff(1).bfill()
            destname = os.path.join(self.new_root, folder, extension_name)
            seq_level_lobe_df.to_csv(destname)

    def _split_save(self, trajectory, change_indexes, level_lobe_df, csv, lobe, folder):
        for i, j in enumerate(trajectory):
            trajectory_indexes = change_indexes + j
            for h, index in enumerate(range(self.length, len(trajectory_indexes), self.length - 1)):
                seq_level_lobe_df = level_lobe_df.loc[
                    trajectory_indexes[(index - self.length): index], self.columns_use,
                ].copy().reset_index()
                extension_name = (os.path.splitext(os.path.basename(csv))[0] + "_" + lobe + "_" + str(i) + "_" + str(h) + ".csv")
                for column in self.columns_use[1:7]:
                    seq_level_lobe_df[column + "_dif"] = seq_level_lobe_df[column].diff(1).bfill()
                destname = os.path.join(self.new_root, folder, extension_name)
                seq_level_lobe_df.to_csv(destname)

    def _clean(self):
        shutil.rmtree(os.path.join(self.new_root, "train"), ignore_errors=True)
        shutil.rmtree(os.path.join(self.new_root, "test"), ignore_errors=True)
        shutil.rmtree(os.path.join(self.new_root, "val"), ignore_errors=True)
