import abc
import os
import shutil
import glob
import pandas as pd


class BronchoOrganizer(abc.ABC):
    def __init__(self, origin_root, new_root, split=True, split_size=10, clean=True):
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
        if clean:
            self._clean()

        os.makedirs(new_root, exist_ok=True)
        for subset in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.new_root, subset), exist_ok=True)

        self.lobes = ["bl", "br", "ul", "ur"]
        self.levels = [0, 1, 2, 3]
        self.columns_use = [
            "patient",
            "shift_x",
            "shift_y",
            "shift_z",
            "qx",
            "qy",
            "qz",
            "filename",
        ]

    def create_csvs(self):
        csvfiles = glob.glob(os.path.join(self.origin_root, "*.csv"))
        for csv in csvfiles:
            sample_df = pd.read_csv(csv)
            for lobe in self.lobes:
                lobe_df = sample_df[sample_df["lobe"] == lobe].copy().reset_index()
                for level in self.levels:
                    level_lobe_df = (
                        lobe_df[(lobe_df["level"] == level)].copy().reset_index()
                    )
                    if self.split:
                        self._splitting(level_lobe_df, csv, lobe, level)
                    else:
                        self._no_splitting(level_lobe_df, csv, lobe, level)

    def _splitting(self, level_lobe_df, csv, lobe, level):
        len_split = len(level_lobe_df) // self.split_size
        for i in range(len_split):
            seq_level_lobe_df = level_lobe_df.loc[
                (i * self.split_size) : ((i + 1) * self.split_size), self.columns_use,
            ]
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
            if level == 3 and i == 3:
                destname = os.path.join(self.new_root, "test", extension_name)
            elif level == 3 and i == 2:
                destname = os.path.join(self.new_root, "val", extension_name)
            else:
                destname = os.path.join(self.new_root, "train", extension_name)
            seq_level_lobe_df.to_csv(destname)

    def _no_splitting(self, level_lobe_df, csv, lobe, level):
        seq_level_lobe_df = level_lobe_df.loc[:, self.columns_use].copy()
        extension_name = (
            os.path.splitext(os.path.basename(csv))[0]
            + "_"
            + lobe
            + "_"
            + str(level)
            + ".csv"
        )
        if level == 3 and lobe == "ul":
            destname = os.path.join(self.new_root, "test", extension_name)
        elif level == 2 and lobe == "ul":
            destname = os.path.join(self.new_root, "val", extension_name)
        else:
            destname = os.path.join(self.new_root, "train", extension_name)
        seq_level_lobe_df.to_csv(destname)

    def _clean(self):
        shutil.rmtree(self.new_root)
