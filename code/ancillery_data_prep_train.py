import os
import pathlib
import gc
import json
import typing
import argparse
import itertools
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torch.utils
import torch.utils.data

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, filepath) -> None:
        self._filepath = pathlib.Path(file_path)
        self._cols = ["patient_id", "study_id", "patient_age", "psa", "psad", "prostate_volume", "case_ISUP"]
        self._dataframe = load_data(self._filepath)

    def __len__(self):
        return len(self._dataframe)
    
    def get_dataframe(self):
        return self._dataframe

    def __getitem__(self, idx):
        sample = {'data': self._dataframe[self._cols[2:-1]].iloc[idx], 'target': self._dataframe[self._cols[-1]].iloc[idx]}
        return sample
    
class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices) -> None:
        super(torch.utils.data.Dataset, Subset).__init__()
        self._dataset = dataset
        self._indices = indices
        self._subset = self._dataset.loc[indices]

    def __len__(self) -> int:
        raise len(self._subset)
    
    def __getitem__(self, idx) -> typing.Any:
        sample = {'data': self._subset.iloc[idx], 'target': self._subset.iloc[idx]}
        return sample

class MLP(nn.Module):
    "Small multilayer perceptron model."
    def __init__(self, in_dim:int, out_dim:int, hidden_layer_sizes:iter) -> None:
        super(MLP, nn.Module).__init__()
        self.model = nn.Sequential(
            [
                ("in_layer", nn.Linear(in_dim, hidden_layer_sizes[0])),
                nn.ReLU()
            ]
        )
        if len(hidden_layer_sizes) > 1:
            for i, (inp, out) in enumerate(itertools.pairwise(hidden_layer_sizes)):
                self.model.add_module(f"hl{i+1}", nn.Linear(inp, out))
                self.model.add_module(f"hl{i+1}_act", nn.ReLU())
        self.model.add_module("out_layer", nn.Linear(hidden_layer_sizes(-1), out_dim))

    def forward(self, x):
        return self.model(x)

class Trainer:
    def __init__(self) -> None:
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
    
    def fit(self):
        raise NotImplementedError()


def load_tabular(path:pathlib.Path) -> typing.Union[pd.DataFrame|pd.Series]:
    assert path.suffix == ".csv"
    df = pd.read_csv(path)
    return df

def select_feature_columns(df:typing.Union[pd.DataFrame|pd.Series]) -> typing.Union[pd.DataFrame|pd.Series]:
    cols = ["patient_id", "study_id", "patient_age", "psa", "psad", "prostate_volume", "case_ISUP"]
    id_features_df = df[cols]
    del df
    gc.collect()
    return id_features_df

def calculate_psad(row):
    if pd.isna(row.psad):
        value = np.round(row.psa/row.prostate_volume, decimals=2)
    else:
        value = row.psad
    return value

def fill_psad(df:typing.Union[pd.DataFrame|pd.Series]) -> None:
    df.dropna(axis=0, how="any", subset=["psa", "prostate_volume"], inplace=True)
    df["psad"] = df.apply(calculate_psad, axis=1)

def load_data(path:pathlib.Path) -> typing.Union[pd.DataFrame|pd.Series]:
    df = load_tabular(path)
    df = select_feature_columns(df)
    fill_psad(df)
    return df

def get_splits(splits_path: pathlib.Path) -> list[dict]:
    with open(splits_path, mode="r") as sf:
        splits = json.load(sf)
    return splits

def find_folds(df: typing.Union[pd.DataFrame|pd.Series], splits:list[dict]):
    train_fold_dict = {}
    val_fold_dict = {}
    for i, split in enumerate(splits):
        train_data = split["train"]
        val_data = split["val"]
        train_mask_values = [
            (int(value.split("_")[0]), int(value.split("_")[1])) for value in train_data
        ]
        val_mask_values = [
            (int(value.split("_")[0]), int(value.split("_")[1])) for value in val_data
        ]
        patient_ids_to_match = [val[0] for val in train_mask_values]
        study_ids_to_match = [val[1] for val in train_mask_values]
        filtered_df_idx = df[(df['patient_id'].isin(patient_ids_to_match)) &
                        (df['study_id'].isin(study_ids_to_match))].index.to_list()
        train_fold_dict.update({i: filtered_df_idx})
        patient_ids_to_match = [val[0] for val in val_mask_values]
        study_ids_to_match = [val[1] for val in val_mask_values]
        filtered_df_idx = df[(df['patient_id'].isin(patient_ids_to_match)) &
                        (df['study_id'].isin(study_ids_to_match))].index.to_list()
        val_fold_dict.update({i: filtered_df_idx})
    return train_fold_dict, val_fold_dict


def main(input: pathlib.Path, output:pathlib.Path) -> None:
    raise NotImplementedError()

if __name__ == "__main__":
    file_path = pathlib.Path("/home/rolandnemeth/Radboud/Masters/AIMI/AIMI_project/input/picai_labels/clinical_information/marksheet.csv")
    dataset = TabularDataset(file_path)
    print(dataset.__getitem__(0))
    # parser = argparse.ArgumentParser()
    # parser.add_argument("filename", type=pathlib.Path, default="./ancillery_data_prep.py")
    # parser.add_argument("--input_path", type=pathlib.Path, required=True)
    # parser.add_argument("--output_path", type=pathlib.Path, default="./preprocessed_ancillery_data")
