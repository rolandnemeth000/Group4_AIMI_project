import os
import pathlib
import gc
import json
import typing
import pickle
import argparse
import itertools
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

def load_tabular(path:pathlib.Path) -> typing.Union[pd.DataFrame|pd.Series]:
    assert path.suffix == ".csv"
    df = pd.read_csv(path)
    return df

def select_feature_columns(df:typing.Union[pd.DataFrame|pd.Series]) -> typing.Union[pd.DataFrame|pd.Series]:
    cols = ["patient_id", "study_id", "patient_age", "psa", "psad", "prostate_volume", "case_csPCa"]
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

# def get_splits(splits_path: pathlib.Path) -> list[dict]:
#     with open(splits_path, mode="r") as sf:
#         splits = json.load(sf)
#     return splits

# def find_folds(df: typing.Union[pd.DataFrame|pd.Series], splits:list[dict]):
#     train_fold_dict = {}
#     val_fold_dict = {}
#     for i, split in enumerate(splits):
#         train_data = split["train"]
#         val_data = split["val"]
#         train_mask_values = [
#             (int(value.split("_")[0]), int(value.split("_")[1])) for value in train_data
#         ]
#         val_mask_values = [
#             (int(value.split("_")[0]), int(value.split("_")[1])) for value in val_data
#         ]
#         patient_ids_to_match = [val[0] for val in train_mask_values]
#         study_ids_to_match = [val[1] for val in train_mask_values]
#         filtered_df_idx = df[(df['patient_id'].isin(patient_ids_to_match)) &
#                         (df['study_id'].isin(study_ids_to_match))].index.to_list()
#         train_fold_dict.update({i: filtered_df_idx})
#         patient_ids_to_match = [val[0] for val in val_mask_values]
#         study_ids_to_match = [val[1] for val in val_mask_values]
#         filtered_df_idx = df[(df['patient_id'].isin(patient_ids_to_match)) &
#                         (df['study_id'].isin(study_ids_to_match))].index.to_list()
#         val_fold_dict.update({i: filtered_df_idx})
#     return train_fold_dict, val_fold_dict

def train_ridge_ensemble(data):
    kfold = KFold(n_splits=5)
    models = []
    enc = LabelEncoder()
    scaler = StandardScaler()
    X = scaler.fit_transform(data[["patient_age", "psa", "psad", "prostate_volume"]])
    y = enc.fit_transform(data["case_csPCa"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        model = RidgeClassifier()
        X_train_ = X_train[train_idx]
        y_train_ = y_train[train_idx]
        model.fit(X_train_, y_train_)
        X_val_ = X_train[val_idx]
        y_val_ = y_train[val_idx]
        y_pred = model.predict(X_val_)
        roc_auc = roc_auc_score(y_val_, y_pred)
        f1 = f1_score(y_val_, y_pred)
        print(f"Fold {fold}, ROC-AUC: {roc_auc}, F1-score: {f1}")
        models.append((f"rc{fold}", model))
        os.makedirs("tabular_models", exist_ok=True)
        with open(f"tabular_models/ridge_fold-{fold}.pkl", mode="wb") as model_file:
            pickle.dump(model, model_file)
    
    with open("tabular_models/ridge_scaler.pkl", mode="wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    y_test_pred_ = np.empty((len(models), len(y_test)))
    for i, (_, model) in enumerate(models):
        y_test_pred_[i] = model.predict(X_test) 
    y_test_pred = list(map(lambda y_: 1 if y_ > len(models)//2 else 0, y_test_pred_.sum(axis=0)))
    roc_auc = roc_auc_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print(f"Ridge ensemble: ROC-AUC: {roc_auc}, F1-score: {f1}")

def train_logistic_ensemble(data):
    kfold = KFold(n_splits=5)
    models = []
    enc = LabelEncoder()
    scaler = StandardScaler()
    X = scaler.fit_transform(data[["patient_age", "psa", "psad", "prostate_volume"]])
    y = enc.fit_transform(data["case_csPCa"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        model = LogisticRegression()
        X_train_ = X_train[train_idx]
        y_train_ = y_train[train_idx]
        model.fit(X_train_, y_train_)
        X_val_ = X_train[val_idx]
        y_val_ = y_train[val_idx]
        y_pred = model.predict(X_val_)
        roc_auc = roc_auc_score(y_val_, y_pred)
        f1 = f1_score(y_val_, y_pred)
        print(f"Fold {fold}, ROC-AUC: {roc_auc}, F1-score: {f1}")
        models.append((f"rc{fold}", model))
        os.makedirs("tabular_models", exist_ok=True)
        with open(f"tabular_models/logistic_fold-{fold}.pkl", mode="wb") as model_file:
            pickle.dump(model, model_file)
    
    with open("tabular_models/logistic_scaler.pkl", mode="wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    y_test_pred_ = np.empty((len(models), len(y_test)))
    for i, (_, model) in enumerate(models):
        y_test_pred_[i] = model.predict(X_test) 
    y_test_pred = list(map(lambda y_: 1 if y_ > len(models)//2 else 0, y_test_pred_.sum(axis=0)))
    roc_auc = roc_auc_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print(f"Logistic ensemble: ROC-AUC: {roc_auc}, F1-score: {f1}")

def train_mlp_ensemble(data):
    kfold = KFold(n_splits=5)
    models = []
    enc = LabelEncoder()
    scaler = StandardScaler()
    X = scaler.fit_transform(data[["patient_age", "psa", "psad", "prostate_volume"]])
    y = enc.fit_transform(data["case_csPCa"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        model = MLPClassifier(hidden_layer_sizes=(8), learning_rate="adaptive", max_iter=200, early_stopping=True, tol=0.1)
        X_train_ = X_train[train_idx]
        y_train_ = y_train[train_idx]
        model.fit(X_train_, y_train_)
        X_val_ = X_train[val_idx]
        y_val_ = y_train[val_idx]
        y_pred = model.predict(X_val_)
        roc_auc = roc_auc_score(y_val_, y_pred)
        f1 = f1_score(y_val_, y_pred)
        print(f"Fold {fold}, ROC-AUC: {roc_auc}, F1-score: {f1}")
        models.append((f"rc{fold}", model))
        os.makedirs("tabular_models", exist_ok=True)
        with open(f"tabular_models/MLP_fold-{fold}.pkl", mode="wb") as model_file:
            pickle.dump(model, model_file)
    
    with open("tabular_models/mlp_scaler.pkl", mode="wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    y_test_pred_ = np.empty((len(models), len(y_test)))
    for i, (_, model) in enumerate(models):
        y_test_pred_[i] = model.predict(X_test) 
    y_test_pred = list(map(lambda y_: 1 if y_ > len(models)//2 else 0, y_test_pred_.sum(axis=0)))
    roc_auc = roc_auc_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print(f"MLP ensemble: ROC-AUC: {roc_auc}, F1-score: {f1}")

def train_svc_ensemble(data):
    kfold = KFold(n_splits=5)
    models = []
    enc = LabelEncoder()
    scaler = StandardScaler()
    X = scaler.fit_transform(data[["patient_age", "psa", "psad", "prostate_volume"]])
    y = enc.fit_transform(data["case_csPCa"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        model = SVC()
        X_train_ = X_train[train_idx]
        y_train_ = y_train[train_idx]
        model.fit(X_train_, y_train_)
        X_val_ = X_train[val_idx]
        y_val_ = y_train[val_idx]
        y_pred = model.predict(X_val_)
        roc_auc = roc_auc_score(y_val_, y_pred)
        f1 = f1_score(y_val_, y_pred)
        print(f"Fold {fold}, ROC-AUC: {roc_auc}, F1-score: {f1}")
        models.append((f"rc{fold}", model))
        os.makedirs("tabular_models", exist_ok=True)
        with open(f"tabular_models/svc_fold-{fold}.pkl", mode="wb") as model_file:
            pickle.dump(model, model_file)

    with open("tabular_models/svc_scaler.pkl", mode="wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    
    y_test_pred_ = np.empty((len(models), len(y_test)))
    for i, (_, model) in enumerate(models):
        y_test_pred_[i] = model.predict(X_test) 
    y_test_pred = list(map(lambda y_: 1 if y_ > len(models)//2 else 0, y_test_pred_.sum(axis=0)))
    roc_auc = roc_auc_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print(f"SVC ensemble: ROC-AUC: {roc_auc}, F1-score: {f1}")

def main(input: pathlib.Path, output:pathlib.Path) -> None:
    raise NotImplementedError()

if __name__ == "__main__":
    file_path = pathlib.Path("/home/rolandnemeth/AIMI_project/input/picai_labels/clinical_information/marksheet.csv")
    df = load_data(file_path)
    train_ridge_ensemble(df)
    train_logistic_ensemble(df)
    train_mlp_ensemble(df)
    train_svc_ensemble(df)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("filename", type=pathlib.Path, default="./ancillery_data_prep.py")
    # parser.add_argument("--input_path", type=pathlib.Path, required=True)
    # parser.add_argument("--output_path", type=pathlib.Path, default="./preprocessed_ancillery_data")
