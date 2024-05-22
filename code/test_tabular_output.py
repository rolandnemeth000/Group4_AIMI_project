import json
import pickle
from pathlib import Path
import os

import numpy as np
import pandas as pd

TEST_TABULAR_PATH = Path(
    "/home/rolandnemeth/AIMI_project/repos/picai_unet_semi_supervised_gc_algorithm/test/clinical-information-prostate-mri.json"
)

MODELS_DIR = Path("/home/rolandnemeth/AIMI_project/tabular_models")

os.makedirs("output", exist_ok=True)
OUTPUT_DIR = Path("/home/rolandnemeth/AIMI_project/output")


def main():
    models = []
    for model_path in MODELS_DIR.glob(r"MLP_fold-[0-9].pkl"):
        with open(model_path, mode="rb") as model_file:
            models.append(pickle.load(model_file))

    with open(MODELS_DIR / "mlp_scaler.pkl", mode="rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(TEST_TABULAR_PATH, mode="r") as f:
        tabular_data = json.load(f)

    age = tabular_data["patient_age"]
    psa = tabular_data["PSA_report"]
    psad = tabular_data["PSAD_report"]
    prostate_volume = tabular_data["prostate_volume_report"]

    tabular_data_dict = {
        "patient_age": age,
        "psa": psa,
        "psad": psad,
        "prostate_volume": prostate_volume,
    }

    tabular_data_df = (
        pd.DataFrame(tabular_data_dict, index=[0])
        if isinstance(age, (int, float))
        else pd.DataFrame.from_dict(tabular_data_dict)
    )

    input_data = tabular_data_df

    input_data = scaler.transform(input_data)

    ensemble_output = np.empty((len(models), len(input_data)))

    for i, model in enumerate(models):
        ensemble_output[i] = model.predict_proba(input_data)[:, 1]

    ensemble_output_mean = ensemble_output.mean(axis=0)
    print(float(ensemble_output_mean))

    with open(OUTPUT_DIR / "csPCA_probability_prediction.json", mode="w") as f:
        json.dump(float(ensemble_output_mean), f, indent=4)


if __name__ == "__main__":
    main()
