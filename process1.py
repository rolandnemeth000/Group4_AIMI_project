#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from pathlib import Path
import pickle 
import os 
import pandas as pd 

import numpy as np
import SimpleITK as sitk
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)

from picai_baseline.unet.training_setup.default_hyperparam import \
    get_default_hyperparams
from picai_baseline.unet.training_setup.neural_network_selector import \
    neural_network_for_run
from picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample, PreprocessingSettings, crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter


class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline U-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # set expected i/o paths in gc env (image i/p, algorithms, prediction o/p)
        # see grand-challenge.org/algorithms/interfaces/ for expected path per i/o interface
        # note: these are fixed paths that should not be modified

        # directory to model weights
        self.algorithm_weights_dir = Path("/opt/algorithm/weights/")

        # path to image files
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri/",
            "/input/images/transverse-adc-prostate-mri/",
            "/input/images/transverse-hbv-prostate-mri/",
            # "/input/images/coronal-t2-prostate-mri/",  # not used in this algorithm
            # "/input/images/sagittal-t2-prostate-mri/"  # not used in this algorithm
        ]
        self.image_input_paths = [list(Path(x).glob("*.mha"))[0] for x in self.image_input_dirs]

        # load clinical information
        with open("/input/clinical-information-prostate-mri.json") as fp:
            self.clinical_info = json.load(fp)

        # path to output files
        self.detection_map_output_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.case_level_likelihood_output_file = Path("/output/cspca-case-level-likelihood.json")

        # create output directory
        self.detection_map_output_path.parent.mkdir(parents=True, exist_ok=True)

        # define compute used for training/inference ('cpu' or 'cuda')
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # extract available clinical metadata (not used for this example)
        self.age = self.clinical_info["patient_age"]

        if "PSA_report" in self.clinical_info:
            self.psa = self.clinical_info["PSA_report"]
        else:
            self.psa = None  # value is missing, if not reported

        if "PSAD_report" in self.clinical_info:
            self.psad = self.clinical_info["PSAD_report"]
        else:
            self.psad = None  # value is missing, if not reported

        if "prostate_volume_report" in self.clinical_info:
            self.prostate_volume = self.clinical_info["prostate_volume_report"]
        else:
            self.prostate_volume = None  # value is missing, if not reported

        # extract available acquisition metadata (not used for this example)
        self.scanner_manufacturer = self.clinical_info["scanner_manufacturer"]
        self.scanner_model_name = self.clinical_info["scanner_model_name"]
        self.diffusion_high_bvalue = self.clinical_info["diffusion_high_bvalue"]

        # define input data specs [image shape, spatial res, num channels, num classes]
        self.img_spec = {
            'image_shape': [20, 256, 256],
            'spacing': [3.0, 0.5, 0.5],
            'num_channels': 3,
            'num_classes': 2,
        }

        # load trained algorithm architecture + weights
        self.models = []
        model_arch = ['unet']
        model_folds = [range(5)]

        # for each given architecture
        for arch_name, folds in zip(model_arch, model_folds):

            # for each trained 5-fold instance of a given architecture
            for fold in folds:
                # path to trained weights for this architecture + fold (e.g. 'unet_F4.pt')
                checkpoint_path = self.algorithm_weights_dir / f'{arch_name}_F{fold}.pt'

                # skip if model was not trained for this fold
                if not checkpoint_path.exists():
                    continue

                # define the model specifications used for initialization at train-time
                # note: if the default hyperparam listed in picai_baseline was used,
                # passing arguments 'image_shape', 'num_channels', 'num_classes' and
                # 'model_type' via function 'get_default_hyperparams' is enough.
                # otherwise arguments 'model_strides' and 'model_features' must also
                # be explicitly passed directly to function 'neural_network_for_run'
                args = get_default_hyperparams({
                    'model_type': arch_name,
                    **self.img_spec
                })

                model = neural_network_for_run(args=args, device=self.device)

                # load trained weights for the fold
                checkpoint = torch.load(checkpoint_path)
                print(f"Loading weights from {checkpoint_path}")
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                self.models += [model]
                print("Complete.")
                print("-"*100)

        # display error/success message
        if len(self.models) == 0:
            raise Exception("No models have been found/initialized.")
        else:
            print(f"Success! {len(self.models)} model(s) have been initialized.")
            print("-"*100)
        






        #Doing things for tabular data 
        #loading the different pahts 
        #TEST_TABULAR_PATH = Path(
        #    "C:/Users/nienk.LAPTOP-QC09UOEF/Documents/picai challenge/picai_unet_gc_algorithm - kopie - kopie/test/clinical-information-prostate-mri.json"
        #)  
        
        #MODELS_DIR = Path("C:/Users/nienk.LAPTOP-QC09UOEF/Documents/picai challenge/picai_unet_gc_algorithm - kopie - kopie/tabular_models")
        self.MODELS_DIR = Path("/opt/algorithm/tabular_models/")

        #os.makedirs("outputtesttest", exist_ok=True)
        #OUTPUT_DIR = Path("C:/Users/nienk.LAPTOP-QC09UOEF/Documents/picai challenge/picai_unet_gc_algorithm - kopie - kopie/outputtesttest")

        self.models_tabular = []
        for model_path in self.MODELS_DIR.glob(r"MLP_fold-[0-9].pkl"):
            with open(model_path, mode="rb") as model_file:
                self.models_tabular.append(pickle.load(model_file))
        with open(self.MODELS_DIR / "mlp_scaler.pkl", mode="rb") as scaler_file:
            self.scaler = pickle.load(scaler_file)

        
        
        #with open(TEST_TABULAR_PATH, mode="r") as f:
        #    tabular_data = json.load(f)
        #age = tabular_data["patient_age"]
        #psa = tabular_data["PSA_report"]
        #psad = tabular_data["PSAD_report"]
        #prostate_volume = tabular_data["prostate_volume_report"]


        tabular_data_dict = {
            "patient_age": self.age,
            "psa": self.psa,
            "psad": self.psad,
            "prostate_volume": self.prostate_volume,
        }

        tabular_data_df = pd.DataFrame(tabular_data_dict, index=[0])
        self.input_data = tabular_data_df
        self.input_data = self.scaler.transform(self.input_data)



    # generate + save predictions, given images
    def predict(self):

        print("Preprocessing Images ...")

        # read images (axial sequences used for this example only)
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.image_input_paths
            ],
            settings=PreprocessingSettings(
                matrix_size=self.img_spec['image_shape'], 
                spacing=self.img_spec['spacing']
            )
        )

        # preprocess - align, center-crop, resample
        sample.preprocess()
        cropped_img = [
            sitk.GetArrayFromImage(x)
            for x in sample.scans
        ]

        # preprocessing - intensity normalization + expand channel dim
        preproc_img = [
            z_score_norm(np.expand_dims(x, axis=0), percentile=99.5)
            for x in cropped_img
        ]

        # preprocessing - concatenate + expand batch dim
        preproc_img = np.expand_dims(np.vstack(preproc_img), axis=0)

        # preprocessing - convert to PyTorch tensor
        preproc_img = torch.from_numpy(preproc_img)

        print("Complete.")

        # test-time augmentation (horizontal flipping)
        img_for_pred = [preproc_img.to(self.device)]
        img_for_pred += [torch.flip(preproc_img, [4]).to(self.device)]

        # begin inference
        outputs = []
        print("Generating Predictions ...")

        # for each member model in ensemble
        for p in range(len(self.models)):

            # switch model to evaluation mode
            self.models[p].eval()

            # scope to disable gradient updates
            with torch.no_grad():
                # aggregate predictions for all tta samples
                preds = [
                    torch.sigmoid(self.models[p](x))[:, 1, ...].detach().cpu().numpy()
                    for x in img_for_pred
                ]

                # revert horizontally flipped tta image
                preds[1] = np.flip(preds[1], [3])

                # gaussian blur to counteract checkerboard artifacts in
                # predictions from the use of transposed conv. in the U-Net
                outputs += [
                    np.mean([
                        gaussian_filter(x, sigma=1.5)
                        for x in preds
                    ], axis=0)[0]
                ]

        # ensemble softmax predictions
        ensemble_output = np.mean(outputs, axis=0).astype('float32')

        # read and resample images (used for reverting predictions only)
        sitk_img = [
            sitk.ReadImage(str(path)) for path in self.image_input_paths
        ]
        resamp_img = [
            sitk.GetArrayFromImage(
                resample_img(x, out_spacing=self.img_spec['spacing'])
            )
            for x in sitk_img
        ]

        # revert softmax prediction to original t2w - reverse center crop
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            ensemble_output, size=resamp_img[0].shape))
        cspca_det_map_sitk.SetSpacing(list(reversed(self.img_spec['spacing'])))

        # revert softmax prediction to original t2w - reverse resampling
        cspca_det_map_sitk = resample_img(cspca_det_map_sitk,
                                          out_spacing=list(reversed(sitk_img[0].GetSpacing())))

        # process softmax prediction to detection map
        cspca_det_map_npy = extract_lesion_candidates(
            sitk.GetArrayFromImage(cspca_det_map_sitk), threshold='dynamic')[0]

        # remove (some) secondary concentric/ring detections
        cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/5)] = 0

        # make sure that expected shape was matched after reverse resampling (can deviate due to rounding errors) 
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            cspca_det_map_npy, size=sitk.GetArrayFromImage(sitk_img[0]).shape))

        # works only if the expected shape matches
        cspca_det_map_sitk.CopyInformation(sitk_img[0])

        # save detection map
        atomic_image_write(cspca_det_map_sitk, self.detection_map_output_path)

        
        #Doing things for tabular data         
        ensemble_output_2 = np.empty((len(self.models_tabular), 2))

        for i, model in enumerate(self.models_tabular):
            ensemble_output_2[i] = model.predict_proba(self.input_data)
                
        ensemble_output_2_mean = ensemble_output_2.mean(axis=1)[1]

        # with open(OUTPUT_DIR / "csPCA_probability_prediction.json", mode="w") as f:
        #     json.dump(float(ensemble_output_2_mean), f, indent=4)


        ## Combining the things (real)
        prob_img = float(np.max(cspca_det_map_npy))
        prob_tab = ensemble_output_2_mean
        weight_img = 0.7 
        weight_tab = 0.3 

        weighted_img = weight_img * prob_img
        weighted_tab = weight_tab * prob_tab

        total_prob_value = weighted_img+weighted_tab



        # save case-level likelihood
        with open(str(self.case_level_likelihood_output_file), 'w') as f:
            json.dump(total_prob_value, f)


if __name__ == "__main__":
    csPCaAlgorithm().predict()
