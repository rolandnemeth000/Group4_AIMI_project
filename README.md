# Artificial Intelligence for Medical Imaging Project
## Group 4
 - Roland Németh
 - Nienke Muskee
 - Yunda Qiu
 - Amany El-Gharabawy

## Description
Welcome to the GitHub repository hosting the code and experiment settings of Group 4 that was used to participate in the project phase of Artificial Intelligence for Medical Imaging and PI-CAI challenge. The followind documents describes what is needed to reproduce our experiments.

## Prerequisites
The experiments heavily rely on the code provided by the hosts. Therefore, I would like to refer the reader to the corresponding repositories:

> [picai_baseline].(https://github.com/DIAGNijmegen/picai_baseline)
The ```picai_baseline``` repository contains scripts to train a UNet models that were used in our experiments and as such essential to reproduce our results.

> [picai_unet_semi_supervised_gc_algorithm].(https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm)
The ```picai_unet_semi_supervised_gc_algorithm``` repository contains scripts to prepare the docker container for submissions and the an already trained baseline model which we also utilized in our experiments. 

> [picai_labels].(https://github.com/DIAGNijmegen/picai_labels)
Contains the label data and clinical data of the challenge. 

> [PI-CAI challenge].(https://pi-cai.grand-challenge.org/)
The challenge website for additional information.

Please follow the instructions on their corresponding webpages for proper installations.

## Structure of the repository
This repository contains multiple directories. The directories and subdirectories are described in the upcoming subsections.

### code
This directory contains the 5 subdirectories and the ```postprocessing.py``` file. 

#### postprocessing.py
This file contains the development code of the postprocessing experiment.

#### hp_exp_jobs
This directory contains ```.sh``` or ```.bash``` used to submit jobs on Snellius. Not all jobs were successful (due to errors) and therefore not all of them were reported about in the project report. However, the ones not listed below can be submitted and reproduced on Snellius by just submitting the corresponding  ```.sh``` or ```.bash``` file.

Experiment 5 were submitted but was cancelled after being deemed redundant.
Experiment 6 took to long to complete with the queue time and therefore was cancelled.
Experiments 7, 8, 9 and10 resulted in an error which we did not manage to debug. (The bug only persists on Snellius.)
Experiment 12 contained SparseAdam optimizer which was not suited for training this kind of neural network.
Experiments 14 and 16 took too much memory and eventually stopped with a memory error.
Experiments 17, 18 and LeakyRelu-BatchNorm were unsuccesful modifications of the model architecture which resulted in errors which we didn't have time to debug.

#### process_files
This directory contains the modified ```process.py``` files that were used for local testing and constructing docker containers for the postprocessing experiment and clinical data experiments. While most of the code was taken from ```[picai_unet_semi_supervised_gc_algorithm].(https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm)```it contains our modifications as well.

#### training_extensions
This directory contains modifications to the training procedures such as changing the optimizer.

#### tabular_data
This directory contains the script used to train our clinical data models.

#### utils
This directory contains utilities such as log analyzer and testing the output of the clinical data models.

### docker 
This directory contains 3 subdirectories.
1. ```docker_hp``` is a subdirectory that contains (almost) all necessary components to build and export a Docker container of the chosen hyperparameter (hp) experiment. Please copy the weights of the experiments you want to dockerize into the weights folder. (All weights can be downloaded from [here].(url))
2. ```docker_postprocessing``` is a subdirectory that contains (almost) all necessary components to build and export a Docker container of the postprocessing experiment. Please copy the **baseline** ```weights``` into the ```weights``` folder before dockerizing. (The weights can be downloaded from [here].(url))
3. ```docker_tabular``` is a subdirectory that contains (almost) all necessary components to build and export a Docker container of the tabular experiment. Please copy the **baseline** weights into the ```weights``` folder and the selected tabular model **files** (without their directory) from ```AIMI_project/tabular_models``` into the ```docker/docker_tabular/tabular_models``` before dockerizing it. (The weights can be downloaded from [here].(url))

### logs
Contains the retrieved logs from training on Snellius. The most up to date can be found ```logs/logs/logs```

### tabular_models
Contains the directories of the models. Ridge and MLP were not submitted to the challenge. If the reader wants to reuse these, only copy the files (not the directories) as instructed in the previous section.
