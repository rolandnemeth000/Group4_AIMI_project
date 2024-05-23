#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/mnienke/logs/slurm-%j.out

# Load modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Set up environment
cp repos/AIMI_project/code/train_ext.py $HOME/repos/picai_baseline/src/picai_baseline/unet/
python3 -m pip install --user --upgrade pip
python3 -m pip install --user scikit-build
python3 -m pip install --user -e $HOME/repos/picai_baseline
python3 -m pip install --user -r $HOME/repos/picai_baseline/src/picai_baseline/unet/requirements.txt

# Copy data to scratch storage
# echo "Copying data to $TMPDIR"
# cp -r $HOME/data/ $TMPDIR/data


mkdir $TMPDIR/data
cp -rv $HOME/data/images $TMPDIR/data/images
cp -rv $HOME/data/picai_labels/ $TMPDIR/data/picai_labels

# Prepare the data
python3 $HOME/repos/picai_baseline/src/picai_baseline/prepare_data_semi_supervised.py --workdir=$TMPDIR/data/preprocessed --imagesdir=$TMPDIR/data/images --labelsdir=$TMPDIR/data/picai_labels --spacing 3.0 0.5 0.5 --matrix_size 20 256 256

#overviews
python3 $HOME/repos/picai_baseline/src/picai_baseline/unet/plan_overview.py --task=Task2203_picai_baseline --workdir=$TMPDIR/data --preprocessed_data_path=$TMPDIR/data/preprocessed/nnUNet_raw_data/{task} --overviews_path=$TMPDIR/data/overviews/unet

# Copy the data back for later user
cp -rv $TMPDIR/data/preprocessed $HOME/data/preprocessed
cp -rv $TMPDIR/data/overviews $HOME/data/overviews


# Run training script
python3 -u $HOME/repos/picai_baseline/src/picai_baseline/unet/train_ext.py --weights_dir $HOME/models/picai/unet/hp_exp13 --overviews_dir $TMPDIR/data/overviews/unet --folds 0 1 2 3 4 --num_epochs 100 --optimizer SGD

## Changed optimizer