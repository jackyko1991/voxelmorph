#!/bin/bash
#SBATCH --job-name=synthmorph
#SBATCH --output=/home/jacky/Projects/voxelmorph/slurm_log/synthmorph.log
#
#SBATCH --ntasks=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --qos=deployment

export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:${LD_LIBRARY_PATH}
# need to explicit manage the python path for cluster server for conflicting local tensorflow
export PYTHONPATH=/home/jacky/anaconda3/envs/voxelmorph/lib/python3.6/site-packages

VXM_DIR=/home/jacky/Projects/voxelmorph

# IMAGE_LIST=/home/jacky/DIIR-JK-NAS/data/lung_data/normalized/data_list.txt
# MODEL_DIR=/home/jacky/DIIR-JK-NAS/projects/voxelmorph/models/dir-lab
# IMAGE_LIST=/home/jacky/data/dir-lab/normalized/data_list.txt
# MODEL_DIR=/home/jacky/voxelmorph_models/dir-lab
MODEL_DIR=$VXM_DIR/models/synthmorph

# python ./scripts/dir-lab/dir-lab_prepare.py
# voxelmoph
# python ./scripts/tf/train.py --img-list $IMAGE_LIST --model-dir $MODEL_DIR --gpu 0

# hypermorph
# python ./scripts/tf/train_hypermorph.py --img-list $IMAGE_LIST --model-dir $MODEL_DIR --gpu 0

# synthmorph
python $VXM_DIR/scripts/tf/train_synthmorph_custom.py --shape 128 128 128 --vel-res 8 16 32 --vel-std 3 --num-map=40 --model-dir $MODEL_DIR --sub-dir shape-dice-vel-3-res-8-16-32-128 --gpu 0 --save-freq 100