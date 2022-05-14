#!/bin/bash
#SBATCH --job-name=voxelmorph
#SBATCH --output=./slurm_log/voxelmorph.log
#
#SBATCH --ntasks=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --qos=debug

export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:${LD_LIBRARY_PATH}
# need to explicit manage the python path for cluster server for conflicting local tensorflow
export PYTHONPATH=/home/jacky/anaconda3/envs/voxelmorph/lib/python3.6/site-packages

IMAGE_LIST=/home/jacky/DIIR-JK-NAS/data/lung_data/normalized/data_list.txt
MODEL_DIR=/home/jacky/DIIR-JK-NAS/projects/voxelmorph/models/dir-lab
# IMAGE_LIST=/home/jacky/data/dir-lab/normalized/data_list.txt
# MODEL_DIR=/home/jacky/voxelmorph_models/dir-lab

# python ./scripts/dir-lab/dir-lab_prepare.py
python ./scripts/tf/train.py --img-list $IMAGE_LIST --model-dir $MODEL_DIR --gpu 0