#!/bin/bash
IMAGE_LIST=/mnt/DIIR-JK-NAS/data/lung_data/normalized/data_list.txt
MODEL_DIR=/home/jacky/DIIR-JK-NAS/projects/voxelmorph/models/dir-lab_hypermorph

# voxelmorph
# python ./scripts/tf/train.py --img-list $IMAGE_LIST --model-dir $MODEL_DIR --gpu 0

# hypermorph
python ./scripts/tf/train_hypermorph.py --img-list $IMAGE_LIST --model-dir $MODEL_DIR --gpu 1

# synthmorph
# python ./scripts/tf/train_synthmorph.py --img-list $IMAGE_LIST --model-dir $MODEL_DIR --gpu 1