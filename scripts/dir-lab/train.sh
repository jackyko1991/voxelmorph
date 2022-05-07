#!/bin/bash
IMAGE_LIST=/mnt/DIIR-JK-NAS/data/lung_data/normalized/data_list.txt
MODEL_DIR=/home/jacky/DIIR-JK-NAS/projects/voxelmorph/models/dir-lab

python ./scripts/tf/train.py --img-list $IMAGE_LIST --model-dir $MODEL_DIR --gpu 1