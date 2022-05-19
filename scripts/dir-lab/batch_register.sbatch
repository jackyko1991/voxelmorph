#!/bin/bash
#SBATCH --job-name=voxelmorph
#SBATCH --output=/home/jacky/Projects/voxelmorph/slurm_log/voxelmorph.log
#
#SBATCH --ntasks=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --qos=deployment

# this file will batch register the with moved output image format <FIXED_FILENAME>-<MOVING_FILENAME>.nii using voxelmorph

# need to explicit manage the python path for cluster server for conflicting nvidia tensorflow that installed at ~/.local
export PYTHONPATH=/home/jacky/anaconda3/envs/voxelmorph/lib/python3.6/site-packages

# VXM_PROJ_DIR=/home/jacky/Projects/voxelmorph
VXM_PROJ_DIR=/mnt/DIIR-JK-NAS/projects/voxelmorph

# DIR_LAB_DATA_DIR=/home/jacky/data/dir-lab/normalized
DIR_LAB_DATA_DIR=/mnt/DIIR-JK-NAS/data/lung_data/normalized
# DIR_LAB_DATA_DIR=/mnt/DIIR-JK-NAS/data/lung_data/normalized
# OUTPUT_DIR=/mnt/DIIR-JK-NAS/data/lung_data/registered_voxelmorph
# OUTPUT_DIR=/mnt/DIIR-JK-NAS/data/lung_data/registered_synthmorph
OUTPUT_DIR=/mnt/DIIR-JK-NAS/data/lung_data/registered_synthmorph_shape-dice-vel-3-res-8-16-32-128

# MODEL=/home/jacky/Projects/voxelmorph/models/dir-lab/1500.h5
# MODEL=/home/jacky/Projects/voxelmorph/models/synthmorph/weights.h5

# MODEL=/home/jacky/DIIR-JK-NAS/projects/voxelmorph/models/dir-lab/1500.h5
# MODEL=/home/jacky/DIIR-JK-NAS/projects/voxelmorph/models/synthmorph/shape-dice-vel-3-res-8-16-32-256/weights.h5
MODEL=/home/jacky/DIIR-JK-NAS/projects/voxelmorph/models/synthmorph/shape-dice-vel-3-res-8-16-32-128/01500.h5

echo "Using voxelmorph model:" $MODEL

for case in $DIR_LAB_DATA_DIR/*; do
    if [ ! -d "$case" ]; then
        continue
    fi
    for FIXED in $case/*;do
        FIXED_BASE=${FIXED##*/}
        FIXED_FILENAME=${FIXED_BASE%.*}

        for MOVING in $case/*;do
            MOVING_BASE=${MOVING##*/}
            MOVING_FILENAME=${MOVING_BASE%.*}
            echo $FIXED_FILENAME,$MOVING_FILENAME
            MOVED=$OUTPUT_DIR/${case##*/}/image/$FIXED_FILENAME-${MOVING_FILENAME}.nii
            WARP=$OUTPUT_DIR/${case##*/}/field/$FIXED_FILENAME-${MOVING_FILENAME}.nii

            MOVED_DIR=`echo $MOVED |xargs dirname`
            WARP_DIR=`echo $WARP |xargs dirname`

            mkdir -p ${MOVED_DIR}
            mkdir -p ${WARP_DIR}

            echo "Registering the image:"
            echo "FIXED:" $FIXED
            echo "MOVING:" $MOVING
            echo "MOVED:" $MOVED
            echo "WARP:" $WARP
            python $VXM_PROJ_DIR/scripts/tf/register.py --moving $MOVING --fixed $FIXED --moved $MOVED --model $MODEL --warp $WARP --gpu 0
            break
        done
        break
    done
    break
done 