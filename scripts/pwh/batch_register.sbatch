#!/bin/bash
#SBATCH --job-name=voxelmorph
#SBATCH --output=./slurm_log/voxelmorph.log
#
#SBATCH --ntasks=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --qos=debug

# this file will batch register the with moved output image format <FIXED_FILENAME>-<MOVING_FILENAME>.nii using voxelmorph

# need to explicit manage the python path for cluster server for conflicting local tensorflow
export PYTHONPATH=/home/jacky/anaconda3/envs/voxelmorph/lib/python3.6/site-packages

PWH_DATA_DIR/home/jacky/data/pwh/normalized
OUTPUT_DIR=/home/jacky/data/pwh/registered

MODEL=/home/jacky/Projects/voxelmorph/models/dir-lab/1500.h5
echo "Using voxelmorph model:" $MODEL

for case in $PWH_DATA_DIR/*; do
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
            MOVED=$OUTPUT_DIR/case/image/$FIXED_FILENAME-${MOVING_FILENAME}.nii
            WARP=$OUTPUT_DIR/case/field/$FIXED_FILENAME-${MOVING_FILENAME}.nii
            echo "Registering the image:"
            echo "FIXED:" $FIXED
            echo "MOVED:" $MOVED
            echo "WARP:" $WARP
        # #     python ./scripts/tf/register.py --moving --fixed --moved --model $MODEL --warp 
        done
    done
    break
done 