#!/bin/bash

echo "Batch SyN-Registration for DIR-LAB Data"

export ANTSPATH=/home/jacky/Sources/ANTs/install/bin/
export PATH=$ANTSPATH:$PATH

DIR_LAB_DATA_DIR=/mnt/DIIR-JK-NAS/data/LungDataPWH/data_normalized
OUTPUT_DIR=/mnt/DIIR-JK-NAS/data/LungDataPWH/SyN_df_jacky

SyNQuick_PATH=$ANTSPATH/antsRegistrationSyNQuick.sh

num_proc=$(nproc --all)

function SyNQuick(){
    fixed=$DIR_LAB_DATA_DIR/$1/$1_ins.nii
    moving=$DIR_LAB_DATA_DIR/$1/$1_exp.nii
    warp=$OUTPUT_DIR/$1/$1_ins-$1_exp-

    log_file=$OUTPUT_DIR/$1/log.txt

    echo "Registering case $1..."
    if [ ! -d ${warp%/*} ] 
    then
        mkdir -p ${warp%/*}
    fi

    ${SyNQuick_PATH} -d 3 -f $fixed -m $moving -o $warp -n $num_proc| tee -a $log_file
}

# for i in $(seq 1 18)
# do
#     SyNQuick ${i}
# done

for i in "00" "01"
do 
    SyNQuick ${i}
done