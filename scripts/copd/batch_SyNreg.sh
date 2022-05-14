#!/bin/bash

echo "Batch SyN-Registration for DIR-LAB Data"

export ANTSPATH=/home/jacky/Sources/ANTs/install/bin/
export PATH=$ANTSPATH:$PATH

COPD_DATA_DIR=/mnt/DIIR-JK-NAS/data/LungDataCOPD/normalized
OUTPUT_DIR=/mnt/DIIR-JK-NAS/data/LungDataCOPD/SyN_df

SyNQuick_PATH=$ANTSPATH/antsRegistrationSyNQuick.sh

num_proc=$(nproc --all)

function SyNQuick(){
    fixed=$COPD_DATA_DIR/copd$1/copd$1_iBHCT.nii
    moving=$COPD_DATA_DIR/copd$1/copd$1_eBHCT.nii
    warp=$OUTPUT_DIR/copd$1/copd$1_iBHCT-copd$1_eBHCT-

    log_file=$OUTPUT_DIR/copd$1/log.txt

    echo "Registering case $1..."
    if [ ! -d ${warp%/*} ] 
    then
        mkdir -p ${warp%/*}
    fi

    ${SyNQuick_PATH} -d 3 -f $fixed -m $moving -o $warp -n $num_proc| tee -a $log_file
}

for i in $(seq 1 10)
do
    SyNQuick ${i}
done