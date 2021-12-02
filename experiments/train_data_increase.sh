#!/bin/bash

function one_patient_one_traj () {
    python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories $2 --only-val --patient "$1" --length 2

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_"$2"traj_COS_EU_inter_"$1" --batch-size 16 --mode doublelate  \
    --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_"$traj"traj_COS_EU_inter_"$1"  --batch-size 1 --mode doublelate  --gpus 0 \
    --predict --pred-folder ./data/cleaned/preds_"$2"traj_COS_EU_inter_"$1"
}

function one_patient (){
    one_patient_one_traj $1 15
    one_patient_one_traj $1 30
    one_patient_one_traj $1 60
    one_patient_one_traj $1 120
}

function intra_one_traj () {
    python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories $1 --only-val --intra-patient --length 2

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_"$1"traj_COS_EU_intra --batch-size 16 --mode doublelate  \
    --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_"$1"traj_COS_EU_intra  --batch-size 1 --mode doublelate  --gpus 0 \
    --predict --pred-folder ./data/cleaned/preds_"$1"traj_COS_EU_intra 
}

# intra_one_traj 15
intra_one_traj 60
intra_one_traj 120

one_patient P18
one_patient P20
one_patient P21
one_patient P25
one_patient P30
