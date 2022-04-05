#!/bin/bash

function arch_n_loss () {

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_"$3"_model_"$1"_loss_"$2" --batch-size 4 --model $1  \
    --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss $2

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_"$3"_model_"$1"_loss_"$2"  --batch-size 1 --model $1  --gpus=1 \
    --predict --pred-folder ./data/cleaned/preds_"$3"_model_"$1"_loss_"$2"
}

function rotate_patient () {
    python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --test-patient $1 --length 10 --rotate-patient
    arch_n_loss doublelatetemporal mse outer_"$1"
    arch_n_loss doublelate3d mse outer_"$1"
    arch_n_loss doublelateconvtemporal cos outer_"$1"
}

python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --test-patient P18 --length 10 --save-indexes

arch_n_loss doublelatetemporal mse outer_P18
arch_n_loss doublelate3d mse outer_P18
arch_n_loss doublelateconvtemporal cos outer_P18

rotate_patient P20
rotate_patient P21
rotate_patient P25
rotate_patient P30
