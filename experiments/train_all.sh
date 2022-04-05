#!/bin/bash

function arch_n_loss () {

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_"$3"_model_"$1"_loss_"$2" --batch-size 4 --model $1  \
    --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss $2

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_"$3"_model_"$1"_loss_"$2"  --batch-size 1 --model $1  --gpus=1 \
    --predict --pred-folder ./data/cleaned/preds_"$3"_model_"$1"_loss_"$2"
}

function arch () {
    arch_n_loss $1 cos $2
    arch_n_loss $1 mse $2
    arch_n_loss $1 direction $2
    arch_n_loss $1 quaternion $2
}


python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --intra-patient --length 10

arch doublelate intra
arch doublelatetemporal intra
arch doublelate3d intra
arch doublelateconvtemporal intra

