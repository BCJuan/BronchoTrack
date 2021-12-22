#!/bin/bash
# python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --intra-patient --only-val --length 2

function train_prune_simple () {
    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_15traj_COS_prune_"$1" --batch-size 16 --mode doublelate \
    --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos --prune $2

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_15traj_COS_prune_$1 --batch-size 1 --mode doublelate  --gpus 0 \
    --predict --pred-folder ./data/cleaned/preds_15traj_COS_prune_$1
}

train_prune_simple 01 0.1
train_prune_simple 02 0.2
train_prune_simple 03 0.3
train_prune_simple 04 0.4
train_prune_simple 05 0.5
train_prune_simple 06 0.6
train_prune_simple 07 0.7
train_prune_simple 08 0.8
train_prune_simple 09 0.9
