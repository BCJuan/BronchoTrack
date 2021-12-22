#!/bin/bash
# python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --intra-patient --only-val --length 2

function train_prune_simple () {
    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_15traj_COS_prune_"$1" --batch-size 16 --mode doublelate \
    --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos --prune $2

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_15traj_COS_prune_$1 --batch-size 1 --mode doublelate  --gpus 0 \
    --predict --pred-folder ./data/cleaned/preds_15traj_COS_prune_"$1"
}

function train_prune_distill_teacher () {
    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_15traj_COS_prune_"$1"_distill_teacher_"$4" --batch-size 16 --mode doublelate \
    --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos --prune $2 --distill-teacher $3 \
    --teacher-alpha $5

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_15traj_COS_prune_$1_distill_teacher_"$4" --batch-size 1 --mode doublelate  --gpus 0 \
    --predict --pred-folder ./data/cleaned/preds_15traj_COS_prune_"$1"_distill_teacher_"$4"
}

function train_prune_full_distill () {
    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_15traj_COS_prune_"$1"_distill_teacher_"$4" --batch-size 16 --mode doublelate \
    --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos --prune $2 --distill-teacher $3 \
    --teacher-alpha $5

    python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
    --ckpt-name bronchonet_15traj_COS_prune_$1_distill_teacher_"$4" --batch-size 1 --mode doublelate  --gpus 0 \
    --predict --pred-folder ./data/cleaned/preds_15traj_COS_prune_"$1"_distill_teacher_"$4"
}

# train_prune_simple 01 0.1
# train_prune_simple 03 0.3
# train_prune_simple 05 0.5
# train_prune_simple 07 0.7
# train_prune_simple 09 0.9

train_prune_distill_teacher 01 0.1 ./checkpoints/bronchonet_15traj_COS_prune_01_doublelate/bronchonet_15traj_COS_prune_01_doublelate.ckpt 02 0.2