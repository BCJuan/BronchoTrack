#!/bin/bash

python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --intra-patient --length 15

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_COS_arch --batch-size 16 --mode doublelatetemporal  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_COS_arch   --batch-size 1 --mode doublelatetemporal  --gpus 0 \
--predict --pred-folder ./data/cleaned/preds_15traj_COS_arch 

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_COS_arch  --batch-size 16 --mode doublelate3d  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_COS_arch  --batch-size 1 --mode doublelate3d  --gpus 0 \
--predict --pred-folder ./data/cleaned/preds_15traj_COS_arch 