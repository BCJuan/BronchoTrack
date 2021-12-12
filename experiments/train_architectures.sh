#!/bin/bash

#python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --intra-patient --length 10

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_l10_COS_arch --batch-size 4 --model doublelatetemporal  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name gbronchonet_15traj_l10_COS_arch   --batch-size 1 --model doublelatetemporal  --gpus=1 \
--predict --pred-folder ./data/cleaned/preds_15traj_COS_arch 

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name gbronchonet_15traj_l10_COS_arch  --batch-size 4 --model doublelate3d  \
# --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name gbronchonet_15traj_l10_COS_arch  --batch-size 1 --model doublelate3d  --gpus=1 \
# --predict --pred-folder ./data/cleaned/preds_15traj_COS_arch 

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_l10_COS_arch --batch-size 4 --model doublelate  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name gbronchonet_15traj_l10_COS_arch  --batch-size 1 --model doublelate  --gpus=1 \
--predict --pred-folder ./data/cleaned/preds_15traj_COS_arch 