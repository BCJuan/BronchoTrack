#!/bin/bash

python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --intra-patient --length 10

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_l10_offsetnet --batch-size 4 --model offsetnet  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_l10_COSoffsetnet  --batch-size 1 --model offsetnet  --gpus=1 \
--predict --pred-folder ./data/cleaned/preds_15traj_comparison_offsetnet

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_l10_comparison_prune_03 --batch-size 4 --model doublelateconvtemporal  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos --prune 0.3

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_l10_COS_comparison  --batch-size 1 --model doublelateconvtemporal  --gpus=1 \
--predict --pred-folder ./data/cleaned/preds_15traj_comparison_bronchotrack
