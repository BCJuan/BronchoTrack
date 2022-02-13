#!/bin/bash

# python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --intra-patient --length 10

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_15traj_l10_offsetnet --batch-size 4 --model offsetnet  \
# --gpus 1,2 --accelerator 'ddp' --lr 0.0001

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_15traj_l10_offsetnet  --batch-size 1 --model offsetnet  --gpus=1 \
# --predict --pred-folder ./data/cleaned/preds_15traj_comparison_offsetnet


export CUDA_VISIBLE_DEVICES="0"

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_l10_comparison_prune_03_v2_b10 --batch-size 10 --model doublelateconvtemporal  \
--gpus=1 --lr 0.0001 --rot-loss cos --prune 0.3

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_l10_comparison_prune_03_v2_b10  --batch-size 1 --model doublelateconvtemporal  --gpus=1 \
--predict --pred-folder ./data/cleaned/preds_15traj_comparison_bronchotrack_v2_b16

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_15traj_l10_deependovo --batch-size 1 --model deependovo  \
# --gpus=1 --lr 0.0001

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_15traj_l10_deependovo  --batch-size 1 --model deependovo  --gpus=1 \
# --predict --pred-folder ./data/cleaned/preds_15traj_comparison_deependovo

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_15traj_l10_comparison_no_prune --batch-size 4 --model doublelateconvtemporal  \
# --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_15traj_l10_comparison_no_prune  --batch-size 1 --model doublelateconvtemporal  --gpus=1 \
# --predict --pred-folder ./data/cleaned/preds_15traj_comparison_bronchotrack_no_prune