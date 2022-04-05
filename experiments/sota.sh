#!/bin/bash

python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --n-trajectories 15 --intra-patient --length 10 --clean

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_intra_model_convtemporal_loss_mse_ce --batch-size 4 --model doublelateconvtemporal  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_intra_model_convtemporal_loss_mse_ce --batch-size 1 --model doublelateconvtemporal --gpus=1 \
--predict --pred-folder ./data/cleaned/preds_broncho_intra_model_convtemporal_loss_mse_ce_FINAL

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name offsetnet_loss_mse_ce --batch-size 4 --model offsetnet  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name offsetnet_loss_mse_ce --batch-size 1 --model offsetnet --gpus=1 \
--predict --pred-folder ./data/cleaned/preds_offsetnet_loss_mse_ce_FINAL
