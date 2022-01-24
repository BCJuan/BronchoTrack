#!/bin/bash
# python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 30 --intra-patient --only-val --length 2

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_30traj_MSE --batch-size 16 --mode doublelate \
# --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss mse --pos-loss mse

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_30traj_MSE --batch-size 1 --mode doublelate  --gpus 0 \
# --predict --pred-folder ./data/cleaned/preds_30traj_MSE

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_30traj_COS --batch-size 16 --mode doublelate  \
# --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos --pos-loss mse

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_30traj_COS --batch-size 1 --mode doublelate  --gpus 0 \
# --predict --pred-folder ./data/cleaned/preds_30traj_COS

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_30traj_DE --batch-size 16 --mode doublelate  \
# --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss direction --pos-loss mse

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_30traj_DE --batch-size 1 --mode doublelate  --gpus 0 \
# --predict --pred-folder ./data/cleaned/preds_30traj_DE

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_30traj_QUAT --batch-size 16 --mode doublelate  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss quaternion --pos-loss mse

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_30traj_QUAT --batch-size 1 --mode doublelate  --gpus 0 \
--predict --pred-folder ./data/cleaned/preds_30traj_QUAT

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_30traj_COS_EU --batch-size 16 --mode doublelate  \
# --gpus 1,2 --accelerator 'ddp' --lr 0.0001 --rot-loss cos --pos-loss euclidean

# python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
# --ckpt-name bronchonet_30traj_COS_EU --batch-size 1 --mode doublelate  --gpus 0 \
# --predict --pred-folder ./data/cleaned/preds_30traj_COS_EU
