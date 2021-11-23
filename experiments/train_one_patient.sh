python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --test-pacient "$1" --only-val
python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --compute-statistics --n-trajectories 15 --test-pacient "$1" --only-val

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_MSE_"$1" --batch-size 2 --mode doublelatetemporal  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --loss mse

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_MSE_"$1" --batch-size 1 --mode doublelatetemporal  --gpus 0 \
--predict --pred-folder ./data/cleaned/preds_15traj_MSE_"$1" --only-val

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_COS_"$1" --batch-size 2 --mode doublelatetemporal  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --loss cos

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_COS_"$1" --batch-size 1 --mode doublelatetemporal  --gpus 0 \
--predict --pred-folder ./data/cleaned/preds_15traj_COS_"$1" --only-val

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_DE_"$1" --batch-size 2 --mode doublelatetemporal  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --loss direction

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_DE_"$1" --batch-size 1 --mode doublelatetemporal  --gpus 0 \
--predict --pred-folder ./data/cleaned/preds_15traj_DE_"$1" --only-val

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_QUAT_"$1" --batch-size 2 --mode doublelatetemporal  \
--gpus 1,2 --accelerator 'ddp' --lr 0.0001 --loss quaternion

python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ \
--ckpt-name bronchonet_15traj_QUAT_"$1" --batch-size 1 --mode doublelatetemporal  --gpus 0 \
--predict --pred-folder ./data/cleaned/preds_15traj_QUAT_"$1" --only-val