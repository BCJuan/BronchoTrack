# BronchoTrack

Lightweight bronchoscopy tracking through a hierarchically pruned and distilled recurrent convolutional neural network

## Instructions for data

Go to `experiments`. To prepare train, val and test splits, place the files (images and csvs) into `experiments/data/raw_data` folder and then run:

`python organize.py`

`python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --test-pacient P18 --only-val`

## Training

models: "singletemporal", "doubleearlytemporal", "doublelatetemporal", "doublelate", "offsetnet"

`python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ --ckpt-name bronchonet_15traj_MSE --batch-size 2 --mode doublelatetemporal  --gpus 1,2 --accelerator 'ddp' --lr 0.0001`

## Testing

`python train.py --root data/cleaned/ --image-root data/raw_data/ --predict --ckpt <checkpoint-file>`

`python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ --ckpt-name bronchonet_15traj_MSE --batch-size 1 --mode doublelatetemporal  --gpus 0  --predict --ckpt checkpoints/Loss/bronchonet_15traj_MSE-epoch\=055-val_loss\=3.13769.ckpt --pred-folder ./data/cleaned/preds_15traj_MSE --only-val`