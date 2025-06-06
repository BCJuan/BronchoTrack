# BronchoTrack

Lightweight bronchoscopy tracking through a hierarchically pruned and distilled recurrent convolutional neural network

## Instructions for data

Go to `experiments`. To prepare train, val and test splits, place the files (images and csvs) into `experiments/data/raw_data` folder and then run:

`python organize.py`

For outer patient scheme:

`python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --test-pacient P18 --only-val --length 2 `

For intra patient scheme:

`python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --intra-patient --length 2`

For trajectories of 15 image (14 pairs):

`python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --clean --n-trajectories 15 --only-val --intra-patient --length 15`

## Training

models: "singletemporal", "doubleearlytemporal", "doublelatetemporal", "doublelate", "offsetnet"

`python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ --ckpt-name bronchonet_15traj_MSE --batch-size 2 --mode doublelatetemporal  --gpus 1,2 --accelerator 'ddp' --lr 0.0001`

## Testing

`python train.py --root data/cleaned/ --image-root data/raw_data/ --predict --ckpt <checkpoint-file>`

`python train.py --root data/cleaned/ --image-root /mnt/DADES/datasetcalibracio/ --ckpt-name bronchonet_15traj_MSE --batch-size 1 --mode doublelatetemporal  --gpus 0  --predict --ckpt checkpoints/Loss/bronchonet_15traj_MSE-epoch\=055-val_loss\=3.13769.ckpt --pred-folder ./data/cleaned/preds_15traj_MSE --only-val`


### Pruning libraries that might support inference acceleration

+ https://github.com/VainF/Torch-Pruning
+ https://github.com/marcoancona/TorchPruner
+ https://github.com/jacobgil/pytorch-pruning

Other interesting infos

+ https://jacobgil.github.io/deeplearning/pruning-deep-learning
+ https://arxiv.org/pdf/2011.14691.pdf
+ https://github.com/SforAiDl/KD_Lib
+ https://spell.ml/blog/model-pruning-in-pytorch-X9pXQRAAACIAcH9h