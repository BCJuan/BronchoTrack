# BronchoTrack

Lightweight bronchoscopy tracking through a hierarchically pruned and distilled recurrent convolutional neural network

## Instructions for data

Go to `experiments`. To prepare train, val and test splits, place the files (images and csvs) into `experiments/data/raw_data` folder and then run:

`python organize.py`

`python organize.py --root data/raw_data/ --new-root data/cleaned --split --compute-statistics`
`python organize.py --root data/raw_data/ --new-root data/cleaned --split --clean --split-size=10`
`python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --split --clean --split-size=20`
`python organize.py --root /mnt/DADES/datasetcalibracio/ --new-root data/cleaned --split --clean --split-size=5 --cutdown=0.01`

## Training

`python train.py --root data/cleaned/ --image-root data/raw_data/ --ckpt-name bronchonet --batch-size 16 --model singletemporal --gpus 0,1 --acceleration 'dp' --log-every-n-steps 4 --lr 0.005`

models: "singletemporal", "doubleearlytemporal", "doublelatetemporal", "doublelate", "offsetnet"

`python train.py --root data/cleaned/ --image-root data/raw_data/ --ckpt-name offsetnet --batch-size 4 --m offsetnet --gpus 2 --accelerator 'ddp' --log_every_n_steps 4 --lr 0.0005`

## Testing

`python train.py --root data/cleaned/ --image-root data/raw_data/ --predict --ckpt <checkpoint-file>`