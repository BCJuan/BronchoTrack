# BronchoTrack

Lightweight bronchoscopy tracking through a hierarchically pruned and distilled recurrent convolutional neural network

## Instructions for data

Go to `experiments`. To prepare train, val and test splits, place the files (images and csvs) into `experiments/data/raw_data` folder and then run:

`python organize.py`

`python organize.py --root data/raw_data/ --new-root data/cleaned --split --compute-statistics`
`python organize.py --root data/raw_data/ --new-root data/cleaned --split --clean --split-size=10`

## Training

`python train.py --root data/cleaned/ --image-root data/raw_data/`

## Testing

`python train.py --root data/cleaned/ --image-root data/raw_data/ --predict --ckpt <checkpoint-file>`