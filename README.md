# Style-Based GAN in PyTorch

Mostly from https://github.com/rosinality/style-based-gan-pytorch

## Prepare the dataset
```
python prepare_data.py --out data PATH_TO_DIRECTORY
```

## Train the model
```
python train.py --mixing data
```

## Generate new faces
```
python generate.py checkpoint/full-latest.model
```
