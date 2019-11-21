# Style-Based GAN in PyTorch
![example output](https://miro.medium.com/max/3780/1*dhXsTa5zOVk79AowpMHKbg.png)

Repository that gathers code to train StyleGAN.

The default parameters employs progressive training from 8x8 to 128x128, but this may be modified by passing arguments in `train.py`

Based on https://github.com/rosinality/style-based-gan-pytorch

## Getting started

First, create a new virtual environment

```
virtualenv venv -p python3
source venv/bin/activate
```

You might need to make sure your python3 link is ready by typing

```bash
which python3
```

Then install the development requirements

```bash
pip install -r requirements.txt
```

Download the CelebA data (if you're not planning to use your own)

```
python helper.py
```

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
