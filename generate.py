import argparse
import math

import torch
from torchvision import utils, transforms
import cv2
from PIL import Image
import glob

from model import StyledGenerator


@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style


@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device):
    image = generator(torch.randn(n_sample, 512).to(device), step=step, alpha=1, mean_style=mean_style, style_weight=0.7)

    return image


@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
    source_code = torch.randn(n_source, 512).to("cpu")
    target_code = torch.randn(n_target, 512).to("cpu")

    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7)
    target_image = generator(target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7)

    images.append(source_image)

    for i in range(n_target):
        image = generator([target_code[i].unsqueeze(0).repeat(n_source, 1), source_code], step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7, mixing_range=(0, 1))
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)

    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128, help="size of the image")
    parser.add_argument("--n_row", type=int, default=3, help="number of rows of sample matrix")
    parser.add_argument("--n_col", type=int, default=3, help="number of columns of sample matrix")
    parser.add_argument("path", type=str, help="path to checkpoint file")

    args = parser.parse_args()

    device = "cpu"

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path, map_location=torch.device("cpu"))["g_running"])
    generator.eval()

    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2

    resize_img = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=112),
            transforms.ToTensor(),
        ]
    )
    for j in range(500):
        img = sample(generator, step, mean_style, args.n_col * args.n_row, device)
        # img = [resize_img(im) for im in img]
        utils.save_image(img, "images/" + "%03d" % j + "_image.png", nrow=args.n_col, normalize=True, range=(-1, 1), padding=0)
        print(j)
