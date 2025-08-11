
"""
FGSM Adversarial Noise Generator
--------------------------------
Generates an adversarial image from an input image such that a pretrained CNN
misclassifies it into a user-specified target class, using Fast Gradient Sign Method (FGSM).

Usage:
    python fgsm_advgen.py --input path/to/image.jpg --target 281 --eps 8/255 --out output_dir
"""

import argparse
import json
from pathlib import Path
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from processing_utils import *

def fgsm(model, image, target_class, epsilon):
    """ 
    Adding advrsarial noise to the images
    Parameters: models> pretrained model
               image: input image
               target_class
               epsilon: weight of adv noise
     Returns: Image with adversarial noise
     """
    image = image.clone().detach().requires_grad_(True)
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([target_class])

    output = model(image)
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()

    # adv attack equation
    perturbation = -epsilon * image.grad.sign()
    adv_image = image + perturbation
    adv_image = torch.clamp(adv_image, -3, 3)  # normalised
    return adv_image.detach()



def main():
    parser = argparse.ArgumentParser(description="Targeted FGSM Adversarial Image Generator")
    parser.add_argument("--input", type=Path, required=True, help="Path to input image")
    parser.add_argument("--target", type=int, required=True, help="Target class ID (0-999 for ImageNet)")
    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name")
    parser.add_argument("--eps", type=float, default=8/255, help="Perturbation magnitude")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()




if __name__ == "__main__":
    main()

