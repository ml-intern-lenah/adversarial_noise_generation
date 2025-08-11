"""
Assistive functions for data loadign and model processing
"""

# importing all libraries that may be needed
import argparse
import json
from pathlib import Path
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# data loading form ImageNet datasets
def load_imagenet_classes():
    """
    Returns: Loaded examples of class names from the imagenet class examples
    """
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    class_names = urllib.request.urlopen(url).read().decode('utf-8').splitlines()
    return class_names

class_names = load_imagenet_classes()
print('check class names loaded ok', class_names)


# preprocessing- normalising, augment the dataset in preparation for CNN pretrained classifier
def preprocess_image(image_path: Path):
    """
    Parameters: image_path to images

    Returns : normalised and augmented image type
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0)


def deprocess(tensor):
    """
    Parameter: output tensor image
    Returns: un-normalised PIL image
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu().clone().squeeze(0)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return transforms.ToPILImage()(img)


# Tranfer learning - loading the pretrained CNN classification model

def load_model(name: str):
    """
    Parameters: specific name of pretained model e.g ResNet18/50
    Returns: Loadeda pretrained torchvision model
    """
    model = getattr(models, name)(pretrained=True)
    model.eval()
    return model


# prediction function
def predict(model, image, class_names):
    """
    Parameters: specific name of pretained model e.g ResNet18/50,input image, name of target classes
    Returns: predicted class, probability of class and 'human-visible label'
    """
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        conf, label = torch.max(probs, 1)
    return label.item(), conf.item(), class_names[label]
