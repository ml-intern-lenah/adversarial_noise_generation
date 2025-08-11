
"""
FGSM Adversarial Noise Generator
--------------------------------
Generates an adversarial image from an input image such that a pretrained CNN
misclassifies it into a user-specified target class, using Fast Gradient Sign Method (FGSM).
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
    
    # Prepare output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Load model, class names, and image
    model = load_model(args.model)
    class_names = load_imagenet_classes()
    image = preprocess_image(args.input)

    # prediction label before adding adversarial noise
    pre_label, pre_conf, pre_name = predict(model, image, class_names)
    
    # fgsm run 
    adv_image = fgsm(model, image, args.target, args.eps)

    # Predict after attack
    post_label, post_conf, post_name = predict(model, adv_image, class_names)

    # Save results
    adv_path = args.out / f"adv_{args.input.name}"
    diff_path = args.out / f"diff_{args.input.stem}.png"
    meta_path = args.out / f"meta_{args.input.stem}.json"

    deprocess(adv_image).save(adv_path)

    # amplified difference for visualisation
    diff = (adv_image - image).squeeze(0)
    diff = diff / (2 * args.eps) + 0.5  # normalize to [0,1]
    diff_img = transforms.ToPILImage()(diff)
    diff_img.save(diff_path)
    
    
    # adding metadata
    success = (post_label == args.target)
    meta = {
        "input": str(args.input),
        "adv": str(adv_path),
        "diff": str(diff_path),
        "model": args.model,
        "target": args.target,
        "target_name": class_names[args.target],
        "eps": args.eps,
        "success": success,
        "pre_label": pre_label,
        "pre_name": pre_name,
        "pre_conf": pre_conf,
        "post_label": post_label,
        "post_name": post_name,
        "post_conf": post_conf,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Adversarial image saved to {adv_path}")
    print(f"Diff image saved to {diff_path}")
    print(f"Metadata saved to {meta_path}")
    print(f"Attack success: {success}")
    print(f"Before: {pre_name} ({pre_conf:.4f})")
    print(f"After: {post_name} ({post_conf:.4f})")






if __name__ == "__main__":
    main()

