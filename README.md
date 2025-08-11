## Aim of project

This project creates an adversarial noise generator suing the targeted fast gradient sign method on a pre-trained torchvision ResNet50 CNN.


## Project overview 
**Inputs:** specific input image and a target classification identity from, in this case, ImageNet class ID.

**Script functionality:** adds a small amount on adversarial noise to encourage the model to misclassify the input image as the target class, while ensuring the image to the human eye still looks highly similar to the input image.

**Outputs:** output image (which looks similar to input image but with adversarial noise), adversarial noise image, json file, 

## Usage
SHould have all files in the same folder

```bash

python adversarial_noise_generator.py --input (specify directory to example image) --target (specify target number from ImageNet or whichever dataset) --eps (specify weighting of adversarial noise) --out (specify desired output directory)

example: 
python adversarial_noise_generator.py --input samples/dog.jpg --target 281 --eps 0.03 --out adversarial_output/

or on a Spyder comman_line/ window
!python adversarial_noise_generator.py --input samples/dog.jpg --target 281 --eps 0.03 --out adversarial_output/
