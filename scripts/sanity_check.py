"""
Step 1: prove the pipeline works before doing any real modeling.

Expected folder layout (create this next to this script):

    data/
        good/         <- photos of normal, defect-free plates
        defective/    <- photos with the blank/void defect visible

Run with:  python scripts/sanity_check.py
"""

import glob
import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_image_paths():
    good = sorted(glob.glob(os.path.join(DATA_DIR, "good", "*.*")))
    defective = sorted(glob.glob(os.path.join(DATA_DIR, "defective", "*.*")))
    print(f"Found {len(good)} good images and {len(defective)} defective images.")
    if len(good) == 0:
        print("!! No images found in data/good/ -- check the folder path and that "
              "you've sorted your 15 photos into data/good/ and data/defective/.")
    return good, defective


def show_a_few(paths, title, n=3):
    n = min(n, len(paths))
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, paths[:n]):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(os.path.basename(p), fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def test_pretrained_forward_pass(sample_path):
    """
    Loads an ImageNet-pretrained ResNet18 and runs one image through it.
    We are NOT classifying anything useful here -- this just proves
    torch/torchvision are installed correctly and can run a real model.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    img = Image.open(sample_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        out = model(x)

    print(f"Forward pass OK. Output shape: {tuple(out.shape)} "
          f"(1000 = ImageNet class logits, not meaningful for us yet).")


if __name__ == "__main__":
    good, defective = load_image_paths()
    show_a_few(good, "Sample 'good' images")
    show_a_few(defective, "Sample 'defective' images")

    all_images = good + defective
    if all_images:
        test_pretrained_forward_pass(all_images[0])
    else:
        print("Add images to data/good/ and data/defective/ and re-run.")
