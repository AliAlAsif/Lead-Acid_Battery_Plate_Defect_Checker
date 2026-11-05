"""Cluster real plate photos by visual similarity to propose plate groupings.

Uses the ImageNet ResNet18 (same backbone as the project model) as a feature
extractor. Images of the SAME physical plate should have near-identical
grid/paste features and thus very high cosine similarity; different plates are
more distant. This is a *proposal* — plate identity must still be confirmed.
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

ROOT = r"E:\PycharmProjects\Battery defect ML\data"

pre = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()
feat = {}

def hook(name):
    def fn(_m, _i, o):
        feat[name] = o
    return fn

model.avgpool.register_forward_hook(hook("pool"))


def embed(path):
    img = Image.open(path).convert("RGB")
    x = pre(img).unsqueeze(0)
    with torch.no_grad():
        model(x)
    v = feat["pool"].squeeze()  # 512-d
    return v / (v.norm() + 1e-9)


# Real image lists (synthetic 767x1023 files excluded).
REAL = {
    "good": sorted(glob.glob(os.path.join(ROOT, "good", "*.*"))),
    "surface_error": sorted(glob.glob(os.path.join(ROOT, "surface_error", "*.*"))),
    "crack": [p for p in sorted(glob.glob(os.path.join(ROOT, "crack", "*.*")))
              if os.path.basename(p)[0].isdigit() and not os.path.basename(p).startswith(("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_"))],
}

all_paths = []
for cls, ps in REAL.items():
    for p in ps:
        all_paths.append((cls, os.path.basename(p), p))

embs = np.stack([embed(p).numpy() for _, _, p in all_paths])
sim = embs @ embs.T  # cosine similarity

names = [f"{c}/{b}" for c, b, _ in all_paths]
print(f"{len(all_paths)} real images\n")

# Top matches per image (within same class).
print("=== nearest neighbours (same class, similarity > 0.90) ===")
for i, (c, b, _) in enumerate(all_paths):
    row = [(sim[i, j], names[j]) for j in range(len(all_paths))
           if j != i and all_paths[j][0] == c]
    row.sort(reverse=True)
    strong = [f"{n} ({s:.3f})" for s, n in row if s > 0.90]
    if strong:
        print(f"{names[i]:28s} ~ {', '.join(strong[:6])}")

# Cross-class top matches (to spot the same plate appearing in two folders).
print("\n=== cross-class matches (similarity > 0.90) ===")
for i in range(len(all_paths)):
    for j in range(i + 1, len(all_paths)):
        if all_paths[i][0] != all_paths[j][0] and sim[i, j] > 0.90:
            print(f"{names[i]:28s} ~ {names[j]:28s} ({sim[i,j]:.3f})")
