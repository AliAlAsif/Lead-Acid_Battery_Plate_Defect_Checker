"""
Step 2 (revised): supervised transfer-learning classifier.

With a roughly balanced good/defective split, this is a better fit
than the anomaly-detection route (train_padim.py), which is designed
for the opposite situation -- mostly good, rare defects. Here we use
the labels you actually have.

Approach: fine-tune an ImageNet-pretrained ResNet18. We only retrain
the last block + final layer, because with 35 source images (even
heavily augmented) there isn't enough data to safely retrain the
whole network -- doing so would just memorize your 35 photos instead
of learning generalizable features. Freezing most of the network and
relying on ImageNet's already-learned edge/texture detectors is what
makes this workable at your data size at all.

Expected folder layout (same as before):
    data/
        good/
        defective/
"""
import os
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PlateDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def load_paths():
    import glob
    good = sorted(glob.glob(os.path.join(DATA_DIR, "good", "*.*")))
    defective = sorted(glob.glob(os.path.join(DATA_DIR, "defective", "*.*")))
    paths = good + defective
    labels = [0] * len(good) + [1] * len(defective)  # 0 = good, 1 = defective
    print(f"{len(good)} good, {len(defective)} defective, {len(paths)} total.")
    return paths, labels


def build_model():
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # Freeze everything except the last residual block and the final layer --
    # this is the part that adapts to "battery plate" instead of ImageNet's
    # 1000 general object categories, without overfitting to 35 photos.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4") or name.startswith("fc")

    model.fc = nn.Linear(model.fc.in_features, 2)  # good vs defective
    return model.to(device)


def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    paths, labels = load_paths()
    if len(paths) < 10:
        print("Not enough images found -- check data/good/ and data/defective/.")
        return

    train_p, val_p, train_y, val_y = train_test_split(
        paths, labels, test_size=0.25, stratify=labels, random_state=SEED
    )
    print(f"Train: {len(train_p)}  Val: {len(val_p)}")

    train_ds = PlateDataset(train_p, train_y, train_transform)
    val_ds = PlateDataset(val_p, val_y, eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, train=False)
        print(f"Epoch {epoch:02d}  train_loss={tr_loss:.3f} train_acc={tr_acc:.2f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.2f}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

    print(f"\nBest validation accuracy: {best_acc:.2f}")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "best_model.pt"))

    # Final report on the validation set with the best checkpoint
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x.to(device))
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_true.extend(y.tolist())

    print("\nClassification report (0=good, 1=defective):")
    print(classification_report(all_true, all_preds, target_names=["good", "defective"]))
    print("Confusion matrix:")
    print(confusion_matrix(all_true, all_preds))


if __name__ == "__main__":
    main()
