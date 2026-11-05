"""
Step 2: a minimal PaDiM-style anomaly detector.

Trains ONLY on 'good' images (no labels needed for the fitting step
itself). For each spatial patch position, it learns the normal
distribution of pretrained-CNN features seen across your good images.
At test time it measures how far a new image's features are from that
distribution, patch by patch -- producing a heatmap that should light
up over the blank-spot defect.

This is a simplified, from-scratch version of PaDiM (Defard et al.,
2021, https://doi.org/10.1007/978-3-030-68799-1_35) written for
clarity rather than production use. With only a handful of 'good'
training images, treat this as a proof-of-concept for your pipeline --
the per-patch statistics get noticeably more reliable once you have
30+ good images to estimate them from.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMG_SIZE = 224
N_PROJECTED_DIMS = 50  # random feature-dimension reduction, as in the PaDiM paper
EPS = 0.01              # covariance regularization so matrices stay invertible

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_extractor():
    """Loads a pretrained ResNet18 and taps its mid-level feature maps."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.eval().to(device)

    features = {}

    def hook(name):
        def fn(_module, _input, output):
            features[name] = output
        return fn

    model.layer2.register_forward_hook(hook("layer2"))
    model.layer3.register_forward_hook(hook("layer3"))

    preprocess = weights.transforms()
    return model, features, preprocess


def extract_patch_embeddings(model, features, preprocess, image_path):
    """Runs one image through the network and returns one feature vector per spatial patch."""
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)

    f2 = features["layer2"].squeeze(0)  # (128, 28, 28)
    f3 = features["layer3"].squeeze(0)  # (256, 14, 14)
    f3_up = F.interpolate(f3.unsqueeze(0), size=f2.shape[-2:], mode="nearest").squeeze(0)
    embedding = torch.cat([f2, f3_up], dim=0)  # (384, 28, 28)

    C, H, W = embedding.shape
    embedding = embedding.reshape(C, H * W).permute(1, 0)  # (num_patches, C)
    return embedding.cpu().numpy(), (H, W)


def fit_padim(good_image_paths):
    """Learns a per-patch mean + covariance of 'normal' features from good images only."""
    model, features, preprocess = get_feature_extractor()

    all_embeddings, hw = [], None
    for p in good_image_paths:
        emb, hw = extract_patch_embeddings(model, features, preprocess, p)
        all_embeddings.append(emb)
    all_embeddings = np.stack(all_embeddings, axis=0)  # (N, num_patches, C)

    N, num_patches, C = all_embeddings.shape
    print(f"Fitting on {N} good images, {num_patches} patches/image, {C} channels.")
    if N < 10:
        print("Heads up: with fewer than ~10 good images the per-patch covariance "
              "is poorly estimated -- fine for testing the pipeline, not for "
              "trusting the numbers yet.")

    rng = np.random.default_rng(0)
    proj_idx = rng.choice(C, size=min(N_PROJECTED_DIMS, C), replace=False)
    proj = all_embeddings[:, :, proj_idx]  # (N, num_patches, d)

    means = proj.mean(axis=0)  # (num_patches, d)
    d = proj.shape[-1]
    covs = np.zeros((num_patches, d, d))
    for i in range(num_patches):
        x = proj[:, i, :]  # (N, d)
        cov = np.cov(x, rowvar=False) if N > 1 else np.zeros((d, d))
        covs[i] = cov + EPS * np.eye(d)

    return {
        "model": model, "features": features, "preprocess": preprocess,
        "proj_idx": proj_idx, "means": means,
        "cov_inv": np.linalg.inv(covs), "hw": hw,
    }


def score_image(state, image_path):
    """Returns a per-pixel anomaly score map: high = far from the learned 'normal' distribution."""
    emb, hw = extract_patch_embeddings(state["model"], state["features"], state["preprocess"], image_path)
    proj = emb[:, state["proj_idx"]]

    diffs = proj - state["means"]
    dists = np.einsum("pi,pij,pj->p", diffs, state["cov_inv"], diffs)
    dists = np.sqrt(np.clip(dists, 0, None))

    H, W = hw
    score_map = dists.reshape(H, W)
    score_map_upsampled = np.array(
        Image.fromarray(score_map).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    )
    return score_map_upsampled


def visualize(image_path, score_map, save_path=None):
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(img)
    axes[1].imshow(score_map, cmap="jet", alpha=0.5)
    axes[1].set_title("Anomaly heatmap")
    axes[1].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    good_paths = sorted(glob.glob(os.path.join(DATA_DIR, "good", "*.*")))
    defective_paths = sorted(glob.glob(os.path.join(DATA_DIR, "defective", "*.*")))

    if len(good_paths) < 5:
        print("Put at least a handful of 'good' images in data/good/ before running this.")
    else:
        state = fit_padim(good_paths)

        # Score defective images -- the heatmap should light up over the blank spot.
        for p in defective_paths[:3]:
            visualize(p, score_image(state, p))

        # Score one good image as a sanity check -- heatmap should look mostly flat.
        if good_paths:
            visualize(good_paths[0], score_image(state, good_paths[0]))
