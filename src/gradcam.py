"""Grad-CAM: which part of the image made the model choose a class?

Idea (Selvaraju et al., 2017):
  1. Take the feature maps A_k of the last conv layer (e.g. 1280 maps of 7x7).
  2. Compute the gradient of the class score y_c with respect to each map.
  3. Average each gradient map -> one weight alpha_k per feature map
     ("how important is map k for class c").
  4. Heatmap = ReLU( sum_k alpha_k * A_k ), upsampled to the image size.
     ReLU keeps only regions that push the score UP.

Leaf-focus score (our addition): what share of the heatmap lies on the leaf,
measured with the PlantVillage leaf masks. See leaf_focus_score().
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD
from .evaluate import short_name
from .utils import ensure_dir, save_json


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        # A forward hook runs every time target_layer produces an output.
        self.handle = target_layer.register_forward_hook(self._save_activation)

    def remove(self):
        """Detach the hook so the model behaves normally again."""
        self.handle.remove()

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()
        # Ask PyTorch to also give us the gradient of this output during backward().
        if output.requires_grad:  # False inside torch.no_grad(), e.g. normal evaluation
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def __call__(self, image_tensor, class_idx=None):
        """image_tensor: (3, H, W) normalised. Returns (heatmap HxW in [0,1], class, probability)."""
        self.model.eval()
        x = image_tensor.unsqueeze(0).requires_grad_(True)  # batch of 1
        logits = self.model(x)
        probs = torch.softmax(logits.detach(), dim=1)[0]
        if class_idx is None:
            class_idx = int(probs.argmax())  # explain the predicted class

        self.model.zero_grad()
        logits[0, class_idx].backward()  # gradient of this class score only

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)        # (1, K, 1, 1) = alpha_k
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=image_tensor.shape[1:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # scale to [0, 1]
        return cam.cpu().numpy(), class_idx, float(probs[class_idx])


def denormalize(image_tensor):
    """Undo ImageNet normalisation -> numpy (H, W, 3) image in [0, 1] for plotting."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(image, cam, alpha=0.45):
    """Blend the heatmap (jet colours: red = important) on top of the image."""
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


def mask_for_cam(dataset, idx, cam_shape):
    """Leaf mask of sample idx as a 0/1 array with the same size as the heatmap."""
    mask = dataset.load_mask(idx).resize((cam_shape[1], cam_shape[0]), Image.NEAREST)
    return (np.asarray(mask) > 127).astype(np.float32)


def leaf_focus_score(cam, mask):
    """Share of the Grad-CAM "heat" that lies on the leaf (0..1).

    score = sum(cam * mask) / sum(cam). 1.0 = all heat on the leaf,
    0.0 = all heat on the background. Compare it with the leaf's share of
    the image area: a heatmap spread evenly would score exactly that.
    """
    return float((cam * mask).sum() / (cam.sum() + 1e-8))


def evaluate_leaf_focus(model, target_layer, dataset, device, out_path, max_images=300, seed=0):
    """Average leaf-focus score over (up to) max_images PlantVillage test images.

    `dataset` must have leaf masks (pv_dataset(...) from data.py), so this
    only works on PlantVillage, not on PlantDoc (no masks there).
    """
    cam_fn = GradCAM(model, target_layer)
    order = np.random.default_rng(seed).permutation(len(dataset))[:max_images]
    scores, areas, correct = [], [], []
    for idx in order:
        image, label = dataset[int(idx)]
        cam, pred, _ = cam_fn(image.to(device))
        mask = mask_for_cam(dataset, int(idx), cam.shape)
        scores.append(leaf_focus_score(cam, mask))
        areas.append(float(mask.mean()))
        correct.append(pred == label)
    cam_fn.remove()
    scores, areas, correct = np.array(scores), np.array(areas), np.array(correct)
    result = {
        "num_images": int(len(scores)),
        "mean_leaf_focus": float(scores.mean()),
        "median_leaf_focus": float(np.median(scores)),
        # what an evenly spread heatmap would score: the leaf's share of the image
        "mean_leaf_area": float(areas.mean()),
        "mean_leaf_focus_correct": float(scores[correct].mean()) if correct.any() else None,
        "mean_leaf_focus_wrong": float(scores[~correct].mean()) if (~correct).any() else None,
    }
    save_json(result, out_path)
    print(f"leaf focus {result['mean_leaf_focus']:.3f} (leaf area {result['mean_leaf_area']:.3f}, "
          f"{len(scores)} images) -> {out_path}")
    return result


def save_gradcam_examples(model, target_layer, dataset, class_names, device, out_path,
                          title, n_correct=4, n_incorrect=4, seed=0, max_tries=400):
    """Find some correctly and some wrongly classified images and plot
    original | Grad-CAM overlay for each. Returns the number of each found.
    If the dataset has leaf masks, the leaf-focus score is shown in the title."""
    cam_fn = GradCAM(model, target_layer)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))[:max_tries]

    correct, incorrect = [], []
    for idx in order:
        if len(correct) >= n_correct and len(incorrect) >= n_incorrect:
            break
        image, label = dataset[int(idx)]
        cam, pred, prob = cam_fn(image.to(device))
        focus = (leaf_focus_score(cam, mask_for_cam(dataset, int(idx), cam.shape))
                 if getattr(dataset, "mask_root", None) else None)
        item = (image, cam, label, pred, prob, focus)
        if pred == label and len(correct) < n_correct:
            correct.append(item)
        elif pred != label and len(incorrect) < n_incorrect:
            incorrect.append(item)
    cam_fn.remove()

    rows = correct + incorrect
    if not rows:
        print("No images found for Grad-CAM.")
        return 0, 0
    fig, axes = plt.subplots(len(rows), 2, figsize=(7, 3.2 * len(rows)), squeeze=False)
    for r, (image, cam, label, pred, prob, focus) in enumerate(rows):
        img = denormalize(image)
        status = "CORRECT" if pred == label else "WRONG"
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"true: {short_name(class_names[label])}", fontsize=8)
        axes[r, 1].imshow(overlay(img, cam))
        focus_text = f"\nleaf focus {focus:.2f}" if focus is not None else ""
        axes[r, 1].set_title(f"{status} pred: {short_name(class_names[pred])} ({prob:.2f})"
                             + focus_text, fontsize=8, color="green" if pred == label else "red")
        axes[r, 0].axis("off")
        axes[r, 1].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))  # leave room for the title
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path} ({len(correct)} correct, {len(incorrect)} wrong)")
    return len(correct), len(incorrect)
