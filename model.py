import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
import time
import psutil

# torch.manual_seed(0)

colors = ["#FF0000FD", "#8CF1FF", "#0D723400"]


def get_patches(img: torch.Tensor, patch_size: int):
    """
    Split the image tensor into non-overlapping patches.

    Args:
        img: Tensor of shape (3, H, W) in [0,1]
        patch_size: size of each patch
    Returns:
        patches_tensor: (n_h, n_w, 3, patch_size, patch_size)
        n_h, n_w: number of patches
    """
    _, H, W = img.shape
    n_h, n_w = H // patch_size, W // patch_size
    # Crop extra
    img_cropped = img[:, :n_h * patch_size, :n_w * patch_size]
    # Unfold into grid of patches
    patches_tensor = img_cropped.unfold(1, patch_size, patch_size)
    patches_tensor = patches_tensor.unfold(2, patch_size, patch_size)
    patches_tensor = patches_tensor.permute(1, 2, 0, 3, 4)
    return patches_tensor, n_h, n_w


def get_scores_batched(patches_tensor: torch.Tensor, prompt: str,
                       model: CLIPModel, processor: CLIPProcessor, device: str) -> np.ndarray:
    """
    Batch all patches through CLIP model for a single prompt.
    Returns normalized scores [0,1] of shape (n_h, n_w).
    """
    n_h, n_w = patches_tensor.shape[:2]
    pil_list = []
    for y in range(n_h):
        for x in range(n_w):
            patch = patches_tensor[y, x]
            arr = (patch.cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            pil_list.append(Image.fromarray(arr))
    texts = [prompt]
    inputs = processor(text=texts, images=pil_list, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits_per_image  # shape: (num_images, 1)
        scores_flat = logits.squeeze(1).cpu().numpy()  # shape: (num_images,)

    scores = scores_flat.reshape(n_h, n_w)
    # normalize
    scores = np.clip(scores - scores.mean(), 0, None)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


def get_box(scores: np.ndarray, patch_size: int, threshold: float):
    """Compute bounding box from thresholded score map."""
    det = scores > threshold
    if not det.any():
        return None
    ys, xs = np.nonzero(det)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return (x0 * patch_size, y0 * patch_size,
            (x1 - x0) * patch_size, (y1 - y0) * patch_size)


def detect(prompt: str, img: torch.Tensor,
           model: CLIPModel, processor: CLIPProcessor,
           patch_size: int, threshold: float, device: str):
    """Detect prompt in image and display bounding box."""
    patches_tensor, n_h, n_w = get_patches(img.cpu(), patch_size)
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(img_np.shape[1] / 100, img_np.shape[0] / 100))
    ax.imshow(img_np); ax.axis('off')

    scores = get_scores_batched(patches_tensor, prompt, model, processor, device)
    box = get_box(scores, patch_size, threshold)
    if box:
        x, y, w, h = box
        rect = mpatches.Rectangle((x, y), w, h,
                                  edgecolor=colors[0], facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y - 5, prompt,
                color=colors[0], backgroundcolor='white', fontsize=10)
    else:
        print(f"No detection for '{prompt}'")
    plt.show()
