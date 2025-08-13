#!/usr/bin/env python3
import argparse
import time
import psutil
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor

# Import ONLY what you need from your model.py
# (Avoid wildcard imports so it's obvious what comes from where.)
from model import detect  # add other functions if you actually use them


def load_clip(quantize: bool = True, device: str = "cpu"):
   
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    model.to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def load_image(image_path: Path, resize_hw=(512, 512), device: str = "cpu"):
    
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(resize_hw),
        T.ToTensor(),
    ])
    tensor_img = transform(image).to(device)
    return tensor_img


def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-shot object detection with CLIP ViT-B/32"
    )
    p.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to input image (e.g., examples/sample.jpg)",
    )
    p.add_argument(
        "--prompt",
        required=True,
        help='Text prompt to detect (e.g., "sharp object")',
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=124,
        help="Tile/patch size used by your detect() for localization (default: 124)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Detection threshold passed to detect() (default: 0.6)",
    )
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable dynamic quantization (enabled by default)",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(512, 512),
        help="Resize H W before tiling (default: 512 512)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Force CPU because dynamic quantization is CPU-only.
    device = "cpu"

    # Load model/processor
    model, processor = load_clip(quantize=not args.no_quantize, device=device)

    # Load & preprocess image
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    tensor_img = load_image(args.image, resize_hw=tuple(args.resize), device=device)

    # Measure memory & time
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1e6
    start = time.time()

    # Call your implementation from model.py
    detect(
        prompt=args.prompt,
        img=tensor_img,
        model=model,
        processor=processor,
        patch_size=args.patch_size,
        threshold=args.threshold,
        device=device,
    )

    latency_ms = (time.time() - start) * 1000
    mem_after = proc.memory_info().rss / 1e6
    print(f"Latency: {latency_ms:.1f} ms, ΔRAM: {mem_after - mem_before:.1f} MB")


if __name__ == "__main__":
    main()
