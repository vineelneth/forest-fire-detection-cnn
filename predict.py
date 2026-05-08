"""
Run fire detection inference on a single image.

Usage
-----
    python predict.py --image path/to/image.jpg        # direct path
    python predict.py                                  # opens a file-picker GUI
    python predict.py --model path/to/weights.pth --image photo.png
"""

import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import FireCNN

# Label convention matches training: fire=0, nofire=1.
# Model sigmoid output < 0.5 → fire; ≥ 0.5 → no fire.
LABELS = {True: ("Fire Detected", "red"), False: ("No Fire", "green")}


def load_model(weights_path: str) -> FireCNN:
    model = FireCNN()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model


def preprocess(img_path: str, img_size: int = 128) -> tuple[np.ndarray, torch.Tensor]:
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))
    tensor = torch.tensor(
        (resized / 255.0).transpose(2, 0, 1), dtype=torch.float32
    ).unsqueeze(0)
    return rgb, tensor


def predict(model: FireCNN, tensor: torch.Tensor) -> tuple[bool, float]:
    with torch.no_grad():
        confidence = model(tensor).item()
    is_fire = confidence < 0.5
    return is_fire, confidence


def show_result(display_img: np.ndarray, is_fire: bool, confidence: float):
    label, color = LABELS[is_fire]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(display_img)
    ax.set_title(label, fontsize=20, fontweight="bold", color=color, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    print(f"{label}  (model output: {confidence:.4f})")


def pick_image_gui() -> str:
    try:
        from tkinter import Tk, filedialog
    except ImportError:
        print("tkinter is not available. Pass --image <path> instead.")
        sys.exit(1)
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
    )
    root.destroy()
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Fire detection inference")
    parser.add_argument("--model", default="fire_model.pth",
                        help="Path to trained model weights (default: fire_model.pth)")
    parser.add_argument("--image", default=None,
                        help="Path to input image (omit to open a file-picker GUI)")
    return parser.parse_args()


def main():
    args = parse_args()

    img_path = args.image or pick_image_gui()
    if not img_path:
        print("No image selected.")
        sys.exit(0)

    model = load_model(args.model)
    display_img, tensor = preprocess(img_path)
    is_fire, confidence = predict(model, tensor)
    show_result(display_img, is_fire, confidence)


if __name__ == "__main__":
    main()
