"""Small helpers: reproducibility, device, dataset counting, architecture visuals."""
import os
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary as torchinfo_summary


def set_seed(seed=42):
    """consistent python, numpy and pytorch's random functions
    and pytorch's random functions on GPUs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    #use GPU if available
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_images(root): # root = directory
    counts = {}
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls) #join() is used to combine pieces of a file/folder path
        if os.path.isdir(cls_dir):
            counts[cls] = len([f for f in os.listdir(cls_dir)
                                if f.lower().endswith(".jpg")])
    return counts


def draw_architecture_diagram(blocks, title, figsize_width=6.5):
    """draws a top-to-bottom block diagram from a list of text labels.
     Used for all three models' architecture overview."""
    n = len(blocks)
    box_h, gap = 0.7, 0.35
    total_h = n * (box_h + gap)
    fig, ax = plt.subplots(figsize=(figsize_width, max(3, n * 0.85)))
    for i, label in enumerate(blocks):

        y = total_h - i * (box_h + gap) - box_h
        rect = mpatches.FancyBboxPatch((0.05, y), 0.9, box_h, boxstyle="round,pad=0.02",
                                        linewidth=1.5, edgecolor="#333333", facecolor="#cfe8ff")
        ax.add_patch(rect)
        ax.text(0.5, y + box_h / 2, label, ha="center", va="center", fontsize=9, wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(0.5, y - 0.03), xytext=(0.5, y),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h + 0.2)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_model_summary(model, input_size, depth=2, show_trainable=False):
    """Layer-by-layer torchinfo summary (shapes, parameter counts).
    input_size is (batch, channels, height, width), e.g. (1, 3, 224, 224)."""
    col_names = ["output_size", "num_params"]
    if show_trainable:
        col_names.append("trainable")  # makes a frozen backbone visible
    print(torchinfo_summary(model, input_size=input_size, col_names=col_names,
                            depth=depth, verbose=0))
