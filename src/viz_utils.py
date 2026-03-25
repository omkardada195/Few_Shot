import matplotlib.pyplot as plt
import numpy as np


def show_image_mask_pair(image, mask, title_prefix="Sample"):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"{title_prefix} Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.title(f"{title_prefix} Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_prediction_triplet(image, true_mask, pred_mask, title_prefix="Prediction"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"{title_prefix} Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.squeeze(), cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()