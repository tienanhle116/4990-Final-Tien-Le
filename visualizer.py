import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask(image_path, mask_path, alpha=0.5, color=(0, 255, 0)):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise FileNotFoundError("Image or mask not found.")

    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = color

    overlayed = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlayed

def show_image_with_mask(image_path, mask_path, alpha=0.5, color=(0, 255, 0)):
    overlayed = overlay_mask(image_path, mask_path, alpha, color)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image with Mask Overlay")
    plt.show()
