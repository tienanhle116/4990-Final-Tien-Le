import argparse
import os
import cv2
import numpy as np
from skimage import measure
from scipy.ndimage import binary_closing

def refine_mask(mask_path, output_path, closing_kernel_size=5):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask at {mask_path}")

    # Threshold
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Morphological closing to fill holes
    closed = binary_closing(binary_mask / 255.0, structure=np.ones((closing_kernel_size, closing_kernel_size))).astype(np.uint8)

    # Keep largest connected component
    labeled = measure.label(closed, connectivity=2)
    props = measure.regionprops(labeled)
    if not props:
        print(f"No connected regions found in {mask_path}")
        return

    largest = max(props, key=lambda x: x.area).label
    refined = (labeled == largest).astype(np.uint8) * 255

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, refined)
    print(f"Refined mask saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input binary mask image")
    parser.add_argument("--output", required=True, help="Path to save refined mask")
    parser.add_argument("--kernel", type=int, default=5, help="Size of morphological closing kernel")
    args = parser.parse_args()

    refine_mask(args.input, args.output, args.kernel)
