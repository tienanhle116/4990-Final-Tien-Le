import argparse
import cv2
import torch
import numpy as np
import os
from segment_anything import SamPredictor, sam_model_registry

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_mask(mask, output_path):
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask)

def main(image_path, output_path, model_type='vit_b', checkpoint='sam_vit_b.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    predictor = SamPredictor(sam)

    image = load_image(image_path)
    predictor.set_image(image)

    # Example point prompt
    input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # Save best mask
    best_idx = np.argmax(scores)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_mask(masks[best_idx], output_path)
    print(f"Mask saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save mask")
    parser.add_argument("--checkpoint", type=str, default="sam_vit_b.pth", help="Path to SAM checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM model type (vit_b, vit_l, vit_h)")
    args = parser.parse_args()

    main(args.image, args.output, args.model_type, args.checkpoint)
