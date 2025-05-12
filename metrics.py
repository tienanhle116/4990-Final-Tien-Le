import numpy as np
from skimage.metrics import adapted_rand_error as dice_score
from sklearn.metrics import jaccard_score

def compute_iou(pred_mask, true_mask):
    pred = pred_mask.flatten() > 0
    truth = true_mask.flatten() > 0
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_dice(pred_mask, true_mask):
    pred = pred_mask.flatten() > 0
    truth = true_mask.flatten() > 0
    intersection = np.sum(pred & truth)
    return 2. * intersection / (np.sum(pred) + np.sum(truth))

def compute_precision_recall(pred_mask, true_mask):
    pred = pred_mask.flatten() > 0
    truth = true_mask.flatten() > 0
    tp = np.sum(pred & truth)
    fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision, recall

def evaluate_mask(pred_path, gt_path):
    import cv2
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        raise FileNotFoundError("Prediction or ground truth image not found.")

    iou = compute_iou(pred, gt)
    dice = compute_dice(pred, gt)
    precision, recall = compute_precision_recall(pred, gt)

    return {
        "IoU": round(iou, 4),
        "Dice": round(dice, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }
