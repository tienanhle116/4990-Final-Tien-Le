# Segment Anything Enhanced: Evaluation and Extensions

This project builds on Meta AI's Segment Anything Model (SAM) by evaluating its performance across different domains and proposing enhancements for accuracy, speed, and usability.

## 🔍 Project Overview

- Evaluates **SAM**, **FastSAM**, and **MedSAM** on COCO, ADE20K, and medical CT scan images.
- Adds a lightweight **post-processing pipeline** to improve mask quality.
- Tests **prompt auto-suggestion** using saliency maps.
- Measures accuracy (IoU, AP) and efficiency (latency) across all models.

## 📁 Folder Structure

```
segment-anything-enhanced/
├── README.md
├── requirements.txt
├── notebooks/
├── scripts/
├── utils/
├── data/           # Sample data (images & GT masks)
└── outputs/        # Output masks from models
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure `torch` is installed with GPU support if available.

### 2. Run Evaluations

```bash
python scripts/run_sam.py
python scripts/run_fastsam.py
python scripts/run_medsam.py
```

To refine masks:
```bash
python scripts/refine_masks.py
```

## 📊 Results Summary

| Model        | Dataset  | mIoU (%) | AP@0.5 | Latency (ms) |
|--------------|----------|----------|--------|---------------|
| SAM          | COCO     | 81.6     | 89.4   | 950           |
| FastSAM      | COCO     | 84.9     | 91.2   | 210           |
| SAM+Refine   | COCO     | 85.3     | 92.0   | 980           |
| MedSAM       | MedSet   | 82.4     | 88.9   | 960           |



## 📚 Acknowledgments

This project builds on:
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

## 🧑‍💻 Author

Tien Le (tle1@cpp.edu)
