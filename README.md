# Gun Detection — YOLOv11s

Real-time firearm detection system trained on surveillance and general gun imagery using YOLOv11s fine-tuned on multiple datasets.

---

## Model

| Attribute | Value |
|-----------|-------|
| Architecture | YOLOv11s (small) |
| Parameters | 9.4M |
| FLOPs | 21.3 GFLOPs |
| Input size | 640×640 |
| Classes | 1 (`gun`) |
| Inference speed | ~2.5ms (T4 GPU) |
| Pretrained weights | COCO (Ultralytics) |

YOLOv11s was chosen over the larger YOLOv11l (25.3M params) to prioritize inference speed for real-time deployment, as the single-class nature of the task reduces the accuracy gap between model sizes [1].

---

## Training Infrastructure

- **Platform:** [Modal.com](https://modal.com) serverless GPU
- **GPU:** NVIDIA H100 80GB HBM3
- **Framework:** Ultralytics YOLOv11 [1]

---

## Datasets

Two publicly available datasets were combined for the final model. All annotations were remapped to a single `gun` class. A clean 80/10/10 train/val/test split was applied across the merged pool after MD5-based cross-dataset deduplication to prevent data leakage.

### 1. CCTV Gun Detector (Anvari)
- **Source:** Roboflow Universe — [mohammadreza-anvari-h6ase/cctv-gun-detector](https://universe.roboflow.com/mohammadreza-anvari-h6ase/cctv-gun-detector)
- **Images:** ~1,009
- **Classes:** Guns, Guns perspective, Long guns → remapped to `gun`
- **License:** CC BY 4.0
- **Role:** Security camera domain adaptation. Oversampled 5× in training to compensate for class imbalance against larger datasets.

### 2. The Monash Guns Dataset v2
- **Source:** Roboflow Universe — [arms/the-monash-guns-dataset](https://universe.roboflow.com/arms/the-monash-guns-dataset/dataset/2)
- **Images:** ~7,780
- **Classes:** gun
- **License:** CC BY 4.0
- **Role:** Academic/security research dataset providing diverse real-world gun imagery

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 150 (early stopping) |
| Batch size | 128 |
| Image size | 640 |
| Optimizer | AdamW |
| Learning rate (lr0) | 0.0001 |
| LR final (lrf) | 0.01 |
| LR schedule | Cosine annealing |
| Warmup epochs | 3 |
| Patience (early stop) | 25 |
| Mosaic augmentation | 1.0 |
| Mixup | 0.1 |
| Horizontal flip | 0.5 |

The low learning rate (0.0001) was chosen for fine-tuning from COCO pretrained weights to preserve learned feature representations while adapting to the gun detection domain, consistent with practices in transfer learning for object detection [2][3].

---

## Training Procedure

1. **Initial training (gun-v2)** — YOLOv11s trained from COCO pretrained weights on muha/gun-soqmi v2 (~20K images). Early stopping triggered at epoch 122 out of 200.
2. **CCTV fine-tuning (gun-v2-cctv)** — Model fine-tuned on Anvari CCTV dataset alone (lr=0.00005) to adapt to security camera domain. Early stopping at epoch 58.
3. **Final combined training (gun-v3)** — Model retrained from gun-v2 weights on merged Anvari + Monash datasets (~8.8K images) with Anvari oversampled 5× in training. This is the deployed model.

---

## Results

Evaluated on held-out validation set (CCTV domain):

| Metric | Value |
|--------|-------|
| mAP50 | 0.984 |
| mAP50-95 | 0.692 |
| Precision | 0.995 |
| Recall | 0.980 |

---

## Inference

```bash
pip install ultralytics
python detect_video.py path/to/video.mp4
```

`detect_video.py` runs inference with `conf=0.6` and `imgsz=1280` (higher resolution improves small object detection on security camera footage). Output video is saved to `runs/detect/`.

### Using a specific weights file

By default the script loads `best.pt` from the current directory. To test a different checkpoint, edit the first line of `detect_video.py`:

```python
model = YOLO("best.pt")        # default
model = YOLO("best_v4.pt")     # a specific version
model = YOLO("test_midrun.pt") # a mid-run checkpoint downloaded during training
```

### Downloading checkpoints mid-run

Checkpoints are saved to the Modal volume every 5 epochs during training. To grab the latest checkpoint without stopping the run:

```bash
# Most recent epoch
modal volume get gun-detect-v4-runs gun-v4/weights/last.pt test_midrun.pt

# Best so far
modal volume get gun-detect-v4-runs gun-v4/weights/best.pt test_best.pt
```

Use `--force` if the file already exists locally:

```bash
modal volume get --force gun-detect-v4-runs gun-v4/weights/last.pt test_midrun.pt
```

---

## Limitations

- Detection performance degrades on **small or distant guns** in wide-angle security footage
- Model is **pistol-biased** due to dataset composition — long guns (rifles, shotguns) at security camera angles are underrepresented
- Validated on a small CCTV val set (202 images) — real-world performance may differ
- Not tested on night-vision or infrared footage

---

## Related Work

The following studies informed design decisions around dataset construction, preprocessing, and evaluation methodology for gun detection in surveillance contexts:

[1] Redmon, J. et al. — Ultralytics YOLOv11. https://github.com/ultralytics/ultralytics

[2] Olmos, R., Tabik, S., & Herrera, F. (2018). *Automatic handgun detection alarm in videos using deep learning.* Neurocomputing, 275, 66–72. https://doi.org/10.1016/j.neucom.2017.05.044

[3] Grega, M., Matiolański, A., Guzik, P., & Leszczuk, M. (2018). *Automated detection of firearms and knives in a CCTV image stream.* Neurocomputing, 330, 333–344. https://doi.org/10.1016/j.neucom.2018.09.080

[4] Castillo-Camacho, I., & Wang, K. (2019). *A comprehensive review of deep-learning-based methods for image forensics.* Information Fusion, 46, 327–347. https://doi.org/10.1016/j.inffus.2018.06.005

[5] Bhatt, D., et al. (2021). *CNN variants for computer vision: History, architecture, application, challenges and future scope.* Knowledge-Based Systems, 188, 105042. https://doi.org/10.1016/j.knosys.2020.105042
