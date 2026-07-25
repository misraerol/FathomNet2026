# FathomNet 2026 — Marine Species Object Detection

Object detection on deep-sea survey imagery for the [FathomNet 2026 Kaggle competition](https://www.kaggle.com/competitions/fathomnet-2026) (part of LifeCLEF / CVPR-FGVC). The task is framed as **Positive-Unlabeled (PU) detection**: in underwater survey images, annotators label only the species relevant to their survey, so many visible organisms are left unmarked. A naive detector treats those unlabeled regions as background, which is exactly the problem this project explores.

Fine-tuned **YOLOv8s** on 6,439 images across 32 species classes, then analyzed how heavy class imbalance drives per-class performance.

## Results

Evaluated on the validation split with `model.val()`:

| Metric | Value |
|---|---|
| mAP@0.5 | 0.484 |
| mAP@0.5:0.95 | 0.353 |
| Precision | 0.732 |
| Recall | 0.419 |

The model is conservative: when it fires, it's right ~73% of the time, but it finds fewer than half of all true objects. mAP50 was still climbing at 30 epochs, so it hadn't hit its ceiling.

## The interesting part: instances, not visual difficulty

Per-class accuracy tracks the number of labeled instances almost perfectly — not how hard the species is to see:

| Class | Instances | AP50 |
|---|---|---|
| urchin | 1,437 | 0.884 |
| bony fish | 589 | 0.747 |
| sea fan | 623 | 0.641 |
| ... | ... | ... |
| benthic worm | 9 | 0.001 |
| isopod | 2 | 0.000 |
| sea slug | 2 | 0.000 |

Classes with 2 instances score 0.00. This isn't a bug — it's the core reason the competition is framed as Positive-Unlabeled rather than standard supervised detection. Rare classes here aren't rare in nature; they're the ones most likely to be *present but unlabeled*, and the model can't tell "genuinely rare" apart from "under-labeled."

## Dataset

- 6,463 listed images → 6,439 usable (24 broken links removed)
- 22,186 validated annotations, 32 species classes
- Every bounding box checked for required fields, positive width/height, and in-bounds coordinates (source data was clean — zero corrupt boxes)
- 80/20 train/val split, fixed seed (`random.seed(42)`) for reproducibility
- COCO format converted to YOLO normalized center-coordinates; `data.yaml` generated with all 32 class names

## Approach

- **Model:** YOLOv8s, COCO-pretrained, fine-tuned on the marine imagery. The "small" variant balanced speed and accuracy on a single Kaggle Tesla T4.
- **Training:** 30 epochs, imgsz 512, batch 32, RAM caching enabled.
- **A tuning lesson:** comparing `lr0=0.01` vs `0.001` gave near-identical results — Ultralytics' `optimizer='auto'` silently ignores manual `lr0` and computed its own (AdamW, lr ≈ 0.00028). Worth verifying what a hyperparameter actually resolves to before drawing conclusions.
- **Throughput:** switching from `batch=16, imgsz=640` to `batch=32, imgsz=512, cache=True` cut epoch time ~7x (7.3 min → 1.0 min), bringing a full 30-epoch run down from an estimated ~3.5 hours to ~30 minutes.
- **Augmentation:** default Ultralytics mosaic + horizontal flip. Vertical flip disabled — seafloor imagery has no meaningful up/down orientation.

## Demo

`app.py` is a Streamlit app built around the fine-tuned `best.pt` weights: upload any image, run inference, and see the annotated result with detected species, confidence scores, and an adjustable confidence threshold.

```bash
pip install -r requirements.txt
streamlit run app.py
```

On a previously unseen image, the model localized and classified a bony fish at 0.93 confidence — consistent with bony fish being well-represented in training (589 instances, AP50 0.747).

## Next steps

- Implement a PU-specific training strategy (confidence-gated self-training or non-negative risk-weighted loss) to address the rare-class failures directly.
- Train beyond 30 epochs, since mAP50 hadn't plateaued.
- Targeted oversampling / augmentation for the lowest-instance classes.
- Benchmark YOLOv8 against RT-DETR or Faster R-CNN to validate the architecture choice empirically.

## Environment

Python 3.12, PyTorch 2.10 (CUDA 12.8), Ultralytics 8.4. Trained on Kaggle Notebooks (Tesla T4, free tier).

## Files

```
app.py               # Streamlit demo
best.pt              # fine-tuned YOLOv8s weights
FathomNet.ipynb      # training + evaluation notebook
fathomnet-2026/      # data config / working files
requirements.txt
```
