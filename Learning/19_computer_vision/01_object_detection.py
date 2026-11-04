# =============================================================================
# 19_computer_vision / 01 — OBJECT DETECTION
# =============================================================================
#
# LIKE YOU ARE 5:
# Classification: "What is in this photo?" → "A cat."
# Detection: "Where is the cat in this photo?" → "Cat at [x=120, y=85, w=200, h=180]"
# Detection = Classification + Location (a bounding box around each object)
#
# C# ANALOGY:
# Classification returns a single int (class index).
# Detection returns a List<BoundingBox> where each box has:
#   { X, Y, Width, Height, ClassLabel, Confidence }
# The model must both find AND classify all objects in one pass.
#
# =============================================================================

import torch
import torch.nn as nn
import math

print("=" * 60)
print("PART 1: THE DETECTION PROBLEM")
print("=" * 60)
print("""
Input:  image tensor (3, H, W)
Output: list of bounding boxes, each with:
  [x_center, y_center, width, height, confidence, class_scores...]

Challenge: how many objects are there? Could be 0, 1, or 100.
  → Can't output a fixed-size tensor like in classification.
  → Solution: output a grid of CANDIDATE boxes, then filter.

Two families of detectors:
  Two-stage (R-CNN, Faster R-CNN):
    1. Propose regions of interest (RPN)
    2. Classify each proposal
    Pros: very accurate  Cons: slow (two networks)

  One-stage (YOLO, SSD, DETR):
    1. Single forward pass → all boxes at once
    Pros: fast  Cons: historically less accurate (closed the gap recently)

Modern state of the art: YOLO (v8-v11), RT-DETR (transformer-based)
""")

print("=" * 60)
print("PART 2: BOUNDING BOXES AND FORMATS")
print("=" * 60)
print("""
Bounding box formats:
  [x_min, y_min, x_max, y_max]   ← corner format (most common in datasets)
  [x_center, y_center, w, h]     ← YOLO format
  [x_min, y_min, w, h]           ← COCO format

Coordinates can be:
  Absolute pixels: x_min=120, y_min=85  (depends on image size)
  Relative [0, 1]: x_min=0.234, y_min=0.166  (independent of size)
  YOLO uses relative [0, 1] for all coordinates.
""")

def xyxy_to_xywh(box):
    """Convert corner format to center format."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return [cx, cy, w, h]

def xywh_to_xyxy(box):
    """Convert center format to corner format."""
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

box_corner = [120, 85, 320, 265]   # a cat bounding box in pixels
box_center = xyxy_to_xywh(box_corner)
print(f"Corner format [x1,y1,x2,y2]: {box_corner}")
print(f"Center format [cx,cy,w,h]:   {[int(v) for v in box_center]}")
print(f"Back to corner: {[int(v) for v in xywh_to_xyxy(box_center)]}")

print()
print("=" * 60)
print("PART 3: IoU — HOW SIMILAR ARE TWO BOXES?")
print("=" * 60)
print("""
IoU (Intersection over Union):
  IoU = Area of Overlap / Area of Union

  IoU = 0.0: boxes don't overlap at all
  IoU = 1.0: boxes are identical

Used for:
  - Evaluating detection quality (mAP metric)
  - Non-Maximum Suppression (NMS) — remove duplicate detections
  - Anchor matching — which anchor should predict which object
""")

def iou(box1, box2):
    """
    Compute IoU for two boxes in [x1, y1, x2, y2] format.
    box1, box2: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union  = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

box_a = [100, 100, 300, 300]   # 200×200 box
box_b = [200, 200, 400, 400]   # same size, offset by 100,100
box_c = [500, 500, 700, 700]   # completely separate

print(f"IoU(a, a) = {iou(box_a, box_a):.3f}  (identical → 1.0)")
print(f"IoU(a, b) = {iou(box_a, box_b):.3f}  (partial overlap)")
print(f"IoU(a, c) = {iou(box_a, box_c):.3f}  (no overlap → 0.0)")

print()
print("=" * 60)
print("PART 4: NON-MAXIMUM SUPPRESSION (NMS)")
print("=" * 60)
print("""
Problem: a detector outputs many overlapping boxes for the same object.
  "98% cat at [100,100,300,300]"
  "95% cat at [105,105,295,295]"  ← almost same box
  "92% cat at [110,100,305,310]"  ← almost same box

We want ONE box per object, not many.

NMS algorithm:
  1. Sort boxes by confidence (highest first)
  2. Take the top-confidence box → it's a detection
  3. Remove all boxes with IoU > threshold (0.45) with that box
  4. Repeat with the next remaining box
""")

def nms(boxes, scores, iou_threshold=0.45):
    """
    Non-Maximum Suppression.
    boxes:  list of [x1, y1, x2, y2] in pixel coords
    scores: list of confidence values
    Returns: indices of kept boxes
    """
    if not boxes:
        return []

    # Sort by score descending
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept  = []

    while order:
        i = order.pop(0)   # highest score
        kept.append(i)
        # Remove boxes with high IoU with box i
        order = [j for j in order if iou(boxes[i], boxes[j]) <= iou_threshold]

    return kept

# Demo: 4 boxes, 3 for the same object
boxes_raw = [
    [100, 100, 300, 300],   # cat A
    [105, 103, 298, 302],   # cat A (almost identical)
    [108, 100, 305, 305],   # cat A (slightly different)
    [400, 150, 600, 350],   # cat B (different location)
]
scores_raw = [0.98, 0.95, 0.92, 0.88]

kept = nms(boxes_raw, scores_raw, iou_threshold=0.45)
print(f"Before NMS: {len(boxes_raw)} boxes")
print(f"After NMS:  {len(kept)} boxes (indices {kept})")
for idx in kept:
    print(f"  box {idx}: score={scores_raw[idx]:.2f}, coords={boxes_raw[idx]}")

print()
print("=" * 60)
print("PART 5: YOLO ARCHITECTURE (concept)")
print("=" * 60)
print("""
YOLO (You Only Look Once) predicts all boxes in a single forward pass.

How it works (simplified):
  1. Divide image into an S×S grid (e.g., 20×20 for 640×640 input)
  2. Each grid cell predicts B boxes (B=3 at each scale in YOLOv8)
  3. Each box: [x, y, w, h, confidence] + C class scores
  4. Output tensor: (S, S, B × (5 + C))

Multi-scale detection:
  Detect at 3 scales: 80×80 (small objects), 40×40 (medium), 20×20 (large)
  Each scale uses the feature pyramid (different FPN levels)

Training:
  For each ground-truth box: assign to best-matching anchor
  Loss = box loss (IoU-based) + confidence loss + class loss
""")

# Minimal YOLO-style head (just the concept)
class TinyYOLOHead(nn.Module):
    """
    One YOLO detection head for one scale.
    In a real YOLO this runs at 3 scales (after FPN).
    """
    def __init__(self, in_channels, num_anchors=3, num_classes=80):
        super().__init__()
        # Predict: (x, y, w, h, conf) + num_classes per anchor
        out_channels = num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

head = TinyYOLOHead(in_channels=256, num_anchors=3, num_classes=80)
feat = torch.randn(1, 256, 20, 20)   # backbone feature at 20×20 scale
out  = head(feat)
print(f"YOLO head input:  {feat.shape}  (1, 256, 20, 20)")
print(f"YOLO head output: {out.shape}   (1, 3×(5+80), 20, 20) = 1×255×20×20")
print(f"Each 20×20 cell predicts 3 boxes × 85 values = 255 channels")

print()
print("=" * 60)
print("PART 6: USING YOLO IN PRACTICE")
print("=" * 60)
print("""
In practice, don't implement YOLO from scratch.
Use ultralytics (the official YOLOv8-v11 library):

  pip install ultralytics

  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")   # nano (fastest), or s/m/l/x (larger)

  # Inference on an image
  results = model("photo.jpg")
  for r in results:
      boxes      = r.boxes.xyxy   # [x1,y1,x2,y2] for each detection
      confidences= r.boxes.conf   # confidence per box
      classes    = r.boxes.cls    # class index per box
      print(r.summary())

  # Fine-tune on custom data:
  model.train(data="my_dataset.yaml", epochs=100, imgsz=640)

DETR (transformer-based detection):
  from transformers import DetrForObjectDetection
  model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
  → No anchor boxes, no NMS needed; uses set prediction loss
""")

print("=" * 60)
print("QUICK REFERENCE: DETECTION METRICS")
print("=" * 60)
print("""
Precision:  of all my detections, what fraction was correct?
            = TP / (TP + FP)

Recall:     of all true objects, what fraction did I find?
            = TP / (TP + FN)

AP (Average Precision):  area under the precision-recall curve at one IoU threshold
mAP (mean AP):           average AP across all classes
mAP@0.5:   AP at IoU threshold 0.5 (a detected box must overlap ≥ 50% with ground truth)
mAP@0.5:0.95: average over IoU thresholds 0.5, 0.55, ..., 0.95 (strict metric)

Good numbers on COCO (2017 val):
  YOLOv8n:   mAP@0.5:0.95 = 37.3
  YOLOv8x:   mAP@0.5:0.95 = 53.9
  DETR-R50:  mAP@0.5:0.95 = 42.0
""")

# =============================================================================
# EXPECTED OUTPUT:
# Bounding box conversions correct (reversible)
# IoU(a,a)=1.0, IoU(a,c)=0.0
# NMS keeps 2 boxes from 4 (removes duplicates for same object)
# YOLO head output: torch.Size([1, 255, 20, 20])
# =============================================================================
