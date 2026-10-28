"""
Step 3 – Full Evaluation + Failure Case Analysis
=================================================
Runs best.pt on the entire test set, computes per-class metrics,
then saves annotated images of failure cases (FP + FN).
"""

import cv2, json, os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
BEST_PT    = BASE / "runs" / "yolov8n_320" / "weights" / "best.pt"
METAL_NUT  = BASE / "metal_nut"
LABEL_DIR  = BASE / "labels"
OUT_DIR    = BASE / "eval_output"
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "failures").mkdir(exist_ok=True)

CLASSES    = ["bent", "color", "flip", "scratch"]
IOU_THRESH = 0.50
CONF_THRESH= 0.25   # same as YOLO default

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt(label_path):
    """Load YOLO label txt → list of [cls, x1, y1, x2, y2] in pixel coords for given img size."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            boxes.append([int(cls), xc, yc, w, h])   # normalised YOLO format
    return boxes

def yolo_to_xyxy(box, W, H):
    cls, xc, yc, w, h = box
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    return [cls, x1, y1, x2, y2]

def iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def draw_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
    cv2.putText(img, label, (x1+2, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

# ── Evaluate ─────────────────────────────────────────────────────────────────

def main():
    model = YOLO(str(BEST_PT))

    # Per-class accumulators
    stats = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in CLASSES}
    failures = []   # (img_path, gt_boxes, pred_boxes, fail_type, reason)

    print("Running inference on full test set...\n")

    for cls_name in CLASSES:
        cls_id    = CLASSES.index(cls_name)
        img_dir   = METAL_NUT / "test" / cls_name
        lbl_dir   = LABEL_DIR / cls_name

        for img_path in sorted(img_dir.glob("*.png")):
            stem  = img_path.stem
            gt_raw = load_gt(lbl_dir / f"{stem}.txt")

            img   = cv2.imread(str(img_path))
            H, W  = img.shape[:2]

            gt_boxes = [yolo_to_xyxy(b, W, H) for b in gt_raw]  # [cls,x1,y1,x2,y2]

            # Inference
            results  = model.predict(img_path, conf=CONF_THRESH,
                                     imgsz=320, verbose=False)[0]
            pred_boxes = []
            for box in results.boxes:
                pc   = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                pred_boxes.append([pc, x1, y1, x2, y2, conf])

            # Match GT ↔ pred by IoU ≥ 0.5 & same class
            gt_matched   = [False] * len(gt_boxes)
            pred_matched = [False] * len(pred_boxes)

            for pi, pb in enumerate(pred_boxes):
                best_iou = 0; best_gi = -1
                for gi, gb in enumerate(gt_boxes):
                    if gt_matched[gi]: continue
                    if pb[0] != gb[0]: continue   # class must match
                    v = iou(pb[1:5], gb[1:5])
                    if v > best_iou:
                        best_iou = v; best_gi = gi
                if best_iou >= IOU_THRESH:
                    gt_matched[best_gi]  = True
                    pred_matched[pi]     = True
                    stats[cls_name]["tp"] += 1
                else:
                    stats[cls_name]["fp"] += 1

            fn_count = gt_matched.count(False)
            stats[cls_name]["fn"] += fn_count

            # Collect failure cases
            fp_preds = [pb for pi,pb in enumerate(pred_boxes) if not pred_matched[pi]]
            fn_gts   = [gb for gi,gb in enumerate(gt_boxes)   if not gt_matched[gi]]

            if fp_preds or fn_gts:
                failures.append({
                    "img_path":  img_path,
                    "cls_name":  cls_name,
                    "gt_boxes":  gt_boxes,
                    "pred_boxes":pred_boxes,
                    "fp_preds":  fp_preds,
                    "fn_gts":    fn_gts,
                })

    # Also run on good test images (should produce zero detections)
    for img_path in sorted((METAL_NUT / "test" / "good").glob("*.png")):
        results = model.predict(img_path, conf=CONF_THRESH, imgsz=320, verbose=False)[0]
        pred_boxes = []
        for box in results.boxes:
            pc = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            pred_boxes.append([pc, x1, y1, x2, y2, conf])
        if pred_boxes:
            stats_key = CLASSES[pred_boxes[0][0]] if pred_boxes else "bent"
            failures.append({
                "img_path": img_path,
                "cls_name": "good",
                "gt_boxes": [],
                "pred_boxes": pred_boxes,
                "fp_preds": pred_boxes,
                "fn_gts": [],
            })
            # FP on good images → add to all classes FP equally
            for pb in pred_boxes:
                stats[CLASSES[pb[0]]]["fp"] += 1

    # ── Print metrics ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Class':<10} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"{'-'*65}")
    all_tp = all_fp = all_fn = 0
    for cls_name in CLASSES:
        s  = stats[cls_name]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        all_tp += tp; all_fp += fp; all_fn += fn
        P  = tp/(tp+fp) if (tp+fp) > 0 else 0
        R  = tp/(tp+fn) if (tp+fn) > 0 else 0
        F1 = 2*P*R/(P+R) if (P+R) > 0 else 0
        print(f"{cls_name:<10} {tp:>5} {fp:>5} {fn:>5} {P:>10.3f} {R:>8.3f} {F1:>8.3f}")

    print(f"{'-'*65}")
    P_all  = all_tp/(all_tp+all_fp) if (all_tp+all_fp) > 0 else 0
    R_all  = all_tp/(all_tp+all_fn) if (all_tp+all_fn) > 0 else 0
    F1_all = 2*P_all*R_all/(P_all+R_all) if (P_all+R_all) > 0 else 0
    print(f"{'ALL':<10} {all_tp:>5} {all_fp:>5} {all_fn:>5} {P_all:>10.3f} {R_all:>8.3f} {F1_all:>8.3f}")
    print(f"\n(mAP@0.5 from training val: 0.906 — see runs/yolov8n_320/)")
    print(f"\nTotal failure cases collected: {len(failures)}")

    # ── Save failure case images ──────────────────────────────────────────────
    print(f"\nSaving failure case visualisations...")

    saved = 0
    for fail in failures[:15]:   # save up to 15
        img = cv2.imread(str(fail["img_path"]))
        H, W = img.shape[:2]
        cls_name = fail["cls_name"]

        # Draw all GT in green
        for gb in fail["gt_boxes"]:
            _, x1,y1,x2,y2 = gb
            draw_box(img, x1,y1,x2,y2, f"GT:{CLASSES[gb[0]]}", (0,200,0))

        # Draw TP predictions in blue
        for pb in fail["pred_boxes"]:
            pc,x1,y1,x2,y2,conf = pb
            draw_box(img, x1,y1,x2,y2,
                     f"PRED:{CLASSES[pc]} {conf:.2f}", (200,100,0))

        # Annotate FP in red
        for pb in fail["fp_preds"]:
            pc,x1,y1,x2,y2,conf = pb
            draw_box(img, x1,y1,x2,y2,
                     f"FP:{CLASSES[pc]} {conf:.2f}", (0,0,220))

        # Annotate FN in orange
        for gb in fail["fn_gts"]:
            _, x1,y1,x2,y2 = gb
            draw_box(img, x1,y1,x2,y2,
                     f"MISS:{CLASSES[gb[0]]}", (0,140,255))

        # Determine failure label for filename
        if fail["fn_gts"] and fail["fp_preds"]:
            ftype = "FP_FN"
        elif fail["fn_gts"]:
            ftype = "FN"
        else:
            ftype = "FP"

        out_name = f"{saved:02d}_{cls_name}_{fail['img_path'].stem}_{ftype}.jpg"
        out_path = OUT_DIR / "failures" / out_name
        cv2.imwrite(str(out_path), img)
        saved += 1
        print(f"  Saved: {out_name}")

    print(f"\nAll output saved to: {OUT_DIR}")
    print(f"Failure images:      {OUT_DIR/'failures'}")

if __name__ == "__main__":
    main()
