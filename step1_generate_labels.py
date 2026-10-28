"""
Step 1 – Mask → YOLO Bounding-Box Labels
=========================================
Converts MVTec AD binary pixel masks into YOLO-format label files.

Strategy
--------
* Each *_mask.png is a grayscale image where white pixels (>0) mark the defect.
* We use OpenCV connected-components so that if one image has TWO separate
  defect blobs we get TWO bounding boxes (rather than one huge box wrapping
  empty space between them).
* Very tiny components (< MIN_AREA_FRAC of the image) are ignored as noise.
* For the 'flip' class the mask typically covers most of the nut (the whole
  nut is flipped), so a single large box is expected and correct.

Output directory layout (YOLO convention):
  labels/
    bent/       <- .txt files alongside test/bent/ images
    color/
    flip/
    scratch/
    good_train/ <- empty .txt files (no defect = no boxes)
    good_test/

YOLO label format (one line per box):
  <class_id>  <x_center>  <y_center>  <width>  <height>
  all values normalised to [0, 1] relative to image W x H.

Class map: bent=0, color=1, flip=2, scratch=3
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent / "metal_nut"
GT_DIR    = BASE / "ground_truth"
TEST_DIR  = BASE / "test"
TRAIN_DIR = BASE / "train"
LABEL_DIR = Path(__file__).parent / "labels"

CLASS_MAP = {"bent": 0, "color": 1, "flip": 2, "scratch": 3}

# Ignore blobs smaller than this fraction of total pixels
MIN_AREA_FRAC = 0.0005   # 0.05 % of image area

# ── Helpers ───────────────────────────────────────────────────────────────────

def mask_to_bboxes(mask_path, class_id):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    H, W = mask.shape
    min_area = MIN_AREA_FRAC * H * W

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    rows = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        x_c = (x + bw / 2) / W
        y_c = (y + bh / 2) / H
        w_n = bw / W
        h_n = bh / H

        rows.append((class_id, x_c, y_c, w_n, h_n))

    return rows


def write_label(label_path, rows):
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    stats_table = {}

    print("Converting masks to YOLO bounding-box labels...")
    print()

    # 1. Defective classes
    for class_name, class_id in CLASS_MAP.items():
        gt_class_dir  = GT_DIR  / class_name
        out_label_dir = LABEL_DIR / class_name

        mask_files = sorted(gt_class_dir.glob("*_mask.png"))
        if not mask_files:
            print(f"[WARN] No masks found in {gt_class_dir}")
            continue

        box_count = 0
        for mask_path in mask_files:
            stem = mask_path.stem.replace("_mask", "")
            label_path = out_label_dir / f"{stem}.txt"
            rows = mask_to_bboxes(mask_path, class_id)
            write_label(label_path, rows)
            box_count += len(rows)

        stats_table[class_name] = {"images": len(mask_files), "boxes": box_count}
        print(f"  [{class_name:8s}]  {len(mask_files):3d} images  ->  {box_count:3d} boxes  "
              f"(avg {box_count/len(mask_files):.1f} box/img)")

    # 2. Good train images -> empty labels
    good_train_dir  = TRAIN_DIR / "good"
    good_label_dir  = LABEL_DIR / "good_train"
    good_imgs = sorted(good_train_dir.glob("*.png"))
    for img_path in good_imgs:
        write_label(good_label_dir / f"{img_path.stem}.txt", [])
    print(f"  [good_train]  {len(good_imgs)} empty label files")

    # 3. Good test images -> empty labels
    good_test_dir   = TEST_DIR / "good"
    good_test_label = LABEL_DIR / "good_test"
    good_test_imgs  = sorted(good_test_dir.glob("*.png"))
    for img_path in good_test_imgs:
        write_label(good_test_label / f"{img_path.stem}.txt", [])
    print(f"  [good_test ]  {len(good_test_imgs)} empty label files")

    print()
    print("=" * 55)
    print(f"{'Class':<12} {'Images':>8} {'Boxes':>8} {'Avg':>10}")
    print("-" * 55)
    total_imgs = total_boxes = 0
    for cls, s in stats_table.items():
        avg = s["boxes"] / s["images"] if s["images"] else 0
        print(f"{cls:<12} {s['images']:>8} {s['boxes']:>8} {avg:>10.2f}")
        total_imgs  += s["images"]
        total_boxes += s["boxes"]
    print("-" * 55)
    print(f"{'TOTAL':<12} {total_imgs:>8} {total_boxes:>8} "
          f"{total_boxes/total_imgs:>10.2f}")
    print(f"\nLabels written to: {LABEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
