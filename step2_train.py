"""Prepare YOLO dataset splits and train YOLOv8n detector."""

import os, shutil, random, yaml
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
METAL_NUT = BASE / "metal_nut"
LABEL_DIR = BASE / "labels"
DS_DIR    = BASE / "dataset"       # will be created

SEED      = 42
VAL_FRAC  = 0.20                   # 20 % of defect images → val

CLASSES   = ["bent", "color", "flip", "scratch"]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def link_or_copy(src: Path, dst: Path):
    """Create a relative symlink; fall back to copy if symlinks not supported."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def split_class(class_name: str, rng: random.Random):
    """Return (train_stems, val_stems) for one defect class."""
    img_dir = METAL_NUT / "test" / class_name
    imgs = sorted(img_dir.glob("*.png"))
    stems = [p.stem for p in imgs]
    rng.shuffle(stems)
    cut = max(1, int(len(stems) * VAL_FRAC))
    return stems[cut:], stems[:cut]    # train, val


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)

    # Clean and recreate dataset dirs
    if DS_DIR.exists():
        shutil.rmtree(DS_DIR)
    for split in ("train", "val"):
        (DS_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DS_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_count = val_count = 0

    # 1. Defect images
    print("Splitting defect images...")
    for cls in CLASSES:
        train_stems, val_stems = split_class(cls, rng)
        print(f"  {cls:8s}: {len(train_stems)} train  {len(val_stems)} val")

        for split, stems in [("train", train_stems), ("val", val_stems)]:
            for stem in stems:
                src_img = METAL_NUT / "test" / cls / f"{stem}.png"
                src_lbl = LABEL_DIR / cls / f"{stem}.txt"
                # prefix filename with class to avoid collisions (e.g. both bent/000 and color/000)
                fname = f"{cls}_{stem}"
                link_or_copy(src_img, DS_DIR / "images" / split / f"{fname}.png")
                link_or_copy(src_lbl, DS_DIR / "labels" / split / f"{fname}.txt")

        train_count += len(train_stems)
        val_count   += len(val_stems)

    # 2. Good (negative) images — put ALL in train, a small slice in val
    print("\nAdding good (negative) images...")
    good_train_imgs = sorted((METAL_NUT / "train" / "good").glob("*.png"))
    good_test_imgs  = sorted((METAL_NUT / "test"  / "good").glob("*.png"))

    # Use all 220 good-train images as negatives in train split
    for img in good_train_imgs:
        fname = f"good_{img.stem}"
        link_or_copy(img,                             DS_DIR / "images" / "train" / f"{fname}.png")
        link_or_copy(LABEL_DIR / "good_train" / f"{img.stem}.txt",
                     DS_DIR / "labels" / "train" / f"{fname}.txt")
    train_count += len(good_train_imgs)

    # Use all 22 good-test images as negatives in val split
    for img in good_test_imgs:
        fname = f"good_{img.stem}"
        link_or_copy(img,                             DS_DIR / "images" / "val" / f"{fname}.png")
        link_or_copy(LABEL_DIR / "good_test" / f"{img.stem}.txt",
                     DS_DIR / "labels" / "val" / f"{fname}.txt")
    val_count += len(good_test_imgs)

    print(f"  good_train: {len(good_train_imgs)} images → train")
    print(f"  good_test : {len(good_test_imgs)} images → val")

    # 3. Write dataset.yaml
    yaml_data = {
        "path": str(DS_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = DS_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*50}")
    print(f"Dataset ready at: {DS_DIR}")
    print(f"  Train images : {train_count}")
    print(f"  Val   images : {val_count}")
    print(f"  YAML         : {yaml_path}")
    print(f"  Classes      : {CLASSES}")

    # 4. Train YOLOv8n
    print(f"\n{'='*50}")
    print("Starting YOLOv8n training...")
    print("  model   : yolov8n.pt  (~3.2 M params)")
    print("  imgsz   : 320")
    print("  epochs  : 100")
    print("  batch   : 16")
    print(f"{'='*50}\n")

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    results = model.train(
        data      = str(yaml_path),
        epochs    = 100,
        imgsz     = 320,
        batch     = 16,
        patience  = 20,           # early stop if no improvement for 20 epochs
        project   = str(BASE / "runs"),
        name      = "yolov8n_320",
        exist_ok  = True,
        device    = "cpu",        # ensure CPU (edge parity)
        workers   = 4,
        verbose   = True,
        # Augmentation (built-in YOLO defaults cover mosaic, flips, HSV)
        mosaic    = 1.0,
        fliplr    = 0.5,
        flipud    = 0.5,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 10.0,         # small rotation (nuts can appear at slight angles)
        scale     = 0.5,
    )

    print("\nTraining complete!")
    print(f"Best weights: {BASE}/runs/yolov8n_320/weights/best.pt")


if __name__ == "__main__":
    main()
