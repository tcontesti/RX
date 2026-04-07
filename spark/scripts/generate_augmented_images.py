#!/usr/bin/env python3
"""Generate augmented images for NODE21 dataset.

Reads metadata_augmented_def2.csv, generates augmented PNG images from
originals using the same Albumentations pipeline as the Colab project,
and outputs an updated CSV with corrected file_paths and recalculated bboxes.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

IMG_DIR = os.path.expanduser("~/nodule_detection/data/png_images")
CSV_PATH = os.path.expanduser("~/nodule_detection/data/metadata_augmented_def2.csv")
OUT_CSV = os.path.expanduser("~/nodule_detection/data/metadata_augmented_spark.csv")

# Same augmentation pipeline as original Colab project
augment = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.5,
        ),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ),
)


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Input CSV: {len(df)} rows")

    new_rows = []
    generated = 0
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        img_name = str(row["img_name"])

        if "_aug" not in img_name:
            # Original image — fix file_path only
            png_name = img_name.replace(".mha", ".png")
            new_path = os.path.join(IMG_DIR, png_name)
            new_row = row.copy()
            new_row["file_path"] = new_path
            new_row["img_name"] = png_name if img_name.endswith(".mha") else img_name
            new_rows.append(new_row)
            continue

        # Augmented image — generate from original
        base = img_name.split("_aug")[0]
        if not base.endswith(".png"):
            base = base + ".png"

        orig_path = os.path.join(IMG_DIR, base)
        if not os.path.exists(orig_path):
            failed += 1
            continue

        img = cv2.imread(orig_path)
        if img is None:
            failed += 1
            continue

        # Find original bboxes for the base image
        base_mha = base.replace(".png", ".mha")
        orig_rows = df[
            ((df["img_name"] == base_mha) | (df["img_name"] == base))
            & (df["label"] == 1)
        ]

        if len(orig_rows) == 0:
            failed += 1
            continue

        h, w = img.shape[:2]

        # Collect ALL bboxes for this base image
        bboxes = []
        labels = []
        for _, orow in orig_rows.iterrows():
            x1 = float(orow["x"])
            y1 = float(orow["y"])
            x2 = x1 + float(orow["width"])
            y2 = y1 + float(orow["height"])
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))
                labels.append(1)

        if len(bboxes) == 0:
            failed += 1
            continue

        # Apply augmentation
        try:
            augmented = augment(image=img, bboxes=bboxes, class_labels=labels)
        except Exception:
            failed += 1
            continue

        if len(augmented["bboxes"]) == 0:
            failed += 1
            continue

        img_aug = augmented["image"]
        h2, w2 = img_aug.shape[:2]

        # Save augmented image
        out_name = img_name if img_name.endswith(".png") else img_name
        out_path = os.path.join(IMG_DIR, out_name)
        cv2.imwrite(out_path, img_aug)

        # Create one row per output bbox
        for bbox_out in augmented["bboxes"]:
            x1a, y1a, x2a, y2a = bbox_out
            x1a = max(0, min(x1a, w2 - 1))
            y1a = max(0, min(y1a, h2 - 1))
            x2a = max(0, min(x2a, w2))
            y2a = max(0, min(y2a, h2))
            bw = x2a - x1a
            bh = y2a - y1a
            if bw < 2 or bh < 2:
                continue
            new_row = row.copy()
            new_row["file_path"] = out_path
            new_row["img_name"] = out_name
            new_row["x"] = x1a
            new_row["y"] = y1a
            new_row["width"] = bw
            new_row["height"] = bh
            new_row["label"] = 1
            new_rows.append(new_row)

        generated += 1

    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(OUT_CSV, index=False)

    print(f"\nGenerated: {generated} augmented images")
    print(f"Failed: {failed}")
    print(f"Total rows: {len(df_new)}")
    print(f"Unique images: {df_new['img_name'].nunique()}")
    print(f"Positives: {(df_new['label'] == 1).sum()}")
    print(f"Negatives: {(df_new['label'] == 0).sum()}")
    print(f"Saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
