#!/usr/bin/env python3
"""
Preprocess NODE21 dataset: MHA -> PNG, generate metadata, splits, YOLO format.
"""

import os
import sys
import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


def load_mha_to_array(mha_path):
    """Load .mha file and return numpy array normalized to [0,1] float32."""
    img = sitk.ReadImage(str(mha_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    # Handle 3D images (take first slice if needed)
    if arr.ndim == 3:
        arr = arr[0]
    # Normalize to [0, 1]
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 0:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)
    return arr


def save_as_png(arr, output_path, target_size=1024):
    """Save float32 [0,1] array as 8-bit PNG, resized to target_size."""
    # Convert to uint8
    img_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    # Resize if needed
    if img_uint8.shape[0] != target_size or img_uint8.shape[1] != target_size:
        img_uint8 = cv2.resize(img_uint8, (target_size, target_size),
                               interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(output_path), img_uint8)
    return img_uint8.shape


def convert_mha_to_png(mha_dir, png_dir, target_size=1024):
    """Convert all .mha files to PNG."""
    os.makedirs(png_dir, exist_ok=True)
    mha_files = sorted(glob.glob(os.path.join(mha_dir, "*.mha")))

    if not mha_files:
        print(f"ERROR: No .mha files found in {mha_dir}")
        sys.exit(1)

    print(f"Found {len(mha_files)} .mha files")
    conversion_info = []

    for mha_path in tqdm(mha_files, desc="Converting MHA -> PNG"):
        name = Path(mha_path).stem
        png_path = os.path.join(png_dir, f"{name}.png")

        try:
            arr = load_mha_to_array(mha_path)
            orig_shape = arr.shape
            save_as_png(arr, png_path, target_size)
            conversion_info.append({
                "img_name": name,
                "mha_path": mha_path,
                "png_path": os.path.abspath(png_path),
                "orig_h": orig_shape[0],
                "orig_w": orig_shape[1],
                "target_size": target_size,
            })
        except Exception as e:
            print(f"  WARNING: Failed to convert {name}: {e}")

    print(f"Converted {len(conversion_info)}/{len(mha_files)} images")
    return conversion_info


def process_metadata(metadata_csv, conversion_info, png_dir, target_size=1024):
    """Process original metadata and create processed CSV with PNG paths."""
    df = pd.read_csv(metadata_csv)
    print(f"\nOriginal metadata: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Build lookup from img_name -> conversion info
    conv_lookup = {c["img_name"]: c for c in conversion_info}

    # Add file paths and adjust coordinates for resize
    records = []
    for _, row in df.iterrows():
        img_name = str(row["img_name"]).replace(".mha", "")
        if img_name not in conv_lookup:
            continue

        info = conv_lookup[img_name]
        scale_x = target_size / info["orig_w"]
        scale_y = target_size / info["orig_h"]

        record = {
            "img_name": img_name,
            "file_path": info["png_path"],
            "label": int(row["label"]),
        }

        if row["label"] == 1 and pd.notna(row.get("x")):
            record["x"] = float(row["x"]) * scale_x
            record["y"] = float(row["y"]) * scale_y
            record["width"] = float(row["width"]) * scale_x
            record["height"] = float(row["height"]) * scale_y
        else:
            record["x"] = 0.0
            record["y"] = 0.0
            record["width"] = 0.0
            record["height"] = 0.0

        records.append(record)

    df_processed = pd.DataFrame(records)

    # Also add images that are in MHA dir but not in metadata (assume negative)
    metadata_names = set(df_processed["img_name"].unique())
    for info in conversion_info:
        if info["img_name"] not in metadata_names:
            records.append({
                "img_name": info["img_name"],
                "file_path": info["png_path"],
                "label": 0,
                "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0,
            })

    df_processed = pd.DataFrame(records)
    return df_processed


def create_splits(df, output_dir, n_splits=5, seed=42):
    """Create stratified group k-fold splits (grouped by image to prevent leakage)."""
    os.makedirs(output_dir, exist_ok=True)

    # Get unique images with their labels
    img_labels = df.groupby("img_name")["label"].max().reset_index()
    img_labels.columns = ["img_name", "has_nodule"]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Groups = img_name (each image is its own group)
    X = img_labels["img_name"].values
    y = img_labels["has_nodule"].values
    groups = img_labels["img_name"].values

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_names = set(X[train_idx])
        val_names = set(X[val_idx])

        train_df = df[df["img_name"].isin(train_names)]
        val_df = df[df["img_name"].isin(val_names)]

        train_path = os.path.join(output_dir, f"train_fold{fold}.csv")
        val_path = os.path.join(output_dir, f"val_fold{fold}.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        n_train_pos = train_df[train_df["label"] == 1]["img_name"].nunique()
        n_val_pos = val_df[val_df["label"] == 1]["img_name"].nunique()
        print(f"Fold {fold}: train={len(train_names)} imgs ({n_train_pos} pos), "
              f"val={len(val_names)} imgs ({n_val_pos} pos)")

    return train_names, val_names  # Return last fold


def create_yolo_dataset(df, png_dir, yolo_dir, train_csv, val_csv, target_size=1024):
    """Create YOLO-format dataset structure."""
    # Create directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        img_names = split_df["img_name"].unique()

        for img_name in tqdm(img_names, desc=f"YOLO {split_name}"):
            # Symlink or copy image
            src_png = os.path.join(png_dir, f"{img_name}.png")
            dst_png = os.path.join(yolo_dir, "images", split_name, f"{img_name}.png")

            if os.path.exists(src_png) and not os.path.exists(dst_png):
                os.symlink(os.path.abspath(src_png), dst_png)

            # Create label file
            img_rows = split_df[split_df["img_name"] == img_name]
            label_path = os.path.join(yolo_dir, "labels", split_name, f"{img_name}.txt")

            with open(label_path, "w") as f:
                for _, row in img_rows.iterrows():
                    if row["label"] == 1 and row["width"] > 0 and row["height"] > 0:
                        # Convert to YOLO format: class_id cx cy w h (normalized)
                        cx = (row["x"] + row["width"] / 2) / target_size
                        cy = (row["y"] + row["height"] / 2) / target_size
                        w = row["width"] / target_size
                        h = row["height"] / target_size

                        # Clip to [0, 1]
                        cx = np.clip(cx, 0, 1)
                        cy = np.clip(cy, 0, 1)
                        w = np.clip(w, 0, 1)
                        h = np.clip(h, 0, 1)

                        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # Create YAML config
    yaml_content = f"""# NODE21 Pulmonary Nodule Detection Dataset
path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val

# Classes
names:
  0: nodule

nc: 1
"""
    yaml_path = os.path.join(yolo_dir, "node21.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nYOLO dataset created at {yolo_dir}")
    print(f"  YAML config: {yaml_path}")


def print_statistics(df, png_dir):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total_images = df["img_name"].nunique()
    pos_images = df[df["label"] == 1]["img_name"].nunique()
    neg_images = total_images - pos_images
    total_nodules = len(df[df["label"] == 1])

    print(f"Total images:      {total_images}")
    print(f"Positive (nodule): {pos_images} ({100*pos_images/total_images:.1f}%)")
    print(f"Negative:          {neg_images} ({100*neg_images/total_images:.1f}%)")
    print(f"Total nodule BBs:  {total_nodules}")

    if total_nodules > 0:
        nodule_df = df[df["label"] == 1]
        print(f"\nNodule size distribution (pixels):")
        print(f"  Width  - mean: {nodule_df['width'].mean():.1f}, "
              f"median: {nodule_df['width'].median():.1f}, "
              f"min: {nodule_df['width'].min():.1f}, "
              f"max: {nodule_df['width'].max():.1f}")
        print(f"  Height - mean: {nodule_df['height'].mean():.1f}, "
              f"median: {nodule_df['height'].median():.1f}, "
              f"min: {nodule_df['height'].min():.1f}, "
              f"max: {nodule_df['height'].max():.1f}")

        # Nodules per image
        nods_per_img = nodule_df.groupby("img_name").size()
        print(f"\nNodules per positive image:")
        print(f"  mean: {nods_per_img.mean():.2f}, max: {nods_per_img.max()}")
        for n in sorted(nods_per_img.unique()):
            count = (nods_per_img == n).sum()
            print(f"  {n} nodule(s): {count} images")

    # Check PNG files
    png_count = len(glob.glob(os.path.join(png_dir, "*.png")))
    print(f"\nPNG files on disk: {png_count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Preprocess NODE21 dataset")
    parser.add_argument("--mha_dir", type=str,
                        default=os.path.expanduser(
                            "~/nodule_detection/data/dataset_node21/cxr_images/proccessed_data/"),
                        help="Directory with .mha files")
    parser.add_argument("--metadata_csv", type=str,
                        default=os.path.expanduser(
                            "~/nodule_detection/data/dataset_node21/cxr_images/proccessed_data/metadata.csv"),
                        help="Path to original metadata.csv")
    parser.add_argument("--png_dir", type=str,
                        default=os.path.expanduser("~/nodule_detection/data/png_images/"),
                        help="Output directory for PNG images")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser("~/nodule_detection/data/"),
                        help="Base output directory")
    parser.add_argument("--target_size", type=int, default=1024,
                        help="Target image size (default 1024)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()

    splits_dir = os.path.join(args.output_dir, "splits")
    yolo_dir = os.path.join(args.output_dir, "yolo_dataset")

    # Step 1: Convert MHA to PNG
    print("=" * 60)
    print("STEP 1: Converting MHA -> PNG")
    print("=" * 60)
    conversion_info = convert_mha_to_png(args.mha_dir, args.png_dir, args.target_size)

    # Step 2: Process metadata
    print("\n" + "=" * 60)
    print("STEP 2: Processing metadata")
    print("=" * 60)
    df = process_metadata(args.metadata_csv, conversion_info, args.png_dir, args.target_size)

    processed_csv = os.path.join(args.output_dir, "metadata_processed.csv")
    df.to_csv(processed_csv, index=False)
    print(f"Saved processed metadata to {processed_csv}")

    # Step 3: Create splits
    print("\n" + "=" * 60)
    print("STEP 3: Creating train/val splits")
    print("=" * 60)
    create_splits(df, splits_dir, n_splits=args.n_splits)

    # Step 4: Create YOLO dataset (using fold 0)
    print("\n" + "=" * 60)
    print("STEP 4: Creating YOLO dataset")
    print("=" * 60)
    train_csv = os.path.join(splits_dir, "train_fold0.csv")
    val_csv = os.path.join(splits_dir, "val_fold0.csv")
    create_yolo_dataset(df, args.png_dir, yolo_dir, train_csv, val_csv, args.target_size)

    # Step 5: Print statistics
    print_statistics(df, args.png_dir)

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
