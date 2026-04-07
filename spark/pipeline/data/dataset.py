"""Generic detection dataset supporting PNG, DICOM, and MHA formats."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    """Load an image file and return a 2-D uint8 grayscale array.

    Supports PNG/JPG (via OpenCV), DICOM (.dcm via pydicom), and
    MHA/MHD (via SimpleITK).
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".dcm":
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # Window/level normalisation
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter[0])
            ww = float(ds.WindowWidth if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth[0])
            lower = wc - ww / 2
            upper = wc + ww / 2
            arr = np.clip(arr, lower, upper)
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return arr

    if ext in (".mha", ".mhd"):
        import SimpleITK as sitk

        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # If 3-D volume, take the middle slice
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return arr

    # Default: PNG, JPG, TIFF, etc.
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


# ---------------------------------------------------------------------------
# Channel construction
# ---------------------------------------------------------------------------

def _grayscale_x3(gray: np.ndarray) -> np.ndarray:
    """Stack grayscale into 3 identical channels (H, W, 3)."""
    return np.stack([gray, gray, gray], axis=-1)


def _multichannel_3ch(gray: np.ndarray) -> np.ndarray:
    """3 channels: original, CLAHE, unsharp-mask. Returns (H, W, 3)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    ch_unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return np.stack([gray, ch_clahe, ch_unsharp], axis=-1)


def _multichannel_4ch(gray: np.ndarray) -> np.ndarray:
    """4 channels: original, CLAHE, unsharp-mask, Canny. Returns (H, W, 4)."""
    base_3 = _multichannel_3ch(gray)
    canny = cv2.Canny(gray, 50, 150)
    return np.concatenate([base_3, canny[:, :, None]], axis=-1)


_INPUT_MODE_FN = {
    "grayscale_x3": _grayscale_x3,
    "multichannel_3ch": _multichannel_3ch,
    "multichannel_4ch": _multichannel_4ch,
}


# ---------------------------------------------------------------------------
# Box format helpers
# ---------------------------------------------------------------------------

def _parse_boxes(group: pd.DataFrame) -> np.ndarray:
    """Parse bounding boxes from a group of rows. Returns (N, 4) in x1,y1,x2,y2."""
    if {"x_min", "y_min", "x_max", "y_max"}.issubset(group.columns):
        boxes = group[["x_min", "y_min", "x_max", "y_max"]].values.astype(np.float32)
    elif {"x", "y", "width", "height"}.issubset(group.columns):
        xywh = group[["x", "y", "width", "height"]].values.astype(np.float32)
        boxes = np.zeros_like(xywh)
        boxes[:, 0] = xywh[:, 0]           # x_min
        boxes[:, 1] = xywh[:, 1]           # y_min
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2]  # x_max
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3]  # y_max
    else:
        raise ValueError(
            "CSV must contain either (x_min, y_min, x_max, y_max) or (x, y, width, height) columns."
        )
    return boxes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DetectionDataset(Dataset):
    """Generic object-detection dataset.

    Args:
        csv_path: Path to annotations CSV with columns:
            file_path, label, and bounding-box columns.
        input_mode: One of "grayscale_x3", "multichannel_3ch", "multichannel_4ch".
        transforms: Optional Albumentations Compose with BboxParams.
        class_map: Optional dict mapping label strings to integer class ids.
            If None, labels are auto-mapped alphabetically starting at 1
            (0 is reserved for background).
    """

    def __init__(
        self,
        csv_path: str,
        input_mode: str = "grayscale_x3",
        transforms: Optional[Any] = None,
        class_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.input_mode = input_mode
        self.transforms = transforms

        if input_mode not in _INPUT_MODE_FN:
            raise ValueError(f"Unknown input_mode '{input_mode}'. Choose from {list(_INPUT_MODE_FN.keys())}")
        self._channel_fn = _INPUT_MODE_FN[input_mode]

        # Build class mapping
        if class_map is not None:
            self.class_map = class_map
        else:
            unique_labels = sorted(self.df["label"].unique())
            self.class_map = {lbl: idx + 1 for idx, lbl in enumerate(unique_labels)}

        # Group annotations by image
        self.groups = list(self.df.groupby("file_path"))
        logger.info(
            "DetectionDataset: %d images, %d annotations, %d classes, mode=%s",
            len(self.groups),
            len(self.df),
            len(self.class_map),
            input_mode,
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        file_path, group = self.groups[idx]

        # Load and convert to multi-channel
        gray = _load_image(file_path)
        image = self._channel_fn(gray)  # (H, W, C)

        # Parse boxes and labels
        boxes = _parse_boxes(group)
        labels = np.array(
            [self.class_map.get(lbl, 0) for lbl in group["label"].values],
            dtype=np.int64,
        )

        # Apply augmentations (Albumentations expects HWC uint8 image)
        if self.transforms is not None:
            bbox_labels_list = labels.tolist()
            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist(),
                labels=bbox_labels_list,
            )
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(transformed["labels"], dtype=np.int64)

        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[0] != image.shape[2]:
            # Already CHW from albumentations ToTensorV2
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        return image, target

    @property
    def num_classes(self) -> int:
        """Number of classes including background (class 0)."""
        return len(self.class_map) + 1


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Custom collate that returns a list of (image, target) tuples unpacked
    into two separate lists, compatible with Faster R-CNN forward()."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
