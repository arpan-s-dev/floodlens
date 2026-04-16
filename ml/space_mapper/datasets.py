from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .io import list_image_files, read_image, read_mask


@dataclass(frozen=True)
class SegmentationRecord:
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str = "train"


def discover_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[SegmentationRecord]:
    """Pair images and masks by filename stem."""

    image_paths = {path.stem: path for path in list_image_files(image_dir)}
    mask_paths = {path.stem: path for path in list_image_files(mask_dir)}
    common_stems = sorted(set(image_paths) & set(mask_paths))
    return [SegmentationRecord(stem, image_paths[stem], mask_paths[stem]) for stem in common_stems]


def split_records(
    records: Sequence[SegmentationRecord],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
) -> list[SegmentationRecord]:
    """Assign deterministic train/val/test splits to records."""

    if not records:
        return []
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive.")
    train_cut = train_ratio / total_ratio
    val_cut = (train_ratio + val_ratio) / total_ratio

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split_records_out: list[SegmentationRecord] = []
    n_records = len(shuffled)
    for index, record in enumerate(shuffled):
        fraction = index / n_records
        if fraction < train_cut:
            split = "train"
        elif fraction < val_cut:
            split = "val"
        else:
            split = "test"
        split_records_out.append(SegmentationRecord(record.sample_id, record.image_path, record.mask_path, split))
    return sorted(split_records_out, key=lambda item: item.sample_id)


def _display_path(path: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path.resolve())
    return os.path.relpath(path.resolve(), relative_to.resolve())


def write_manifest(records: Iterable[SegmentationRecord], output_path: str | Path, *, relative_to: str | Path | None = None) -> Path:
    """Write a CSV manifest with id,image_path,mask_path,split columns."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = Path(relative_to) if relative_to is not None else output.parent
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "mask_path", "split"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.sample_id,
                    "image_path": _display_path(record.image_path, base),
                    "mask_path": _display_path(record.mask_path, base),
                    "split": record.split,
                }
            )
    return output


def make_manifest(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_path: str | Path,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
) -> list[SegmentationRecord]:
    records = discover_pairs(image_dir, mask_dir)
    records = split_records(records, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    write_manifest(records, output_path)
    return records


def read_manifest(manifest_path: str | Path, *, split: str | None = None) -> list[SegmentationRecord]:
    manifest = Path(manifest_path)
    base = manifest.parent
    records: list[SegmentationRecord] = []
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "image_path", "mask_path", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {manifest} is missing columns: {sorted(missing)}")
        for row in reader:
            row_split = row["split"] or "train"
            if split is not None and row_split != split:
                continue
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            if not image_path.is_absolute():
                image_path = base / image_path
            if not mask_path.is_absolute():
                mask_path = base / mask_path
            records.append(SegmentationRecord(row["id"], image_path, mask_path, row_split))
    return records


def _resize_channel(channel: np.ndarray, size_hw: tuple[int, int], *, is_mask: bool) -> np.ndarray:
    from PIL import Image

    height, width = size_hw
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    if is_mask:
        image = Image.fromarray((channel > 0).astype(np.uint8) * 255, mode="L")
        resized = np.asarray(image.resize((width, height), resample=resample)) > 0
        return resized.astype(np.float32)
    image = Image.fromarray(channel.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), resample=resample), dtype=np.float32)


def resize_array(array: np.ndarray, size: int | tuple[int, int], *, is_mask: bool = False) -> np.ndarray:
    """Resize HxW or HxWxC arrays. Size is int or (height, width)."""

    size_hw = (size, size) if isinstance(size, int) else size
    array = np.asarray(array)
    if array.ndim == 2:
        return _resize_channel(array, size_hw, is_mask=is_mask)
    channels = [_resize_channel(array[..., idx], size_hw, is_mask=is_mask) for idx in range(array.shape[-1])]
    return np.stack(channels, axis=-1)


class SegmentationDataset:
    """Small PyTorch-compatible dataset for local segmentation manifests."""

    def __init__(self, records: Sequence[SegmentationRecord], *, resize: int | tuple[int, int] | None = 128):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("PyTorch is required for SegmentationDataset. Install with `pip install torch`.") from exc
        self.torch = torch
        self.records = list(records)
        self.resize = resize

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = read_image(record.image_path, normalize=True).astype(np.float32)
        mask = read_mask(record.mask_path).astype(np.float32)
        if self.resize is not None:
            image = resize_array(image, self.resize, is_mask=False)
            mask = resize_array(mask, self.resize, is_mask=True)
        if image.ndim == 2:
            image = image[..., None]
        image_tensor = self.torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32))
        mask_tensor = self.torch.from_numpy(mask[None, ...].astype(np.float32))
        return image_tensor, mask_tensor
