"""Dataset splitting helpers."""

from __future__ import annotations

import math
import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def split_items(
    items: Sequence[T],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> dict[str, list[T]]:
    """Split a list into train/val/test subsets."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if any(ratio < 0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Ratios must be non-negative")
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    items_list = list(items)
    if shuffle:
        random.seed(seed)
        random.shuffle(items_list)

    n = len(items_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": items_list[:train_end],
        "val": items_list[train_end:val_end],
        "test": items_list[val_end:],
    }


__all__ = ["split_items"]
