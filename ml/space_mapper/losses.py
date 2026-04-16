"""Segmentation losses for imbalanced binary masks.

Plain BCE collapses when the positive class is rare (see docs/ml_math_explained.md
section 10). These losses make the rare "water" class count.
"""
from __future__ import annotations


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required for losses. Install with `pip install torch`.") from exc
    return torch


def soft_dice_loss(logits, targets, *, eps: float = 1.0):
    """1 - Dice, computed on probabilities. Directly optimizes overlap, ignores TN.

    L = 1 - (2 * sum(p*y) + eps) / (sum(p) + sum(y) + eps)
    """
    torch = _require_torch()
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum() + eps
    den = probs.sum() + targets.sum() + eps
    return 1.0 - num / den


def make_criterion(name: str, *, pos_weight=None):
    """Return a loss callable(logits, targets).

    name: 'bce' | 'dice' | 'bce_dice'
    pos_weight: optional scalar tensor for weighted BCE (weights the positive class).
    """
    torch = _require_torch()
    name = name.lower()
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if name == "bce":
        return bce
    if name == "dice":
        return lambda logits, targets: soft_dice_loss(logits, targets)
    if name == "bce_dice":
        return lambda logits, targets: bce(logits, targets) + soft_dice_loss(logits, targets)
    raise ValueError(f"Unknown loss '{name}'. Choose from bce, dice, bce_dice.")


def compute_pos_weight(records, *, cap: float = 100.0):
    """Estimate pos_weight = (#negative pixels / #positive pixels) over the masks.

    This is the factor that balances the two classes in weighted BCE.
    Capped so a mask with almost no water doesn't produce an explosive weight.
    """
    torch = _require_torch()
    from .io import read_mask

    pos = 0
    neg = 0
    for rec in records:
        mask = read_mask(rec.mask_path)
        p = int((mask > 0).sum())
        pos += p
        neg += int(mask.size - p)
    if pos == 0:
        return None
    weight = min(neg / pos, cap)
    return torch.tensor(weight, dtype=torch.float32)
