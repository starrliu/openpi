"""Loss utilities shared by training entrypoints."""

import torch


def masked_mean(losses: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return an element-wise masked mean, or the ordinary mean when no mask is given.

    An empty mask produces a differentiable zero rather than NaN.
    """
    if mask is None:
        return losses.mean()
    if mask.shape != losses.shape:
        raise ValueError(f"mask shape {mask.shape} does not match loss shape {losses.shape}")
    mask = mask.to(device=losses.device, dtype=losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1)
