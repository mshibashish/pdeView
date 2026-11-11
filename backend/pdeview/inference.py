"""Autoregressive FNO rollout (same scheme as PDEBench's training / evaluation loop)."""

from __future__ import annotations

import numpy as np
import torch

from pdeview.model.fno import FNO2d


def make_grid(x: np.ndarray, y: np.ndarray) -> torch.Tensor:
    """Raw cell-centre coordinates as an ``[X, Y, 2]`` tensor (``indexing='ij'``, like PDEBench)."""
    gx, gy = torch.meshgrid(torch.as_tensor(x), torch.as_tensor(y), indexing="ij")
    return torch.stack((gx, gy), dim=-1).float()


@torch.inference_mode()
def rollout(
    model: FNO2d,
    init_frames: np.ndarray,
    grid: torch.Tensor,
    n_steps: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Roll the FNO forward from ground-truth input frames.

    Args:
        init_frames: ``[B, initial_step, X, Y]`` (or ``[initial_step, X, Y]``) input frames.
        grid: ``[X, Y, 2]`` coordinates from :func:`make_grid`.
        n_steps: total number of output frames (including the input frames).

    Returns:
        ``[B, n_steps, X, Y]`` (or ``[n_steps, X, Y]``) float32 array whose first
        ``initial_step`` frames are the inputs and the rest are FNO predictions. Each
        prediction is fed back into a sliding window of the last ``initial_step`` frames.
    """
    squeeze = init_frames.ndim == 3
    frames = init_frames[None] if squeeze else init_frames
    b, k, nx, ny = frames.shape
    if k != model.initial_step:
        raise ValueError(f"model expects {model.initial_step} input frames, got {k}")

    xx = torch.tensor(np.asarray(frames), dtype=torch.float32, device=device)
    xx = xx.permute(0, 2, 3, 1)  # [B, X, Y, k]: time is the (innermost) channel axis
    g = grid.to(device).unsqueeze(0).expand(b, -1, -1, -1)

    out = torch.empty((b, nx, ny, n_steps), dtype=torch.float32, device=device)
    out[..., :k] = xx
    for step in range(k, n_steps):
        nxt = model(xx, g)[..., 0, 0]  # [B, X, Y]
        out[..., step] = nxt
        xx = torch.cat((xx[..., 1:], nxt.unsqueeze(-1)), dim=-1)

    res = out.permute(0, 3, 1, 2).cpu().numpy()
    return res[0] if squeeze else res


def persistence(init_frames: np.ndarray, n_steps: int) -> np.ndarray:
    """Trivial baseline: hold the last input frame constant."""
    k = init_frames.shape[-3]
    reps = np.repeat(init_frames[..., -1:, :, :], n_steps - k, axis=-3)
    return np.concatenate([init_frames, reps], axis=-3)
