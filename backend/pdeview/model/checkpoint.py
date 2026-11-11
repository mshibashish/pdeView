"""Loading (and integrity-checking) the PDEBench pretrained FNO checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from pdeview.config import (
    CKPT_DIR,
    FNO_INITIAL_STEP,
    FNO_MODES,
    FNO_NUM_CHANNELS,
    FNO_WIDTH,
    SWE_FNO_CKPT,
)
from pdeview.data.remote import md5sum
from pdeview.model.fno import FNO2d


@dataclass
class CheckpointInfo:
    path: str
    md5: str
    md5_expected: str
    md5_ok: bool
    epoch: int | None
    stored_val_loss: float | None  # PDEBench saves the best validation loss under "loss"
    optimizer_lr: float | None
    n_parameters: int
    n_real_parameters: int
    keys: list[str] = field(default_factory=list)
    hyperparameters: dict = field(default_factory=dict)


def build_model() -> FNO2d:
    return FNO2d(
        num_channels=FNO_NUM_CHANNELS,
        modes1=FNO_MODES,
        modes2=FNO_MODES,
        width=FNO_WIDTH,
        initial_step=FNO_INITIAL_STEP,
    )


def load_fno(
    path: Path | None = None, device: str | torch.device = "cpu"
) -> tuple[FNO2d, CheckpointInfo]:
    """Load the pretrained FNO with ``strict=True`` and return it with integrity metadata."""
    path = Path(path or CKPT_DIR / SWE_FNO_CKPT.name)
    digest = md5sum(path)
    # weights_only=True: the checkpoint is a plain dict of tensors / numbers, no pickled code.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = build_model()
    model.load_state_dict(state, strict=True)  # raises on any missing / unexpected / mis-shaped key
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    loss = ckpt.get("loss") if isinstance(ckpt, dict) else None
    try:
        lr = float(ckpt["optimizer_state_dict"]["param_groups"][0]["lr"])
    except (KeyError, IndexError, TypeError):
        lr = None
    info = CheckpointInfo(
        path=str(path),
        md5=digest,
        md5_expected=SWE_FNO_CKPT.md5,
        md5_ok=digest == SWE_FNO_CKPT.md5,
        epoch=int(epoch) if epoch is not None else None,
        stored_val_loss=float(loss) if loss is not None else None,
        optimizer_lr=lr,
        n_parameters=n_params,
        n_real_parameters=n_real,
        keys=sorted(ckpt.keys()) if isinstance(ckpt, dict) else [],
        hyperparameters={
            "num_channels": FNO_NUM_CHANNELS,
            "modes": FNO_MODES,
            "width": FNO_WIDTH,
            "initial_step": FNO_INITIAL_STEP,
        },
    )
    return model, info
