"""Shared temporal placement helpers for clean LTX reference tokens.

LTX VAEs normally compress eight pixel frames into one latent-time step.  The
public node input is expressed in latent steps so ``-1`` consistently means
"one VAE temporal step before frame zero" without hard-coding a particular
VAE scale factor.
"""

from __future__ import annotations

import torch

DEFAULT_REFERENCE_TEMPORAL_OFFSET_LATENTS = 0


def reference_temporal_offset_input():
    """Return a fresh ComfyUI input declaration for reference-only placement."""
    return (
        "INT",
        {
            "default": DEFAULT_REFERENCE_TEMPORAL_OFFSET_LATENTS,
            "min": -16,
            "max": 16,
            "step": 1,
            "tooltip": (
                "Temporal RoPE offset for appearance/reference tokens only, in VAE latent-time steps. "
                "0 keeps the standard frame-0 placement used by existing LoRAs. "
                "-1 places the reference one latent step before frame 0 (normally -8 pixel frames), "
                "which can be tested to reduce overlap frame-0 reference leakage. "
                "Guide, mask, and target positions are never changed."
            ),
        },
    )


def reference_temporal_pixel_offset(
    offset_latents: int, temporal_scale_factor: int
) -> int:
    """Convert a latent-step offset to the raw pixel-frame units used by LTX RoPE."""
    return int(offset_latents) * int(temporal_scale_factor)


def shift_reference_temporal_positions(
    positions: torch.Tensor,
    offset_latents: int,
    temporal_scale_factor: int,
) -> torch.Tensor:
    """Shift only the temporal axis of a reference coordinate tensor.

    The helper accepts both ComfyUI's ``[B, axes, tokens]`` coordinates and
    trainer-style tensors with extra trailing dimensions.  A zero offset is an
    exact no-op and returns the original tensor for backward-compatible specs.
    """
    pixel_offset = reference_temporal_pixel_offset(
        offset_latents, temporal_scale_factor
    )
    if pixel_offset == 0:
        return positions
    if positions.ndim < 3 or positions.shape[1] < 1:
        raise ValueError(
            "Reference temporal positions must have shape [batch, axes, ...] "
            f"with a temporal axis; got {tuple(positions.shape)}"
        )
    shifted = positions.clone()
    shifted[:, 0, ...] += torch.as_tensor(
        pixel_offset,
        device=shifted.device,
        dtype=shifted.dtype,
    )
    return shifted
