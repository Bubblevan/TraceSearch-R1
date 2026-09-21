"""Small CPU/CUDA-compatible fallback for ``flash_attn`` rotary embeddings.

The pinned vLLM release imports this symbol while constructing Qwen2 even when
the actual fused FlashAttention package is not installed.  The implementation
below follows the public flash-attn layout and semantics, but deliberately uses
ordinary PyTorch operations instead of pretending to provide a Triton kernel.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def _position_ids(
    seqlen: int,
    batch: int,
    seqlen_offsets: Union[int, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(seqlen, device=device, dtype=torch.long)
    if isinstance(seqlen_offsets, int):
        return positions + seqlen_offsets
    offsets = seqlen_offsets.to(device=device, dtype=torch.long)
    if offsets.shape != (batch,):
        raise ValueError(f"seqlen_offsets must have shape ({batch},), got {tuple(offsets.shape)}")
    return offsets[:, None] + positions[None, :]


def _apply_rotary_block(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool,
    conjugate: bool,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError("rotary_dim must be less than or equal to the head dimension")
    if conjugate:
        sin = -sin

    cos = cos.unsqueeze(-2).to(dtype=x.dtype)
    sin = sin.unsqueeze(-2).to(dtype=x.dtype)
    x_rot = x[..., :rotary_dim]
    x_tail = x[..., rotary_dim:]

    if interleaved:
        x_even = x_rot[..., ::2]
        x_odd = x_rot[..., 1::2]
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_odd * cos + x_even * sin),
            dim=-1,
        ).flatten(-2)
    else:
        x_first, x_second = torch.chunk(x_rot, 2, dim=-1)
        rotated = torch.cat(
            (x_first * cos - x_second * sin, x_second * cos + x_first * sin),
            dim=-1,
        )
    return torch.cat((rotated, x_tail), dim=-1)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Apply rotary position embeddings with flash-attn-compatible arguments."""

    if cos.shape != sin.shape or cos.ndim != 2:
        raise ValueError("cos and sin must have the same two-dimensional shape")

    if cu_seqlens is None:
        if x.ndim != 4:
            raise ValueError("non-varlen rotary input must have shape [batch, seqlen, heads, dim]")
        batch, seqlen = x.shape[:2]
        positions = _position_ids(seqlen, batch, seqlen_offsets, device=x.device)
        selected_cos = cos.to(device=x.device).index_select(0, positions.reshape(-1))
        selected_sin = sin.to(device=x.device).index_select(0, positions.reshape(-1))
        if positions.ndim == 1:
            selected_cos = selected_cos.reshape(seqlen, -1).unsqueeze(0)
            selected_sin = selected_sin.reshape(seqlen, -1).unsqueeze(0)
        else:
            selected_cos = selected_cos.reshape(batch, seqlen, -1)
            selected_sin = selected_sin.reshape(batch, seqlen, -1)
        output = _apply_rotary_block(
            x,
            selected_cos,
            selected_sin,
            interleaved=interleaved,
            conjugate=conjugate,
        )
    else:
        if x.ndim != 3 or max_seqlen is None:
            raise ValueError("varlen rotary input requires [tokens, heads, dim] and max_seqlen")
        boundaries = cu_seqlens.to(device=x.device, dtype=torch.long)
        batch = boundaries.numel() - 1
        output = x.clone()
        for index in range(batch):
            start = int(boundaries[index].item())
            end = int(boundaries[index + 1].item())
            length = end - start
            offset = (
                seqlen_offsets
                if isinstance(seqlen_offsets, int)
                else int(seqlen_offsets[index].item())
            )
            positions = torch.arange(length, device=x.device, dtype=torch.long) + offset
            output[start:end] = _apply_rotary_block(
                x[start:end].unsqueeze(0),
                cos.to(device=x.device).index_select(0, positions).unsqueeze(0),
                sin.to(device=x.device).index_select(0, positions).unsqueeze(0),
                interleaved=interleaved,
                conjugate=conjugate,
            ).squeeze(0)

    if inplace:
        x.copy_(output)
        return x
    return output
