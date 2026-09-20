"""Pure-PyTorch fallback for flash-attn's padding helpers.

This module only covers the data-layout helpers imported by veRL.  It is not a
FlashAttention implementation and must not be used as one.  Keeping these
helpers local lets veRL's SDPA/FSDP path run in a minimal CUDA environment
where the optional ``flash-attn`` extension is not installed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange as _einops_rearrange


class _IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.first_axis_dim = input.shape[0]
        ctx.other_shape = input.shape[1:]
        return input.index_select(0, indices)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        grad_input = torch.zeros(
            (ctx.first_axis_dim, *ctx.other_shape),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.index_add_(0, indices, grad_output)
        return grad_input, None


index_first_axis = _IndexFirstAxis.apply


class _IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int
    ) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.first_axis_dim = first_axis_dim
        output = torch.zeros(
            (first_axis_dim, *values.shape[1:]),
            device=values.device,
            dtype=values.dtype,
        )
        output.index_copy_(0, indices, values)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        return grad_output.index_select(0, indices), None, None


index_put_first_axis = _IndexPutFirstAxis.apply


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    unused_mask: Optional[torch.Tensor] = None,
):
    """Remove masked positions, matching flash-attn's ``unpad_input`` API."""

    all_masks = attention_mask if unused_mask is None else attention_mask + unused_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.reshape(-1), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    hidden_states = rearrange(hidden_states, "b s ... -> (b s) ...")
    return (
        index_first_axis(hidden_states, indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Restore a flattened tensor to ``(batch, seqlen, ...)`` layout."""

    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def rearrange(*args, **kwargs):
    """Expose the helper expected by veRL's attention utility wrapper."""

    return _einops_rearrange(*args, **kwargs)
