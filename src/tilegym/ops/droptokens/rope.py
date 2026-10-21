# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""RoPE (Rotary Position Embedding) for the droptokens backend.

Runs the cuTile RoPE kernel through the ``cuda-tile-drop-tokens`` pass
(remove token usage from CUDA Tile IR) and executes the resulting cuTile IR
the same way the cutile backend does: through cuda.tile's native launch path
(``ct.launch``). The kernel itself lives only in the cuTile backend
(:mod:`tilegym.ops.cutile.rope`) and is imported from there.

The transformation happens inside ``cuda.tile.kernel._compile`` (a Python
hook called by ``ct.launch``): the kernel's cuTile IR bytecode is round-tripped
through the translation toolchain with the drop-tokens pass applied, the
resulting cuTile IR is compiled to a cubin via tileiras, and cuda.tile's
dispatcher launches that cubin with its usual argument marshalling.
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from tilegym.ops.cutile.rope import _rope_kernel
from tilegym.ops.cutile.utils import next_power_of_2

from .swiglu import _DropTokensKernel

# Same underlying kernel as the cutile backend's `_rope_kernel`; only the
# compilation path differs. occupancy=9 mirrors the cutile backend's
# non-autotuned performance fallback (see `_select_rope_kernel`).
_droptokens_rope_kernel = _DropTokensKernel(
    _rope_kernel._pyfunc,
    occupancy=9,
)


def _rope_forward(q, k, cos, sin, rope_dim=None):
    """
    Apply rotary position encoding **in-place** on 4-D Q/K tensors.

    Supports both full RoPE (rope_dim is None) and partial RoPE
    (rope_dim < head_dim).  The unified kernel uses tile-space indexing so
    only the first ``rope_dim`` elements are touched; no host-side reshape,
    slice, or concatenation is needed.

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor (modified in-place)
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor (modified in-place)
        cos: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Cosine values
        sin: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Sine values
        rope_dim: Number of head dimensions to rotate (None = full head_dim)

    Returns:
        (q, k, cos, sin) — q and k rotated in-place.
    """
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]

    if rope_dim is None:
        rope_dim = head_dim
    half_rope_dim = rope_dim // 2

    # Ensure cos/sin are 3-D: (cos_bs, seq_len, rope_dim)
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    TILE_RD = next_power_of_2(half_rope_dim)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len
    grid = (n_row, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _droptokens_rope_kernel,
        (
            q,
            k,
            cos,
            sin,
            cos.shape[0],
            seq_len,
            TILE_QH,
            TILE_KH,
            TILE_RD,
        ),
    )

    return q, k, cos, sin


class _TileRopeFunction(torch.autograd.Function):
    """
    CUDA Tile implementation of the Rotary Positional Embedding (RoPE) operation. Please note that
    this implements the HuggingFace Llama & Mistral version, whose rotation matrix is slightly different
    than the original RoPE paper.

    Please find the corresponding HuggingFace implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

    For more details about the rotation matrix used here, please refer to:
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1, rope_dim=None):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, rope_dim) or (bsz, seq_len, rope_dim)  — rope_dim == head_dim for full RoPE
        sin size: same as cos
        """
        q, k, cos, sin = _rope_forward(q, k, cos, sin, rope_dim=rope_dim)
        ctx.save_for_backward(cos, sin)
        ctx.rope_dim = rope_dim
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        """
        Backward pass via inverse rotation.
        """
        cos, sin = ctx.saved_tensors
        dq = dq.contiguous()
        dk = dk.contiguous()
        dq_out, dk_out, _, _ = _rope_forward(dq, dk, cos, -sin, rope_dim=ctx.rope_dim)
        return dq_out, dk_out, None, None, None, None, None


@register_impl("apply_rope_base", backend="droptokens")
def apply_rope_base(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, partial_rotary_factor=1.0):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor
        cos: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Cosine tensor
        sin: [1, seq_len, rope_dim] or [bsz, seq_len, rope_dim] - Sine tensor
        position_ids: Optional - Position IDs tensor, default None
        unsqueeze_dim: Optional - Dimension to unsqueeze, default 1
        partial_rotary_factor: Fraction of head dims to rotate (default 1.0 = full RoPE)

    Returns:
        Query and key tensor pair with RoPE applied
    """
    rope_dim = None
    if partial_rotary_factor < 1.0:
        head_dim = q.shape[-1]
        rope_dim = int(head_dim * partial_rotary_factor)
        assert cos.shape[-1] == rope_dim, (
            f"cos last dim ({cos.shape[-1]}) must equal int(head_dim * partial_rotary_factor) "
            f"= int({head_dim} * {partial_rotary_factor}) = {rope_dim}"
        )
    return _TileRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim, rope_dim)


@register_impl("get_apply_rope_func", backend="droptokens")
def get_apply_rope_func(model="llama"):
    if model == "llama" or model == "qwen2" or model == "gemma3" or model == "gpt-oss":
        return apply_rope_base
    elif model == "qwen3_5":

        def wrapper(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return apply_rope_base(q, k, cos, sin, partial_rotary_factor=0.25)

        return wrapper
    elif model == "deepseek":

        def wrapper(q, k, freqs_cis):
            cos, sin = freqs_cis.real, freqs_cis.imag

            b, h, s, d = q.shape
            q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            b, h, s, d = k.shape
            k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

            return apply_rope_base(q, k, cos, sin)

        return wrapper

    else:
        raise ValueError(f"Unsupported model: {model}")
