# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""SwiGLU forward for the tiletranslation backend.

Runs the cuTile SwiGLU kernel through the cuTile -> Triton translation
pipeline. The kernel itself lives only in the cuTile backend
(:mod:`tilegym.ops.cutile.swiglu`) and is imported from there.
"""

import torch

from tilegym.backend import register_impl

from tilegym.ops.cutile.swiglu import _swiglu_forward_kernel
from tilegym.ops.cutile.utils import next_power_of_2

from .translator import invoke_triton_kernel



def _swiglu_forward(a, b):
    """
    a: (batch_size, seq_len, intermediate_size)
    b: (batch_size, seq_len, intermediate_size)
    """
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    TILE_SIZE = next_power_of_2(n_cols)
    # Drop bounds checking only when the power-of-2 tile exactly covers n_cols
    # (every lane in-bounds); irregular shapes keep masking on the partial tile.
    check_bounds = TILE_SIZE != n_cols
    grid = (n_rows,)

    invoke_triton_kernel(
        torch.accelerator.current_stream(a.device),
        (n_rows, 1, 1),
        _swiglu_forward_kernel,
        a, b, c, TILE_SIZE, check_bounds,
    )

    return c.view(*ori_shape)


def swiglu(a, b):
    """SwiGLU forward via the cuTile -> Triton translation pipeline."""
    return _swiglu_forward(a, b)


@register_impl("get_swiglu", backend="tiletranslation")
def get_swiglu():
    return swiglu
