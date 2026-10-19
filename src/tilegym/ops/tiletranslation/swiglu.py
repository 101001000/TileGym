# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""SwiGLU forward for the tiletranslation backend.

Runs the cuTile SwiGLU kernel through the cuTile -> Triton translation
pipeline. The kernel itself lives only in the cuTile backend
(:mod:`tilegym.ops.cutile.swiglu`) and is imported from there.
"""

from functools import lru_cache

import torch

from tilegym.backend import register_impl

from tilegym.ops.cutile.swiglu import _swiglu_forward_kernel
from tilegym.ops.cutile.utils import next_power_of_2

from .translator import invoke_triton_kernel
from .translator import prepare_triton_kernel


@lru_cache(maxsize=None)
def _prepare_swiglu_forward(n_rows, n_cols, dtype, device):
    a = torch.empty((n_rows, n_cols), dtype=dtype, device=device)
    b = torch.empty_like(a)
    c = torch.empty_like(a)

    TILE_SIZE = next_power_of_2(n_cols)
    check_bounds = TILE_SIZE != n_cols
    major, minor = torch.cuda.get_device_capability(device)

    return prepare_triton_kernel(
        _swiglu_forward_kernel,
        (a, b, c, TILE_SIZE, check_bounds),
        f"sm_{major}{minor}",
        f"swiglu_{device.index}_{n_rows}_{n_cols}_{dtype}",
    )


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

    kernel = _prepare_swiglu_forward(n_rows, n_cols, a.dtype, a.device)

    invoke_triton_kernel(
        kernel,
        (n_rows, 1, 1),
        a, *a.shape, *a.stride(),
        b, *b.shape, *b.stride(),
        c, *c.shape, *c.stride(),
    )

    return c.view(*ori_shape)


def swiglu(a, b):
    """SwiGLU forward via the cuTile -> Triton translation pipeline."""
    return _swiglu_forward(a, b)


@register_impl("get_swiglu", backend="tiletranslation")
def get_swiglu():
    return swiglu
