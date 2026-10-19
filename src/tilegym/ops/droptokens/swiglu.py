# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""SwiGLU forward for the droptokens backend.

Runs the cuTile SwiGLU kernel through the ``cuda-tile-drop-tokens`` pass
(remove token usage from CUDA Tile IR) and executes the resulting cuTile IR
the same way the cutile backend does: through cuda.tile's native launch path
(``ct.launch``). The kernel itself lives only in the cuTile backend
(:mod:`tilegym.ops.cutile.swiglu`) and is imported from there.

The transformation happens inside ``cuda.tile.kernel._compile`` (a Python
hook called by ``ct.launch``): the kernel's cuTile IR bytecode is round-tripped
through the translation toolchain with the drop-tokens pass applied, the
resulting cuTile IR is compiled to a cubin via tileiras, and cuda.tile's
dispatcher launches that cubin with its usual argument marshalling.
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from tilegym.ops.cutile.swiglu import _swiglu_forward_kernel
from tilegym.ops.cutile.utils import next_power_of_2

from .translator import drop_tokens_cubin


class _DropTokensKernel(ct.kernel):
    """A :class:`cuda.tile.kernel` compiled through the drop-tokens pass.

    ``ct.launch`` calls ``_compile`` (Python-level) to obtain the cubin for a
    given signature. Instead of the vanilla compilation result, we compile the
    kernel to cuTile IR bytecode, apply ``cuda-tile-drop-tokens``, and compile
    the token-dropped cuTile IR to cubin via the same tileiras path cuda.tile
    uses. The pass only removes token values inside the IR, so the external
    kernel signature (and hence cuda.tile's argument marshalling) is unchanged.
    """

    def _compile(self, signature, context, compute_capability=None):
        from cuda.tile._cext import get_compute_capability
        from cuda.tile._compile import compile_tile
        from cuda.tile._compile import format_sm_arch

        sm_arch = format_sm_arch(*(compute_capability or get_compute_capability()))
        result = compile_tile(
            self._annotated_function,
            (signature,),
            sm_arch,
            self._compiler_options,
            context,
            return_bytecode=True,
            return_cubin=False,
        )
        [kernel_sig] = result.kernel_signatures
        cubin = drop_tokens_cubin(
            result.bytecode,
            sm_arch,
            self._compiler_options,
            context.config.compiler_timeout_sec,
        )
        return cubin, kernel_sig.symbol, None, []


# Same underlying kernel and compiler options as the cutile backend's
# `_swiglu_forward_kernel`; only the compilation path differs.
_droptokens_swiglu_kernel = _DropTokensKernel(
    _swiglu_forward_kernel._pyfunc,
    occupancy=1,
    num_worker_warps=8,
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

    TILE_SIZE = next_power_of_2(n_cols)
    # Drop bounds checking only when the power-of-2 tile exactly covers n_cols
    # (every lane in-bounds); irregular shapes keep masking on the partial tile.
    check_bounds = TILE_SIZE != n_cols
    grid = (n_rows,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _droptokens_swiglu_kernel,
        (
            a,
            b,
            c,
            TILE_SIZE,
            check_bounds,
        ),
    )
    return c.view(*ori_shape)


def swiglu(a, b):
    """SwiGLU forward via cuTile IR with the cuda-tile-drop-tokens pass applied."""
    return _swiglu_forward(a, b)


@register_impl("get_swiglu", backend="droptokens")
def get_swiglu():
    return swiglu
