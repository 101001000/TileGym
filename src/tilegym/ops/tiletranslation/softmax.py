# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd

from .translator import exhaustive_search_triton, get_sms, invoke_triton_kernel

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl
from tilegym.experimental import experimental_kernel

from tilegym.ops.cutile.utils import next_power_of_2

from tilegym.ops.cutile.softmax import (
    _softmax_kernel,
    _softmax_kernel_multi_wave_full_row_reg_cached_ldg,
    _softmax_kernel_tma,
    _softmax_kernel_chunked,
)

ConstInt = ct.Constant[int]

_DEFAULT_TMA_OCCUPANCY = 2
_TMA_FWD_OCCUPANCY_CANDIDATES = (2, 4, 6, 8)
# None keeps the compiler default; worker warps above 8 can hang the gather kernels.
_GATHER_WORKER_WARP_CANDIDATES = (None, 4, 8)
_softmax_tune_cache: dict = {}


# Launch patterns for the kernels:
def _cfg_hints(cfg):
    """Map a tuning-config namespace to a cuTile launch-hint dict."""
    hints = {}
    if getattr(cfg, "occupancy", None) is not None:
        hints["occupancy"] = cfg.occupancy
    if getattr(cfg, "num_worker_warps", None) is not None:
        hints["num_worker_warps"] = cfg.num_worker_warps
    return hints


def _launch_softmax_kernel(input, output, TILE_SIZE=1024):
    """
    Launch the basic cuTile softmax kernel with static persistent scheduling

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
        TILE_SIZE: Tile size for processing
    """
    n_rows, n_cols = input.shape
    original_n_cols = n_cols

    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = get_sms()
    num_programs = min(NUM_SM * 4, n_rows)
    grid = (num_programs, 1, 1)
    args = (output, input, n_rows, TILE_SIZE, original_n_cols)

    candidates = [SimpleNamespace(num_worker_warps=w) for w in _GATHER_WORKER_WARP_CANDIDATES]
    if is_autotune_disabled():
        cfg = candidates[0]
    else:
        cache_key = ("gather_fwd", str(input.device), input.dtype, n_rows, original_n_cols)
        cfg = _softmax_tune_cache.get(cache_key)
        if cfg is None:
            result = exhaustive_search_triton(
                candidates,
                torch.accelerator.current_stream(input.device),
                lambda _: grid,
                _softmax_kernel,
                lambda _: args,
                _cfg_hints,
            )
            cfg = _softmax_tune_cache[cache_key] = result.best.config
    hints = _cfg_hints(cfg)
    #softmax_kernel_forward = cached_replace_hints(_softmax_kernel, **hints) if hints else _softmax_kernel

    #ct.launch(torch.cuda.current_stream(), grid, softmax_kernel_forward, args)
    invoke_triton_kernel(torch.accelerator.current_stream(input.device), grid, _softmax_kernel, args, hints=hints)


def _launch_softmax_kernel_multi_wave_full_row_reg_cached_ldg(input, output, TILE_SIZE):
    n_rows, n_cols = input.shape
    input = input.contiguous()
    output = output.contiguous()

    grid = (n_rows, 1, 1)
    invoke_triton_kernel(
        torch.accelerator.current_stream(input.device),
        grid,
        _softmax_kernel_multi_wave_full_row_reg_cached_ldg,
        (output, input, n_cols, TILE_SIZE),
    )


def _launch_softmax_kernel_tma(
    input,
    output,
):
    """
    Launch the TMA cuTile softmax kernel

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
    """
    # Ensure input is 2D
    if input.dim() == 1:
        input = input.unsqueeze(0)
        output = output.unsqueeze(0)
    elif input.dim() > 2:
        input = input.view(-1, input.shape[-1])
        output = output.view(-1, output.shape[-1])

    n_rows, n_cols = input.shape

    TILE_SIZE = next_power_of_2(n_cols)
    original_n_cols = n_cols

    # Regular TMA path (single tile per row, persistent scheduling)
    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = get_sms()
    args = (output, input, n_rows, original_n_cols, TILE_SIZE)
    candidates = [SimpleNamespace(occupancy=o) for o in _TMA_FWD_OCCUPANCY_CANDIDATES]
    if is_autotune_disabled():
        cfg = candidates[0]
    else:
        cache_key = ("tma_fwd", str(input.device), input.dtype, n_rows, original_n_cols)
        cfg = _softmax_tune_cache.get(cache_key)
        if cfg is None:
            result = exhaustive_search_triton(
                candidates,
                torch.accelerator.current_stream(input.device),
                lambda candidate: (min(NUM_SM * candidate.occupancy, n_rows), 1, 1),
                _softmax_kernel_tma,
                lambda _: args,
                _cfg_hints,
            )
            cfg = _softmax_tune_cache[cache_key] = result.best.config
    occupancy = cfg.occupancy
    grid = (min(NUM_SM * occupancy, n_rows), 1, 1)

    invoke_triton_kernel(torch.accelerator.current_stream(input.device), grid, _softmax_kernel_tma, args, hints={"occupancy": occupancy})


def _launch_softmax_kernel_chunked(
    input,
    output,
    TILE_SIZE=8192,
):
    """
    Launch the chunked cuTile softmax kernel for large tensors

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
        TILE_SIZE: Tile size for processing chunks (default 8192)
    """
    n_rows, n_cols = input.shape
    original_n_cols = n_cols

    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = get_sms()
    num_programs = min(NUM_SM * 4, n_rows)
    grid = (num_programs, 1, 1)

    invoke_triton_kernel(
        torch.accelerator.current_stream(input.device),
        grid,
        _softmax_kernel_chunked,
        (
            output,
            input,
            n_rows,
            original_n_cols,
            TILE_SIZE,
        ),
    )


# Autograd Function Classes


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        use_tma=False,
        use_chunked=False,
        use_multi_wave=False,
    ):
        assert not (use_tma and use_chunked), "Cannot use both TMA and chunked softmax at the same time"
        # TMA may be emulated on this arch; redirect to non-TMA path with a warning.
        if use_tma and torch.cuda.get_device_capability(x.device)[0] < 9:
            import warnings

            warnings.warn(
                "softmax: use_tma=True has no effect on this GPU (TMA is emulated). "
                "Falling back to use_tma=False. Pass use_tma=False to suppress this.",
                UserWarning,
                stacklevel=3,
            )
            use_tma = False
        n_rows, n_cols = x.shape
        TILE_SIZE = next_power_of_2(n_cols)
        MAX_TILE_SIZE = 8192

        # Create output tensor; kernels write in place, so it must be contiguous even if x is not
        y = torch.empty_like(x, memory_format=torch.contiguous_format)

        if use_multi_wave:
            _launch_softmax_kernel_multi_wave_full_row_reg_cached_ldg(x, y, TILE_SIZE=TILE_SIZE)
        elif use_chunked:
            # Use chunked kernel (3-pass algorithm for large tensors)
            # Cap TILE_SIZE at 8192 to enable chunking for very large n_cols
            # For smaller n_cols, use next_power_of_2(n_cols) to match data size
            _launch_softmax_kernel_chunked(x, y, TILE_SIZE=min(TILE_SIZE, MAX_TILE_SIZE))
        elif use_tma:
            # Use TMA implementation
            _launch_softmax_kernel_tma(x, y)
        else:
            # Use grid-based implementation
            _launch_softmax_kernel(x, y, TILE_SIZE=TILE_SIZE)
        return y


@register_impl("softmax", backend="tiletranslation")
def softmax(
    x,
    use_tma=False,
    **kwargs,
):
    """
    Performs softmax using cuTile kernels with automatic gradient support

    Args:
        x: Input tensor of shape (M, N)
        use_tma: Whether to use TMA (Tensor Memory Accelerator) implementation.
                Requires H100+ GPU (compute capability >= 9.0)
        **kwargs: Additional arguments for backend-specific configurations
                  use_chunked: whether to use chunked softmax implementation
                  use_multi_wave: whether to use multi-wave (one block per row)

    Returns:
        Softmax output tensor with gradient support
    """
    use_chunked = kwargs.get("use_chunked", False)
    use_multi_wave = kwargs.get("use_multi_wave", False)
    # Launch on x's device: the launchers use the current device's stream.
    return _Softmax.apply(
        x,
        use_tma,
        use_chunked,
        use_multi_wave,
    )

