# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch
import torch.nn as nn

from tilegym.backend import register_impl
from tilegym.experimental import experimental_kernel

from tilegym.ops.cutile.rms_norm import _rms_norm_kernel_multi_wave_cached, _rms_norm_kernel_gather, _rms_norm_kernel_static_persistent, _rms_bwd_kernel
from .translator import get_capability, exhaustive_search_triton, get_sms, invoke_triton_kernel

from tilegym.ops.cutile.utils import next_power_of_2

_bwd_cfg: dict = {}  # (M, N) → (tile_m, tile_n, grid, N)



def _bwd_tiles(M, N):
    """Heuristic tile configuration for backward kernel."""
    T = next_power_of_2(N)
    if T > 4096:
        tm = 1
    elif T <= 2048 or (M >= 8192 and T <= 4096):
        tm = 4
    else:
        tm = 1
    NUM_SMS = get_sms()
    tiles = (M + tm - 1) // tm
    g = min(NUM_SMS, tiles)
    if tiles <= 64:
        g = min(g, 32)
    return (tm, T, g, N)


def rms_norm_backward_workspace_bytes(M: int, N: int) -> int:
    """Return temporary workspace traffic for the standalone backward benchmark."""
    _, tile_n, grid, _ = _bwd_tiles(M, N)
    return grid * tile_n * 4 * 2


def compute_rms_norm_rstd_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute the PyTorch reference RMSNorm rstd used by standalone backward benchmarks."""
    return _TileRMSNorm.compute_rstd_torch(x, eps)


def rms_norm_backward_torch(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for standalone RMSNorm backward."""
    return _TileRMSNorm.rms_norm_backward_torch(x, dy, weight, rstd)


def _rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone backward pass using persistent CuTile kernel."""
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    rstd = rstd.contiguous()

    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    M, N = x.shape

    cfg = _bwd_cfg.get((M, N))
    if cfg is None:
        cfg = _bwd_tiles(M, N)
        _bwd_cfg[(M, N)] = cfg
    tm, T, g, No = cfg

    stream = torch.accelerator.current_stream(x.device)

    dx = torch.empty_like(x)
    dwp = torch.empty((g, T), device=x.device, dtype=torch.float32)
    invoke_triton_kernel(stream, (g,), _rms_bwd_kernel, (dx, dy, x, weight, rstd, dwp, tm, T))

    dw = dwp.sum(0)
    if T != No:
        dw = dw[:No]

    return dx.view(*x_shape), dw.to(weight.dtype)


def rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone cuTile RMSNorm backward entry point."""
    return _rms_norm_backward(x, dy, weight, rstd)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        normalized_shape,
        weight,
        eps,
        bias=None,
        mode=None,
        offset=0.0,
    ):
        """
        Unified RMSNorm forward pass supporting both standard and static persistent modes.

        Args:
            x: Input tensor of shape [M, N]
            normalized_shape: Normalization shape (for compatibility, not used)
            weight: Weight tensor of shape [N]
            eps: Epsilon value for numerical stability
            bias: Bias tensor of shape [N], default is None
            mode: Kernel selection mode (None, "static_persistent", "multi_wave_reload", "multi_wave_cached")
            offset: Offset to add to weight (default 0.0 for Llama, 1.0 for Gemma3)

        Returns:
            Normalized and transformed tensor of same shape as input
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        # Reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        y = torch.empty_like(x_arg)
        M, N = x_arg.shape
        y = y.detach()
        weight = weight.detach()
        if bias is not None:
            bias = bias.detach()
        x_arg = x_arg.detach()

        NUM_SMS = get_sms()

        if mode is None:
            if M > NUM_SMS * 2:
                # Heuristic for static persistent mode: if we need run over 2 waves, use static persistent mode
                mode = "static_persistent"
            else:
                mode = "multi_wave_reload"

        # Allocate rstd for backward (both paths now store it)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        if mode == "static_persistent":
            # Static persistent mode
            if bias is not None:
                raise NotImplementedError("Bias is not supported in static persistent CuTile RMSNorm")

            def ceil_div(a, b):
                return (a + b - 1) // b

            TILE_SIZE_M = 4  # Default value, could be made configurable
            TILE_SIZE_N = next_power_of_2(N)

            # Other block sizes are more optimal when other dimension is too large/too small
            if TILE_SIZE_N <= 1024:
                TILE_SIZE_M = 16
            elif TILE_SIZE_N >= 16384:
                TILE_SIZE_M = 2

            grid_size = min(
                NUM_SMS,
                ceil_div(M, TILE_SIZE_M) * ceil_div(N, TILE_SIZE_N),
            )
            grid = (grid_size,)
            invoke_triton_kernel(
                torch.accelerator.current_stream(x.device),
                grid,
                _rms_norm_kernel_static_persistent,
                (x_arg, y, weight, rstd, TILE_SIZE_M, TILE_SIZE_N, eps, offset),
            )
        elif mode == "multi_wave_cached":
            # Multi-wave cached mode (single tile, inputs cached in registers)
            if bias is not None:
                raise NotImplementedError("Bias is not supported in multi_wave_cached CuTile RMSNorm")

            TILE_SIZE = next_power_of_2(N)
            grid = (M,)
            invoke_triton_kernel(
                torch.accelerator.current_stream(x.device),
                grid,
                _rms_norm_kernel_multi_wave_cached,
                (x_arg, weight, y, rstd, N, eps, offset, TILE_SIZE),
            )
        elif mode == "multi_wave_reload":
            # Standard multi-wave reload mode
            if bias is not None:
                raise NotImplementedError("Bias is not supported in standard CuTile RMSNorm")

            MAX_FUSED_SIZE = 4096 // x.element_size()
            TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))
            grid = (M,)
            invoke_triton_kernel(
                torch.accelerator.current_stream(x.device),
                grid,
                _rms_norm_kernel_gather,
                (
                    x_arg,
                    weight,
                    y,
                    rstd,
                    N,
                    eps,
                    offset,
                    TILE_SIZE,
                ),
            )
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes: None, 'static_persistent', "
                f"'multi_wave_reload', 'multi_wave_cached'"
            )

        # Always save for backward (both paths now produce rstd)
        ctx.save_for_backward(x, weight, rstd)
        ctx.TILE_SIZE = next_power_of_2(N)
        ctx.eps = eps
        ctx.offset = offset

        return y.view(*x.shape)

    @staticmethod
    def backward(ctx, dy):
        """
        Persistent backward pass using grid-stride kernel.
        Supports backward from both gather and static persistent forward modes.
        """
        # Check if offset was used (backward not supported with non-zero offset)
        if ctx.offset != 0.0:
            raise NotImplementedError(
                f"Backward pass not implemented for CuTile RMSNorm with non-zero offset ({ctx.offset})"
            )

        x, weight, rstd = ctx.saved_tensors
        dx, dw = _rms_norm_backward(x, dy, weight, rstd)

        # Gradients: (x, normalized_shape, weight, eps, bias, mode, offset)
        return dx, None, dw, None, None, None, None


@register_impl("rms_norm", backend="tiletranslation")
def rms_norm(input, normalized_shape, weight, eps, bias=None, mode=None, offset=0.0, **kwargs):
    """
    Root mean square normalization implemented using CUDA Tile

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Normalization shape (for compatibility, not used)
        weight: Tensor of shape (N,)
        eps: Small constant added to variance calculation
        bias: Bias tensor of shape (N,), default is None (not supported in cutile)
        mode: Kernel selection mode (None, "static_persistent", "multi_wave_reload", "multi_wave_cached")
        offset: Offset to add to weight (default 0.0 for Llama, 1.0 for Gemma3)
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Normalized tensor with same shape as input
    """
    return _RMSNorm.apply(input, normalized_shape, weight, eps, bias, mode, offset)


class _TileRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, offset=0.0):
        """
        RMSNorm implementation using CUDA Tile

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon value for numerical stability
            offset: Offset value (default: 0.0 for standard RMSNorm, 1.0 for Gemma3)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.offset = offset

    def forward(self, hidden_states, mode=None):
        """
        Forward pass with optional mode override

        Args:
            hidden_states: Input tensor
            mode: Default is None, which means use heuristic to
                               decide which kernel mode to use for better performance
        """
        return rms_norm(
            hidden_states,
            None,
            self.weight,
            self.variance_epsilon,
            mode=mode,
            offset=self.offset,
        )

    def forward_torch(self, hidden_states):
        """PyTorch reference implementation for comparison"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.offset + self.weight) * hidden_states.to(input_dtype)

    @staticmethod
    def compute_rstd_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute rstd (reciprocal standard deviation) for RMSNorm using PyTorch. Simulates what the forward pass would save for backward."""
        x_2d = x.reshape(-1, x.shape[-1])
        x_fp32 = x_2d.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1)
        rstd = torch.rsqrt(variance + eps)
        return rstd

    @staticmethod
    def rms_norm_backward(
        x: torch.Tensor,
        dy: torch.Tensor,
        weight: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Only for testing purposes.
        """
        return _rms_norm_backward(x, dy, weight, rstd)

    @staticmethod
    def rms_norm_backward_torch(
        x: torch.Tensor,
        dy: torch.Tensor,
        weight: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standalone RMSNorm backward pass using PyTorch. This is explicitly the torch reference implementation, not the cutile implementation."""
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        dy = dy.reshape(-1, dy.shape[-1])
        M, N = x.shape

        # Reshape rstd for broadcasting: (M,) -> (M, 1)
        rstd = rstd.view(M, 1)

        # Cast to fp32 up front so all intermediates are full precision
        x_f = x.float()
        dy_f = dy.float()
        w_f = weight.float()

        # Gradient w.r.t. weight: dw = sum((x * rstd) * dy, dim=0)
        x_norm = x_f * rstd
        dw = (dy_f * x_norm).sum(dim=0)

        # Gradient w.r.t. x
        dy_weighted = dy_f * w_f
        c1 = (dy_weighted * x_norm).sum(dim=1, keepdim=True)
        dx = rstd * (dy_weighted - x_norm * c1 / N)

        dx = dx.view(x_shape).to(x.dtype)
        dw = dw.to(weight.dtype)

        return dx, dw

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}"


class _RMSNormForGemma3(_TileRMSNorm):
    """
    RMSNorm implementation for Gemma3 models using CuTile backend.

    Gemma3 uses 'dim' parameter name instead of 'hidden_size', and initializes
    weights with zeros instead of ones, with offset=1.0.
    """

    def __init__(self, dim, eps=0.000001, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False):
        # Initialize parent with offset
        super().__init__(hidden_size=dim, eps=eps, offset=offset)
        # Override weight initialization to zeros for Gemma3
        self.weight = nn.Parameter(torch.zeros(dim))

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}"


@register_impl("get_rms_norm_module", backend="tiletranslation")
def get_rms_norm_module(model: str = "llama"):
    if model == "gemma3":
        return _RMSNormForGemma3
    else:
        return _TileRMSNorm
