# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
torch.set_float32_matmul_precision("highest")

import triton
from bench_utils import profile_benchmark
from bench_utils import use_sptile_prepared_benchmark

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = None,  # Unused - kept for interface compatibility
    trans_b: bool = None,  # Unused - kept for interface compatibility
    static_persistent: bool = True,  # Unused - kept for interface compatibility
    use_tma: bool = True,  # Unused - kept for interface compatibility
):
    """Reference implementation using PyTorch"""
    return torch.matmul(a, b)


register_impl("matmul", "torch")(reference_matmul)


# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("tilecpp", "TileCpp", ("purple", "-")) if is_backend_available("tilecpp") else None,
    ("sptile", "SpTile", ("red", "-")) if is_backend_available("sptile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends(datatype):
    """Filter backends based on datatype support and availability"""
    # Torch cannot support FP8, so exclude it for FP8 datatypes
    candidates = ALL_BACKENDS[:-1] if datatype == torch.float8_e5m2 else ALL_BACKENDS

    return [
        p for p in candidates
        if p is not None and (p[0] != "sptile" or datatype == torch.float32)
    ]


def create_benchmark_config(datatype):
    """Create a benchmark configuration for given datatype and backends"""
    available_backends = get_supported_backends(datatype)
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]  # e.g., 'float16' from 'torch.float16'
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] == 10:
        max_range = 16
    else:
        max_range = 14  # To avoid OOM
    return triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[2**i for i in range(10, max_range)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        xlabel="M/N/K",
        ylabel="TFLOPS",
        plot_name=f"matmul-performance-{dtype_name}-TFLOPS",
        args={"datatype": datatype},
    )


# FP8 requires SM90+ (Hopper/Blackwell); skip on SM80 (A100)
_gpu_cap = torch.cuda.get_device_capability()
#_dtypes = [torch.float16] if _gpu_cap[0] < 9 else [torch.float16, torch.float8_e5m2]

_dtypes = [torch.float32] # spTile only supports float32, so we benchmark only float32 for now

@triton.testing.perf_report([create_benchmark_config(datatype) for datatype in _dtypes])
def benchmark(M, N, K, backend, datatype):
    print("Testing matmul with M={}, N={}, K={}, backend={}, datatype={}".format(M, N, K, backend, datatype))
    if datatype == torch.float8_e5m2:
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
    else:
        a = torch.randn((M, K), device=DEVICE, dtype=datatype)
        b = torch.randn((K, N), device=DEVICE, dtype=datatype)

    quantiles = [0.5, 0.2, 0.8]

    normal_fn = lambda: tilegym.ops.matmul(a, b, use_tma=True, static_persistent=False, backend=backend)

    if datatype != torch.float8_e5m2:
        # torch doesn't support FP8 matmul
        ref = lambda: reference_matmul(a, b)
        torch.testing.assert_close(normal_fn(), ref(), rtol=1e-4, atol=1e-3)

    use_prepared_sptile = backend == "sptile" and use_sptile_prepared_benchmark()
    if use_prepared_sptile:
        from tilegym.ops.sptile.matmul import prepare_matmul_benchmark

        fn = prepare_matmul_benchmark(
            a,
            b,
            static_persistent=False,
        )
    else:
        fn = normal_fn

    ms, min_ms, max_ms = profile_benchmark(
        fn,
        quantiles=quantiles,
        sync_cuda=not use_prepared_sptile,
    )
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True)
