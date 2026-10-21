# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from math import ceil
from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilegym.ops.cutile.matmul import _matmul_kernel
from tilegym.ops.cutile.matmul import _static_persistent_matmul_kernel

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl
from tilegym.logger import get_logger

from .translator import get_capability, exhaustive_search_triton, get_sms
from .translator import prepare_triton_kernel, invoke_triton_kernel, _launch_args
from functools import lru_cache

# Module-level tune caches: (M, N, K, dtype, device) -> (best_cfg, tuned_kernel)
_matmul_tune_cache: dict = {}
_static_persistent_matmul_tune_cache: dict = {}

logger = get_logger(__name__)

ConstInt = ct.Constant[int]

_FP64_MATMUL_CONFIGS = {
    (8, 0): ((64, 64, 64, 1, 2),),
    (9, 0): ((16, 16, 32, 1, 2), (128, 64, 16, 1, 2)),
    (10, 0): ((16, 16, 32, 1, 2), (64, 64, 16, 1, 2)),
    (10, 3): ((16, 16, 32, 1, 2),),
    (10, 7): ((16, 16, 32, 1, 2), (64, 64, 16, 1, 2), (128, 64, 16, 1, 2)),
    (12, 0): ((64, 64, 32, 1, 2),),
    (12, 1): ((64, 64, 32, 1, 2),),
}

_FP64_PERSISTENT_MATMUL_CONFIGS = {
    (9, 0): ((64, 128, 16), (32, 16, 32), (128, 64, 16)),
    (10, 0): ((64, 128, 16), (32, 16, 32)),
    (10, 3): ((32, 16, 32),),
    (10, 7): ((32, 16, 32), (64, 128, 16), (128, 64, 16)),
    (12, 0): ((64, 64, 64),),
    (12, 1): ((64, 64, 64),),
}


def _matmul_autotune_configs(dtype):
    """
    Iterator of autotune configurations for matmul kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if dtype == torch.float64 and gpu_capability in _FP64_MATMUL_CONFIGS:
        for TILE_M, TILE_N, TILE_K, num_ctas, occupancy in _FP64_MATMUL_CONFIGS[gpu_capability]:
            yield SimpleNamespace(
                TILE_SIZE_M=TILE_M,
                TILE_SIZE_N=TILE_N,
                TILE_SIZE_K=TILE_K,
                num_ctas=num_ctas,
                occupancy=occupancy,
            )
        return

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=32, num_ctas=1, occupancy=2)
    elif gpu_capability[0] < 9:
        # Pre-SM90: num_ctas=1 (CGA unsupported); sweep TILE_K in [32, 64, 128]
        for TILE_M in [64, 128]:
            for TILE_N in [64, 128]:
                for TILE_K in [32, 64, 128]:
                    for occ in [1, 2]:
                        yield SimpleNamespace(
                            TILE_SIZE_M=TILE_M, TILE_SIZE_N=TILE_N, TILE_SIZE_K=TILE_K, num_ctas=1, occupancy=occ
                        )
    else:
        # sm100+ (Blackwell)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=4, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=512, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1)


def _static_persistent_matmul_autotune_configs(dtype):
    """
    Iterator of autotune configurations for static persistent matmul kernel.
    """
    gpu_capability = get_capability()

    # LOAD_LATENCY = ct.load cost hint (1..10, -1 = compiler-inferred). Only sm90 tunes it
    # today; all other arches pass -1 (compiler-inferred = original behavior), but every
    # config must carry the field since the kernel reads cfg.LOAD_LATENCY unconditionally.
    if dtype == torch.float64 and gpu_capability in _FP64_PERSISTENT_MATMUL_CONFIGS:
        for TILE_M, TILE_N, TILE_K in _FP64_PERSISTENT_MATMUL_CONFIGS[gpu_capability]:
            yield SimpleNamespace(
                TILE_SIZE_M=TILE_M,
                TILE_SIZE_N=TILE_N,
                TILE_SIZE_K=TILE_K,
                GROUP_SIZE_M=8,
                num_ctas=1,
                occupancy=2,
                LOAD_LATENCY=-1,
            )
        return

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        yield SimpleNamespace(
            TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=4, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=4, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1, LOAD_LATENCY=-1
        )
        # Narrow-K (K=32) / wide-N candidates for extreme-skinny shapes (large M,
        # small N/K), where the K=64 tiles above underutilize the persistent grid.
        yield SimpleNamespace(
            TILE_SIZE_M=64, TILE_SIZE_N=128, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=1, LOAD_LATENCY=-1
        )
    elif gpu_capability[0] < 9:
        # sm80 (A100)
        yield SimpleNamespace(
            TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=64, TILE_SIZE_N=128, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=1, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=2, LOAD_LATENCY=-1
        )
    else:
        # sm100+ (Blackwell)
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=512, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=4, occupancy=1, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=2, occupancy=1, LOAD_LATENCY=-1
        )
        # Wide-N and load-cost-tuned tiles for extreme-skinny shapes (large M, small
        # N/K), where the square tiles above leave the persistent grid underutilized.
        yield SimpleNamespace(
            TILE_SIZE_M=256, TILE_SIZE_N=512, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=2, occupancy=1, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=2, occupancy=1, LOAD_LATENCY=4
        )
        yield SimpleNamespace(
            TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1, LOAD_LATENCY=-1
        )
        yield SimpleNamespace(
            TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=128, GROUP_SIZE_M=8, num_ctas=2, occupancy=1, LOAD_LATENCY=-1
        )
        # Narrow-K tile for extreme-skinny float32 shapes: fp32 doubles per-tile shared
        # memory, so the K=64 tiles above exceed the smem budget and are unrealizable,
        # leaving no wide/large-tile candidate for the skinny fp32 case without this one.
        yield SimpleNamespace(
            TILE_SIZE_M=256, TILE_SIZE_N=128, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=2, occupancy=1, LOAD_LATENCY=-1
        )
        # Small-tile candidate for small/rectangular GEMMs, where the entries above cap
        # the persistent grid at 16-32 tile-jobs and strand most SMs.
        yield SimpleNamespace(
            TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1, LOAD_LATENCY=-1
        )

def _tiletranslation_autotune_matmul(stream, a, b, c):
    M, N = c.shape
    K = a.shape[1]
    cache_key = (M, N, K, a.dtype, str(a.device))
    if cache_key not in _matmul_tune_cache:
        configs = list(_matmul_autotune_configs(a.dtype))
        if is_autotune_disabled():
            configs = configs[:1]
        #with ct.compiler_timeout(5):
        result = exhaustive_search_triton(
            configs,
            stream,
            lambda cfg: (ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N), 1, 1),
            _matmul_kernel,
            lambda cfg: (a, b, c, cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, cfg.TILE_SIZE_K),
            lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _matmul_tune_cache[cache_key] = result.best.config

    best_cfg = _matmul_tune_cache[cache_key]

    kernel_args = (
        a, b, c,
        best_cfg.TILE_SIZE_M,
        best_cfg.TILE_SIZE_N,
        best_cfg.TILE_SIZE_K,
    )


    grid = (
        ceil(M / best_cfg.TILE_SIZE_M) * ceil(N / best_cfg.TILE_SIZE_N),
        1,
        1,
    )

    invoke_triton_kernel(
        _matmul_kernel,
        grid,
        *kernel_args,
        options={"num_ctas": best_cfg.num_ctas},
    )

    #ct.launch(
    #    stream,
    #    (ceil(M / best_cfg.TILE_SIZE_M) * ceil(N / best_cfg.TILE_SIZE_N), 1, 1),
    #    tuned_kernel,
    #    (a, b, c, best_cfg.TILE_SIZE_M, best_cfg.TILE_SIZE_N, best_cfg.TILE_SIZE_K),
    #)
    return c


def _tiletranslation_autotune_static_persistent_matmul(stream, a, b, c, M, N, K, trans_a, trans_b):
    NUM_SMS = get_sms()
    cache_key = (M, N, K, trans_a, trans_b, a.dtype, str(a.device))
    if cache_key not in _static_persistent_matmul_tune_cache:
        configs = list(_static_persistent_matmul_autotune_configs(a.dtype))
        if is_autotune_disabled():
            configs = configs[:1]
        #with ct.compiler_timeout(5):
        result = exhaustive_search_triton(
            configs,
            stream,
            lambda cfg: (
                min(NUM_SMS // cfg.num_ctas, ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N)) * cfg.occupancy,
                1,
                1,
            ),
            _static_persistent_matmul_kernel,
            lambda cfg: (
                a,
                b,
                c,
                M,
                N,
                K,
                cfg.TILE_SIZE_M,
                cfg.TILE_SIZE_N,
                cfg.TILE_SIZE_K,
                trans_a,
                trans_b,
                cfg.GROUP_SIZE_M,
                cfg.LOAD_LATENCY,
            ),
            lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _static_persistent_matmul_tune_cache[cache_key] = result.best.config

    best_cfg = _static_persistent_matmul_tune_cache[cache_key]
    #ct.launch(
    #    stream,
    #    (
    #        min(NUM_SMS // best_cfg.num_ctas, ceil(M / best_cfg.TILE_SIZE_M) * ceil(N / best_cfg.TILE_SIZE_N))
    #        * best_cfg.occupancy,
    #        1,
    #        1,
    #    ),
    #    tuned_kernel,
    #    (
    #        a,
    #        b,
    #        c,
    #        M,
    #        N,
    #        K,
    #        best_cfg.TILE_SIZE_M,
    #        best_cfg.TILE_SIZE_N,
    #        best_cfg.TILE_SIZE_K,
    #        trans_a,
    #        trans_b,
    #        best_cfg.GROUP_SIZE_M,
    #        best_cfg.LOAD_LATENCY,
    #    ),
    #)

    kernel_args = (
        a, b, c, M, N, K,
        best_cfg.TILE_SIZE_M,
        best_cfg.TILE_SIZE_N,
        best_cfg.TILE_SIZE_K,
        trans_a,
        trans_b,
        best_cfg.GROUP_SIZE_M,
        best_cfg.LOAD_LATENCY,
    )

    grid = (
        min(
            NUM_SMS // best_cfg.num_ctas,
            ceil(M / best_cfg.TILE_SIZE_M) * ceil(N / best_cfg.TILE_SIZE_N),
        ) * best_cfg.occupancy,
        1,
        1,
    )

    invoke_triton_kernel(
        _static_persistent_matmul_kernel,
        grid,
        *kernel_args,
        options={"num_ctas": best_cfg.num_ctas},
    )

    return c


@register_impl("matmul", backend="tiletranslation")
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a=False,
    trans_b=False,
    static_persistent=None,
    use_tma=False,
    **kwargs,
):
    if static_persistent is None:
        static_persistent = False

    # Get matrix dimensions
    if trans_a:
        K, M = a.shape
    else:
        M, K = a.shape
    if trans_b:
        N, KB = b.shape
    else:
        KB, N = b.shape
    assert K == KB, f"Incompatible matrices: K dimension of A is {K}, K dimension of B is {KB}"

    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stream = torch.accelerator.current_stream(a.device)
    if static_persistent:
        _tiletranslation_autotune_static_persistent_matmul(stream, a, b, c, M, N, K, trans_a, trans_b)
    else:
        assert trans_a == False, "trans_a is not supported for tiletranslation"
        assert trans_b == False, "trans_b is not supported for tiletranslation"
        _tiletranslation_autotune_matmul(stream, a, b, c)
    return c
