# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import time

import torch
import triton
from torch.profiler import ProfilerActivity
from torch.profiler import profile

PROFILE_MODE_ENV = "TILEGYM_BENCH_PROFILE_MODE"
SPTILE_PREPARED_BENCHMARK_ENV = "TILEGYM_SPTILE_PREPARED_BENCHMARK"
DEFAULT_PROFILE_MODE = "wall"
PROFILE_MODES = {"triton", "triton_cudagraph", "wall", "l2flush"}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def get_benchmark_profile_mode(default: str = DEFAULT_PROFILE_MODE) -> str:
    mode = os.environ.get(PROFILE_MODE_ENV, default).lower()
    if mode not in PROFILE_MODES:
        raise ValueError(
            f"Unknown benchmark profile mode {mode!r}. "
            f"Set {PROFILE_MODE_ENV} to one of {sorted(PROFILE_MODES)}."
        )
    return mode


def use_sptile_prepared_benchmark(default: bool = True) -> bool:
    value = os.environ.get(SPTILE_PREPARED_BENCHMARK_ENV)
    if value is None:
        return default
    return value.lower() not in FALSE_ENV_VALUES


def _synchronize_if_cuda_available(enabled: bool = True) -> None: #TODO: Implement proper synchronization for other backends
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _summarize_wall_times(times_ms, quantiles, return_mode):
    times = sorted(times_ms)
    if return_mode == "all":
        return times
    if quantiles is not None:
        return tuple(times[min(len(times) - 1, max(0, int(q * (len(times) - 1))))] for q in quantiles)
    if return_mode == "min":
        return times[0]
    if return_mode == "max":
        return times[-1]
    if return_mode == "mean":
        return sum(times) / len(times)
    if return_mode == "median":
        return times[len(times) // 2]
    return sum(times) / len(times)


def do_wall_bench(fn, *, warmup: int, rep: int, quantiles=None, return_mode="mean", sync_cuda: bool = True):
    fn()
    _synchronize_if_cuda_available(sync_cuda)

    estimate_runs = 5
    estimate_start = time.perf_counter()
    for _ in range(estimate_runs):
        fn()
    _synchronize_if_cuda_available(sync_cuda)
    estimate_ms = (time.perf_counter() - estimate_start) * 1000.0 / estimate_runs

    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    for _ in range(n_warmup):
        fn()
    _synchronize_if_cuda_available(sync_cuda)

    times_ms = []
    for _ in range(n_repeat):
        _synchronize_if_cuda_available(sync_cuda)
        start = time.perf_counter()
        fn()
        _synchronize_if_cuda_available(sync_cuda)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return _summarize_wall_times(times_ms, quantiles, return_mode)


def profile_benchmark(
    fn,
    *,
    mode: str | None = None,
    default_mode: str = DEFAULT_PROFILE_MODE,
    warmup=25,
    rep=100,
    quantiles=None,
    return_mode="mean",
    sync_cuda: bool = True,
):
    selected_mode = mode or get_benchmark_profile_mode(default_mode)
    if selected_mode == "triton":
        return triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
            return_mode=return_mode,
        )
    if selected_mode == "triton_cudagraph":
        return triton.testing.do_bench_cudagraph(
            fn,
            rep=rep,
            quantiles=quantiles,
            return_mode=return_mode,
        )
    if selected_mode == "wall":
        return do_wall_bench(
            fn,
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
            return_mode=return_mode,
            sync_cuda=sync_cuda,
        )
    if selected_mode == "l2flush":
        if quantiles is not None or return_mode != "mean":
            raise ValueError("l2flush profiling currently returns a single mean/median-style runtime.")
        return profile_with_l2flush(fn, warmup=warmup, rep=rep)
    raise ValueError(f"Unknown benchmark profile mode {selected_mode!r}")


def profile_with_l2flush(fn, warmup=20, rep=100):
    """Measure kernel-only GPU time using torch.profiler, with L2 flush between reps."""
    l2_size = torch.cuda.get_device_properties(0).L2_cache_size
    l2_flush = torch.empty(l2_size // 4, dtype=torch.float32, device="cuda")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(rep):
        l2_flush.zero_()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        kernel_us = sum(evt.device_time_total for evt in prof.key_averages() if evt.device_time_total > 0)
        times_us.append(kernel_us)

    times_us.sort()
    median_us = times_us[len(times_us) // 2]
    return median_us / 1000.0  # ms
