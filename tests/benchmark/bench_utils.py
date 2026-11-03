# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
from torch.profiler import profile, ProfilerActivity


def profile_with_l2flush(fn, warmup=20, rep=100, device=None):
    """Measure GPU time using torch.profiler, with cache flush between reps."""
    device = torch.device(device or torch.accelerator.current_accelerator(check_available=True))

    if device.type == "cuda":
        l2_size = torch.cuda.get_device_properties(device).L2_cache_size
        activity = ProfilerActivity.CUDA
    elif device.type == "xpu":
        l2_size = torch.xpu.get_device_properties(device).last_level_cache_size
        activity = ProfilerActivity.XPU
    else:
        raise ValueError(f"Unsupported device: {device}")

    l2_flush = torch.empty(l2_size // 4, dtype=torch.float32, device=device)

    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize(device)

    times_us = []
    for _ in range(rep):
        l2_flush.zero_()
        torch.accelerator.synchronize(device)

        with profile(activities=[activity]) as prof:
            fn()
            torch.accelerator.synchronize(device)

        kernel_us = sum(
            evt.device_time_total
            for evt in prof.key_averages()
            if evt.device_time_total > 0
        )
        times_us.append(kernel_us)

    times_us.sort()
    return times_us[len(times_us) // 2] / 1000.0
