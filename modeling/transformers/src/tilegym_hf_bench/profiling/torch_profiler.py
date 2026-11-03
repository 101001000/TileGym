# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import datetime
import os
import zipfile
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function

from tilegym_hf_bench.kernel_filters import KernelFilter


def run_torch_profiler(forward_wrapper, args, case_id, avg_time, summary_line):
    print("Profile the model...")
    device = torch.accelerator.current_accelerator(check_available=True)
    if device is None:
        raise RuntimeError("No GPU accelerator available")

    device_type = device.type
    is_mps = device_type == "mps"
    is_nvidia = device_type == "cuda" and torch.version.hip is None
    device_name = {"cuda": "CUDA", "xpu": "XPU", "mps": "MPS"}[device_type]

    if os.environ.get("TILEGYM_HF_BENCH_UNDER_NSIGHT") == "1" and is_nvidia:
        # Under `nsys profile --capture-range=cudaProfilerApi`: skip the kineto
        # (torch.profiler) pass. Its CUPTI client conflicts with the CUPTI that
        # nsys injects into the same process; only emit the cudaProfilerStart/Stop
        # window that nsys captures.
        with torch.no_grad():
            torch.cuda.cudart().cudaProfilerStart()
            _ = forward_wrapper.forward()
            torch.cuda.cudart().cudaProfilerStop()
        return

    activities = [ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    elif device_type == "xpu":
        activities.append(ProfilerActivity.XPU)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    mps_capture = (
        torch.mps.profiler.metal_capture(f"{log_dir}/trace_{case_id}_{timestamp}.gputrace")
        if is_mps and torch.mps.profiler.is_metal_capture_enabled()
        else nullcontext()
    )
    mps_profile = torch.mps.profiler.profile() if is_mps else nullcontext()

    with profile(
        activities=activities,
        with_stack=False,
        record_shapes=False,
    ) as prof:
        with mps_capture, mps_profile:
            with record_function("model_inference"):
                with torch.no_grad():
                    _ = forward_wrapper.forward()
                    torch.accelerator.synchronize()

    if is_nvidia:
        with torch.no_grad():
            torch.cuda.cudart().cudaProfilerStart()
            _ = forward_wrapper.forward()
            torch.cuda.cudart().cudaProfilerStop()

    filtered_results = []
    kernel_filter = KernelFilter()

    for item in prof.key_averages():
        if kernel_filter.contains(item.key):
            filtered_results.append(item)

    if filtered_results:
        print("\n===== FILTERED PROFILER RESULTS =====")
        headers = [
            "Name",
            "CPU time total (us)",
            "CPU time avg (us)",
            f"{device_name} time total (us)",
            f"{device_name} time avg (us)",
            "Count",
        ]
        row_format = "{:<80} {:<20} {:<20} {:<20} {:<20} {:<10}"

        print(row_format.format(*headers))
        print("-" * 140)
        total_device_time = 0.0
        for item in filtered_results:
            if kernel_filter.contains(item.key) and not is_mps:
                total_device_time += item.device_time_total
            print(
                row_format.format(
                    item.key[:50],
                    f"{item.cpu_time_total:.2f}",
                    f"{item.cpu_time:.2f}",
                    "N/A" if is_mps else f"{item.device_time_total:.2f}",
                    "N/A" if is_mps else f"{item.device_time:.2f}",
                    item.count,
                )
            )
        if not is_mps:
            total_device_time_pct = 0.0 if avg_time <= 0 else (total_device_time / 1_000_000.0) / avg_time * 100.0
            print(
                f"Total device time: {total_device_time:.2f} us",
                f"Total device time %: {total_device_time_pct:.2f}%",
            )
    print(prof.key_averages().table(row_limit=1))

    all_results = prof.key_averages()
    all_results = sorted(all_results, key=lambda x: x.device_time_total, reverse=True)

    filename = f"{log_dir}/profiler_results_{case_id}_{timestamp}.csv"

    with open(filename, "w") as f:
        f.write(
            f"Name,CPU_time_total_us,CPU_time_avg_us,{device_name}_time_total_us,{device_name}_time_avg_us,Self_CPU_time_total_us,Self_{device_name}_time_total_us,Count,Filtered,DeviceType\n"
        )

        for item in all_results:
            device_type_str = str(item.device_type) if hasattr(item, "device_type") else ""
            device_total = "N/A" if is_mps else f"{item.device_time_total:.2f}"
            device_avg = "N/A" if is_mps else f"{item.device_time:.2f}"
            self_device_total = "N/A" if is_mps else f"{item.self_device_time_total:.2f}"
            f.write(
                f'"{item.key}",{item.cpu_time_total:.2f},{item.cpu_time:.2f},{device_total},{device_avg},{item.self_cpu_time_total:.2f},{self_device_total},{item.count},{kernel_filter.contains(item.key)},{device_type_str}\n'
            )
    trace_filename = f"{log_dir}/trace_{case_id}_{timestamp}.json"
    prof.export_chrome_trace(trace_filename)

    zip_filename = f"{log_dir}/trace_{case_id}_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(trace_filename)
    os.remove(trace_filename)

    print(f"Trace file zipped to {zip_filename}")
    print(f"Profiler results saved to {filename}")
    print(prof.key_averages().table(row_limit=1))

    if args.summary_file:
        device_kernel_time_us = 0.0
        if not is_mps:
            for item in all_results:
                device_type_str = str(item.device_type) if hasattr(item, "device_type") else ""
                if device_type_str == f"DeviceType.{device_name}" and item.key != "model_inference":
                    device_kernel_time_us += item.self_device_time_total

        device_kernel_time_ms = "N/A" if is_mps else f"{device_kernel_time_us / 1000.0:.1f}"
        updated_summary_line = f"{summary_line} | {device_kernel_time_ms:>10}\n"

        lines = []
        if os.path.exists(args.summary_file):
            with open(args.summary_file, encoding="utf-8") as f:
                lines = f.readlines()

        if lines:
            lines[-1] = updated_summary_line
        else:
            lines = [updated_summary_line]

        with open(args.summary_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        if is_mps:
            print("Updated summary with MPS kernel time: N/A")
        else:
            print(f"Updated summary with {device_name} kernel time: {device_kernel_time_ms} ms")