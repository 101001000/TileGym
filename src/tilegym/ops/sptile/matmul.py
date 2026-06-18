# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any
from typing import Callable

import cuda.tile as ct
import cuda.tile.compilation as ct_compilation
import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.logger import get_logger

from .runtime import as_pinned_cpu_tensor
from .runtime import pop_runtime_options
from .runtime import PreparedSpirvKernel
from .runtime import register_tilebc_compiler
from .runtime import return_to_original_device
from .runtime import resolve_spirv_path
from .runtime import run_spirv_kernel
from .runtime import tensor_2d_descriptor

logger = get_logger(__name__)

KERNEL_NAME = "matmul"
STATIC_PERSISTENT_KERNEL_NAME = "matmul_static_persistent"
DEFAULT_ENTRY_NAME = "matmul_fp32"
STATIC_PERSISTENT_ENTRY_NAME = "matmul_static_persistent_fp32"
OP_NAME = "sptile.matmul"
LOCAL_SIZE_X = 1024
MATMUL_TILE_M = 32
MATMUL_TILE_N = 32
MATMUL_TILE_K = 32
STATIC_PERSISTENT_GROUP_SIZE_M = 8
STATIC_PERSISTENT_MAX_PROGRAMS = 132
_TILEBC_GPU_CODE = "sm_100"
ConstInt = ct.Constant[int]


def _swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M):
    bid = ct.bid(0)
    return _swizzle_2d_from_bid(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M, bid)


def _swizzle_2d_from_bid(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M, bid):
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel
def _matmul_kernel(
    A,
    B,
    C,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = _swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))
    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(TILE_SIZE_M, TILE_SIZE_K), padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(TILE_SIZE_K, TILE_SIZE_N), padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)

    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


@ct.kernel
def _static_persistent_matmul_kernel(
    A,
    B,
    C,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
):
    GROUP_SIZE_M = STATIC_PERSISTENT_GROUP_SIZE_M
    bid = ct.bid(0)
    M = A.shape[0]
    N = B.shape[1]
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    output_tile_count = num_bid_m * num_bid_n
    num_programs = ct.num_blocks(0)

    for tile_id in range(bid, output_tile_count, num_programs):
        accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0, dtype=ct.float32)
        bidx, bidy = _swizzle_2d_from_bid(
            M,
            N,
            TILE_SIZE_M,
            TILE_SIZE_N,
            GROUP_SIZE_M,
            tile_id,
        )

        for k in range(num_tiles_k):
            a = ct.load(A, index=(bidx, k), shape=(TILE_SIZE_M, TILE_SIZE_K), padding_mode=zero_pad).astype(dtype)
            b = ct.load(B, index=(k, bidy), shape=(TILE_SIZE_K, TILE_SIZE_N), padding_mode=zero_pad).astype(dtype)
            accumulator = ct.mma(a, b, accumulator)

        accumulator = ct.astype(accumulator, C.dtype)
        ct.store(C, index=(bidx, bidy), tile=accumulator)


def _make_contiguous_2d_constraint(dtype):
    return ct_compilation.ArrayConstraint(
        dtype,
        2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0, 0),
        alias_groups=[],
        may_alias_internally=False,
        stride_constant=(None, 1),
        shape_divisible_by=(1, 1),
        base_addr_divisible_by=1,
    )


def _make_matmul_signature(dtype, tile_size_m, tile_size_n, tile_size_k, symbol):
    array_constraint = _make_contiguous_2d_constraint(dtype)
    return ct_compilation.KernelSignature(
        parameters=[
            array_constraint,
            array_constraint,
            array_constraint,
            tile_size_m,
            tile_size_n,
            tile_size_k,
        ],
        calling_convention=ct_compilation.CallingConvention.cutile_python_v1(),
        symbol=symbol,
    )


@register_tilebc_compiler(KERNEL_NAME)
def compile_matmul_tilebc(
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fp32_signature = _make_matmul_signature(
        dtype=ct.float32,
        tile_size_m=MATMUL_TILE_M,
        tile_size_n=MATMUL_TILE_N,
        tile_size_k=MATMUL_TILE_K,
        symbol=DEFAULT_ENTRY_NAME,
    )
    ct_compilation.export_kernel(
        _matmul_kernel,
        signatures=[fp32_signature],
        output_file=str(output),
        gpu_code=_TILEBC_GPU_CODE,
        output_format="tileir_bytecode",
    )
    return output


@register_tilebc_compiler(STATIC_PERSISTENT_KERNEL_NAME)
def compile_static_persistent_matmul_tilebc(
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fp32_signature = _make_matmul_signature(
        dtype=ct.float32,
        tile_size_m=MATMUL_TILE_M,
        tile_size_n=MATMUL_TILE_N,
        tile_size_k=MATMUL_TILE_K,
        symbol=STATIC_PERSISTENT_ENTRY_NAME,
    )
    ct_compilation.export_kernel(
        _static_persistent_matmul_kernel,
        signatures=[fp32_signature],
        output_file=str(output),
        gpu_code=_TILEBC_GPU_CODE,
        output_format="tileir_bytecode",
    )
    return output


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _torch_to_f32_numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor.")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be torch.float32; the current SPIR-V lowering is f32-only.")
    return tensor.detach().cpu().contiguous().numpy()


def _matmul_dispatch(m: int, n: int) -> tuple[int, int, int]:
    tile_size = MATMUL_TILE_M * MATMUL_TILE_N
    output_tile_count = _ceil_div(m, MATMUL_TILE_M) * _ceil_div(n, MATMUL_TILE_N)
    invocation_count = output_tile_count * tile_size
    return (_ceil_div(invocation_count, LOCAL_SIZE_X), 1, 1)


def _static_persistent_program_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return STATIC_PERSISTENT_MAX_PROGRAMS


def _static_persistent_matmul_dispatch(m: int, n: int) -> tuple[int, int, int]:
    output_tile_count = _ceil_div(m, MATMUL_TILE_M) * _ceil_div(n, MATMUL_TILE_N)
    program_count = min(_static_persistent_program_count(), output_tile_count)
    return (max(1, program_count), 1, 1)


def _dispatch_for_shape(m: int, n: int, *, static_persistent: bool) -> tuple[int, int, int]:
    if static_persistent:
        return _static_persistent_matmul_dispatch(m, n)
    return _matmul_dispatch(m, n)


def _matmul_parameter_arrays(A: torch.Tensor, B: torch.Tensor) -> tuple[list[np.ndarray], int, int]:
    if A.device != B.device:
        raise ValueError("Input tensors must be on the same torch device.")

    a = _torch_to_f32_numpy(A, "A")
    b = _torch_to_f32_numpy(B, "B")
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError("Input matrices have incompatible K dimensions.")

    c = np.zeros((m, n), dtype=np.float32)
    return [
        *tensor_2d_descriptor(a),
        *tensor_2d_descriptor(b),
        *tensor_2d_descriptor(c),
    ], m, n


def _run_spirv_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    spirv_path: str | Path,
    entry_name: str = DEFAULT_ENTRY_NAME,
    static_persistent: bool = False,
) -> torch.Tensor:
    original_device = A.device
    parameter_arrays, m, n = _matmul_parameter_arrays(A, B)
    output_flat = run_spirv_kernel(
        spirv_path=spirv_path,
        entry_name=entry_name,
        parameter_arrays=parameter_arrays,
        output_indices=[10],
        dispatch=_dispatch_for_shape(m, n, static_persistent=static_persistent),
    )[0]
    output = output_flat.reshape(m, n).copy()

    result = torch.from_numpy(output)
    if original_device.type != "cpu":
        result = result.to(original_device)
    return result


def prepare_matmul_benchmark(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    spirv_path: str | Path | None = None,
    tilebc_path: str | Path | None = None,
    translator_path: str | Path | None = None,
    sptile_cache_dir: str | Path | None = None,
    force_retranslate: bool = False,
    entry_name: str = DEFAULT_ENTRY_NAME,
    static_persistent: bool = False,
) -> Callable[[], None]:
    kernel_name = STATIC_PERSISTENT_KERNEL_NAME if static_persistent else KERNEL_NAME
    resolved_spirv_path = resolve_spirv_path(
        kernel_name=kernel_name,
        spirv_path=spirv_path,
        tilebc_path=tilebc_path,
        translator_path=translator_path,
        cache_dir=sptile_cache_dir,
        force_retranslate=force_retranslate,
    )

    parameter_arrays, m, n = _matmul_parameter_arrays(A, B)
    prepared = PreparedSpirvKernel(
        spirv_path=resolved_spirv_path,
        entry_name=entry_name,
        parameter_arrays=parameter_arrays,
        output_indices=[10],
        dispatch=_dispatch_for_shape(m, n, static_persistent=static_persistent),
    )
    return prepared.dispatch


def _validate_matmul_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> None:
    if trans_a or trans_b:
        raise NotImplementedError("sptile backend currently supports only trans_a=False and trans_b=False")

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("sptile.matmul only supports 2D tensors")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("sptile.matmul currently supports only torch.float32 tensors")
    if a.device != b.device:
        raise ValueError("Input tensors must be on the same torch device.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible matrices: A.shape[1]={a.shape[1]} must equal B.shape[0]={b.shape[0]}")


@register_impl("matmul", backend="sptile")
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    static_persistent: bool | None = None,
    use_tma: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    if static_persistent is None:
        static_persistent = False

    _validate_matmul_inputs(a, b, trans_a, trans_b)

    kernel_name = STATIC_PERSISTENT_KERNEL_NAME if static_persistent else KERNEL_NAME
    default_entry_name = STATIC_PERSISTENT_ENTRY_NAME if static_persistent else DEFAULT_ENTRY_NAME
    runtime_options, entry_name = pop_runtime_options(kwargs, default_entry_name=default_entry_name)
    if kwargs:
        raise TypeError(f"Unsupported kwargs for sptile.matmul: {sorted(kwargs)}")

    try:
        spirv_path = resolve_spirv_path(kernel_name=kernel_name, **runtime_options)
        original_device = a.device
        result = _run_spirv_matmul(
            as_pinned_cpu_tensor(a, f"{kernel_name}:input_a", op_name=OP_NAME),
            as_pinned_cpu_tensor(b, f"{kernel_name}:input_b", op_name=OP_NAME),
            spirv_path=spirv_path,
            entry_name=entry_name,
            static_persistent=static_persistent,
        )
        return return_to_original_device(
            result,
            original_device,
            role=f"{kernel_name}:output",
            op_name=OP_NAME,
        )
    except Exception as exc:
        logger.error(f"sptile.matmul failed: {exc.__class__.__name__}")
        if static_persistent and "cuda_tile.token" in str(exc):
            raise RuntimeError(
                "sptile static_persistent matmul is implemented and compiles to cuTile bytecode, "
                "but the current spTiles translator cannot lower the persistent outer loop to SPIR-V "
                "(loop-carried cuda_tile.token around store). Update sp_tiles_translate or pass "
                "static_persistent=False to use the non-persistent SPIR-V kernel."
            ) from exc
        raise RuntimeError(
            "Failed to execute sptile SPIR-V matmul. Ensure spTiles is available "
            "and that SPTILES_TRANSLATE_BIN points to the sp_tiles_translate binary."
        ) from exc
