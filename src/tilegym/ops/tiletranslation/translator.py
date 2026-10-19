import hashlib
import inspect
import math
import os
import subprocess
from pathlib import Path

import torch
import triton

import cuda.tile as ct
from cuda.tile.compilation import CallingConvention, KernelSignature, export_kernel
from cuda.tile.compilation._signature import ArrayConstraint, ConstantConstraint, ScalarConstraint
from cuda.tile._annotated_function import HomogeneousTupleNode, HeterogeneousTupleNode

from tilegym.ops.cache_utils import get_cache_dir

DEFAULT_SM_ARCH = "sm_120"

# torch.dtype -> cuda.tile DType (covers the dtypes TileGym kernels use)
_TORCH_TO_CT_DTYPE = {
    torch.bool: ct.bool_,
    torch.int8: ct.int8,
    torch.int16: ct.int16,
    torch.int32: ct.int32,
    torch.int64: ct.int64,
    torch.uint8: ct.uint8,
    torch.uint16: ct.uint16,
    torch.uint32: ct.uint32,
    torch.uint64: ct.uint64,
    torch.float16: ct.float16,
    torch.bfloat16: ct.bfloat16,
    torch.float32: ct.float32,
    torch.float64: ct.float64,
}

_INT32_MAX = 2**31 - 1


def get_sm_arch() -> str:
    """Target GPU architecture for the translation pipeline.

    Read from the ``TILEGYM_SM_ARCH`` environment variable (e.g. ``sm_120``),
    defaulting to ``sm_120``. Kept out of the per-op prepare paths so the
    target is a pipeline-level decision rather than derived from the current
    device at each call site.
    """
    return os.environ.get("TILEGYM_SM_ARCH", DEFAULT_SM_ARCH)


def _array_constraint_from_tensor(t: torch.Tensor) -> ArrayConstraint:
    """Build an ArrayConstraint from tensor metadata alone (no CUDA driver).

    Mirrors what ``KernelSignature.from_kernel_args`` derives from example
    arguments, using only dtype/shape/strides:

    - ``stride_constant[i] = 1`` for unit strides (exact, safe).
    - ``stride_lower_bound_incl[i] = 0`` for strides > 1.
    - ``stride_divisible_by[i] = gcd(stride[i], 16 / itemsize)`` (the same
      16-byte cap the CUDA-path constraint derivation uses).
    - ``base_addr_divisible_by = 16`` bytes: torch allocations are >= 256B
      aligned on every backend, matching the CUDA-path derivation. Floor of 2
      because ``convert-cuda-tile-to-triton`` asserts on a base alignment of 1.
      Override with ``TILEGYM_BASE_ADDR_ALIGN`` if an op must accept
      sub-16B-aligned bases.
    - ``may_alias_internally`` when a zero stride is present.
    - ``index_dtype`` is int32 unless a shape/stride exceeds int32 range.

    ``shape_divisible_by`` stays at 1 (no verifiable assumption).
    """
    try:
        dtype = _TORCH_TO_CT_DTYPE[t.dtype]
    except KeyError as e:
        raise TypeError(f"Unsupported torch dtype for signature generation: {t.dtype}") from e

    shape = t.shape
    strides = t.stride()
    index_dtype = ct.int32 if all(v <= _INT32_MAX for v in (*shape, *strides)) else ct.int64

    byte_cap = max(1, 16 // (dtype.bitwidth // 8))
    stride_div = tuple(math.gcd(s, byte_cap) for s in strides)
    base_addr_align = max(2, int(os.environ.get("TILEGYM_BASE_ADDR_ALIGN", 16)))

    return ArrayConstraint(
        dtype,
        t.dim(),
        index_dtype=index_dtype,
        stride_lower_bound_incl=tuple(0 if s > 1 else None for s in strides),
        alias_groups=(),
        may_alias_internally=any(s == 0 for s in strides),
        stride_constant=tuple(s if s == 1 else None for s in strides),
        stride_divisible_by=stride_div,
        base_addr_divisible_by=base_addr_align,
    )


def _scalar_constraint_from_value(value) -> ScalarConstraint:
    """Build a ScalarConstraint for a plain (non-Constant) scalar argument."""
    if isinstance(value, bool):
        return ScalarConstraint(ct.bool_)
    if isinstance(value, int):
        return ScalarConstraint(ct.int32 if -_INT32_MAX - 1 <= value <= _INT32_MAX else ct.int64)
    if isinstance(value, float):
        return ScalarConstraint(ct.float32)
    raise TypeError(f"Cannot derive a scalar constraint from value {value!r}")


def signature_from_args(kernel, kernel_args, calling_convention: CallingConvention) -> KernelSignature:
    """Driver-free equivalent of ``KernelSignature.from_kernel_args``.

    Builds the signature the kernel would be compiled with for the given
    arguments, without importing a CUDA context (works on machines with
    Intel/AMD GPUs or no GPU at all). Tensors become ``ArrayConstraint`` from
    metadata only; ``ct.Constant``-annotated args become
    ``ConstantConstraint``; plain scalars become ``ScalarConstraint``.
    """
    annotations = kernel._annotated_function.parameter_annotations
    if len(annotations) != len(kernel_args):
        raise ValueError(
            f"Kernel takes {len(annotations)} parameters, got {len(kernel_args)} arguments"
        )

    parameters = []
    for ann, arg in zip(annotations, kernel_args):
        if isinstance(ann, (HomogeneousTupleNode, HeterogeneousTupleNode)):
            raise NotImplementedError("Tuple kernel parameters are not supported yet")
        if isinstance(arg, torch.Tensor):
            parameters.append(_array_constraint_from_tensor(arg))
        elif ann.constant:
            parameters.append(ConstantConstraint(value=arg))
        else:
            parameters.append(_scalar_constraint_from_value(arg))

    sig = KernelSignature(parameters, calling_convention)
    return sig.with_mangled_symbol(kernel._annotated_function.pyfunc.__name__)

def validate_mlir(mlir_path):
    subprocess.run(
        [
            "cuda-tile-translate",
            "--mlir-to-cudatilebc",
            str(mlir_path),
            "-o",
            "/dev/null",
        ],
        check=True,
    )

def emit_bytecode(kernel, kernel_args, output_path, gpu_code):
    signature = signature_from_args(
        kernel,
        kernel_args,
        CallingConvention.cutile_python_v2(),
    )
    export_kernel(
        kernel,
        [signature],
        output_path,
        gpu_code=gpu_code,
        output_format="tileir_bytecode",
    )


def emit_mlir(bytecode_path, output_path):
    subprocess.run(
        [
            "cuda-tile-translate",
            "--cudatilebc-to-mlir",
            str(bytecode_path),
            "-o",
            str(output_path),
        ],
        check=True,
    )
    validate_mlir(output_path)

def translate_tileir(mlir_path, output_path):
    subprocess.run(
        [
            "triton-cuda-tile-opt",
            str(mlir_path),
            "--pass-pipeline=builtin.module(convert-cuda-tile-to-triton,reconcile-unrealized-casts)",
            "-o",
            str(output_path),
        ],
        check=True,
    )

def _kernel_source_hash(kernel) -> str:
    """Short hash of the kernel's source, to invalidate the cache on edits."""
    pyfunc = getattr(kernel, "_pyfunc", kernel)
    source = inspect.getsource(pyfunc)
    return hashlib.sha1(source.encode()).hexdigest()[:8]


def prepare_triton_kernel(kernel, kernel_args, gpu_code, op, key):
    """Translate a cuTile kernel to Triton, caching artifacts on disk.

    The ``.tilebc`` / ``.tileir.mlir`` / ``.ttir`` artifacts are stored under
    ``<TILEGYM cache root>/tiletranslation/<op>/`` named
    ``<key>.<gpu_code>.<src_hash>`` so different ops, targets and kernel
    versions never collide. When the ``.ttir`` is already cached (it is
    written last, so its presence means the pipeline completed), the
    subprocess pipeline is skipped entirely.
    """
    src_hash = _kernel_source_hash(kernel)
    cache_dir = get_cache_dir("tiletranslation", op)
    base = cache_dir / f"{key}.{gpu_code}.{src_hash}"
    bytecode_path = f"{base}.tilebc"
    mlir_path = f"{base}.tileir.mlir"
    ttir_path = f"{base}.ttir"

    if not Path(ttir_path).exists():
        emit_bytecode(kernel, kernel_args, bytecode_path, gpu_code)
        emit_mlir(bytecode_path, mlir_path)
        translate_tileir(mlir_path, ttir_path)

    return triton.compile(ttir_path)


def invoke_triton_kernel(kernel, grid, *kernel_args):
    kernel[grid](*kernel_args)
