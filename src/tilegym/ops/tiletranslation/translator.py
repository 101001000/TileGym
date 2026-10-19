import hashlib
import inspect
import os
import subprocess
from pathlib import Path

import triton

from cuda.tile.compilation import CallingConvention, KernelSignature, export_kernel

from tilegym.ops.cache_utils import get_cache_dir

DEFAULT_SM_ARCH = "sm_120"


def get_sm_arch() -> str:
    """Target GPU architecture for the translation pipeline.

    Read from the ``TILEGYM_SM_ARCH`` environment variable (e.g. ``sm_120``),
    defaulting to ``sm_120``. Kept out of the per-op prepare paths so the
    target is a pipeline-level decision rather than derived from the current
    device at each call site.
    """
    return os.environ.get("TILEGYM_SM_ARCH", DEFAULT_SM_ARCH)

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
    signature = KernelSignature.from_kernel_args(
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
