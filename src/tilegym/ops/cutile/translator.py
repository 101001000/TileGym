import subprocess
import triton

from cuda.tile.compilation import CallingConvention, KernelSignature, export_kernel

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

def prepare_triton_kernel(kernel, kernel_args, gpu_code, output_prefix):
    bytecode_path = f"{output_prefix}.tilebc"
    mlir_path = f"{output_prefix}.tileir.mlir"
    ttir_path = f"{output_prefix}.ttir"

    emit_bytecode(kernel, kernel_args, bytecode_path, gpu_code)
    emit_mlir(bytecode_path, mlir_path)
    translate_tileir(mlir_path, ttir_path)

    return triton.compile(ttir_path)


def invoke_triton_kernel(kernel, grid, *kernel_args):
    kernel[grid](*kernel_args)