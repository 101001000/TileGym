# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile IR drop-tokens pipeline for the droptokens backend.

Takes the cuTile IR bytecode that cuda.tile would normally hand to tileiras,
round-trips it through the translation toolchain to apply the
``cuda-tile-drop-tokens`` pass (remove token usage from CUDA Tile IR), and
compiles the resulting cuTile IR to a cubin with the same tileiras path the
cutile backend uses:

    bytecode --(cuda-tile-translate --cudatilebc-to-mlir)--> MLIR
             --(triton-cuda-tile-opt --cuda-tile-drop-tokens)--> MLIR
             --(cuda-tile-translate --mlir-to-cudatilebc)--> bytecode
             --(tileiras via cuda.tile._compile.compile_cubin)--> cubin
"""

import subprocess
import tempfile
from pathlib import Path

from cuda.tile._compile import compile_cubin


def cudatilebc_to_mlir(bytecode_path, output_path):
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


def run_drop_tokens_pass(mlir_path, output_path):
    subprocess.run(
        [
            "triton-cuda-tile-opt",
            str(mlir_path),
            "--pass-pipeline=builtin.module(cuda-tile-drop-tokens)",
            "-o",
            str(output_path),
        ],
        check=True,
    )


def mlir_to_cudatilebc(mlir_path, output_path):
    subprocess.run(
        [
            "cuda-tile-translate",
            "--mlir-to-cudatilebc",
            str(mlir_path),
            "-o",
            str(output_path),
        ],
        check=True,
    )


def drop_tokens_cubin(bytecode: bytes, sm_arch: str, compiler_options, timeout_sec=None) -> bytes:
    """Apply cuda-tile-drop-tokens to cuTile IR bytecode and compile to cubin.

    Args:
        bytecode: cuTile IR bytecode (as produced by ``compile_tile`` /
            ``export_kernel`` with ``output_format="tileir_bytecode"``).
        sm_arch: Target GPU architecture (e.g. ``"sm_120"``).
        compiler_options: cuda.tile ``CompilerOptions`` forwarded to tileiras.
        timeout_sec: Optional compiler timeout forwarded to tileiras.

    Returns:
        The cubin bytes of the token-dropped cuTile IR.
    """
    with tempfile.TemporaryDirectory(prefix="tilegym_droptokens_") as tmp:
        in_bc = Path(tmp) / "kernel.tilebc"
        mlir_in = Path(tmp) / "kernel.tileir.mlir"
        mlir_out = Path(tmp) / "kernel.dropped.tileir.mlir"
        out_bc = Path(tmp) / "kernel.dropped.tilebc"

        in_bc.write_bytes(bytecode)
        cudatilebc_to_mlir(in_bc, mlir_in)
        run_drop_tokens_pass(mlir_in, mlir_out)
        mlir_to_cudatilebc(mlir_out, out_bc)

        cubin_path = compile_cubin(str(out_bc), compiler_options, sm_arch, timeout_sec)
        return Path(cubin_path).read_bytes()
