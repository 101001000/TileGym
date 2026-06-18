# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable

import numpy as np
import torch

SPTILES_TRANSLATE_BIN_ENV = "SPTILES_TRANSLATE_BIN"
SPTILES_CACHE_DIR_ENV = "TILEGYM_SPTILE_CACHE_DIR"

TilebcCompiler = Callable[..., Path]
_TILEBC_COMPILERS: dict[str, TilebcCompiler] = {}
_PINNED_CPU_CACHE: dict[tuple[str, tuple[int, ...], torch.dtype, str], torch.Tensor] = {}
_CUDA_OUTPUT_CACHE: dict[tuple[str, tuple[int, ...], torch.dtype, str], torch.Tensor] = {}


def register_tilebc_compiler(kernel_name: str) -> Callable[[TilebcCompiler], TilebcCompiler]:
    def decorator(compiler: TilebcCompiler) -> TilebcCompiler:
        _TILEBC_COMPILERS[kernel_name] = compiler
        return compiler

    return decorator


def _load_kernel_module(kernel_name: str) -> None:
    try:
        importlib.import_module(f"{__package__}.{kernel_name}")
    except ImportError as exc:
        raise NotImplementedError(f"sptile does not know how to compile {kernel_name!r}") from exc


def get_tilebc_compiler(kernel_name: str) -> TilebcCompiler:
    compiler = _TILEBC_COMPILERS.get(kernel_name)
    if compiler is not None:
        return compiler

    _load_kernel_module(kernel_name)
    compiler = _TILEBC_COMPILERS.get(kernel_name)
    if compiler is None:
        raise NotImplementedError(f"sptile module {kernel_name!r} did not register a tilebc compiler")
    return compiler


def find_sp_tiles_translate() -> Path:
    translator = os.environ.get(SPTILES_TRANSLATE_BIN_ENV)
    if translator:
        candidate = Path(translator).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"{SPTILES_TRANSLATE_BIN_ENV} points to a missing file: {candidate}")

    resolved = shutil.which("sp_tiles_translate")
    if resolved is not None:
        return Path(resolved)

    raise FileNotFoundError(
        "Could not locate the spTiles translator binary. Set "
        f"{SPTILES_TRANSLATE_BIN_ENV} to the `sp_tiles_translate` executable."
    )


def _sptile_cache_dir() -> Path:
    cache_dir = os.environ.get(SPTILES_CACHE_DIR_ENV)
    if cache_dir:
        return Path(cache_dir).expanduser()
    return Path.home() / ".cache" / "tilegym" / "sptile"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compiler_digest(compiler: TilebcCompiler) -> str:
    source_file = inspect.getsourcefile(compiler) or inspect.getfile(compiler)
    source_path = Path(source_file)
    if source_path.is_file():
        return _file_digest(source_path)
    return hashlib.sha256(inspect.getsource(compiler).encode()).hexdigest()


def compile_tilebc(
    kernel_name: str,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    compiler = get_tilebc_compiler(kernel_name)
    compiler_key = f"{compiler.__module__}.{compiler.__qualname__}"
    cache_key = hashlib.sha256(f"{kernel_name}:{compiler_key}:{_compiler_digest(compiler)}".encode()).hexdigest()[:16]
    base_cache = Path(cache_dir).expanduser() if cache_dir is not None else _sptile_cache_dir()
    output_dir = base_cache / "tilebc" / kernel_name / cache_key
    output_path = output_dir / f"{kernel_name}.tilebc"

    if output_path.is_file() and not force:
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    compiler(output_path)
    if not output_path.is_file():
        raise RuntimeError(f"sptile compilation finished without producing {output_path}")
    return output_path


def translate_tilebc_to_spirv(
    tilebc_path: str | Path,
    *,
    translator_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    tilebc = Path(tilebc_path).expanduser().resolve()
    if not tilebc.is_file():
        raise FileNotFoundError(f"cuTile bytecode file not found: {tilebc}")

    translator = (
        Path(translator_path).expanduser().resolve()
        if translator_path is not None
        else find_sp_tiles_translate().resolve()
    )
    if not translator.is_file():
        raise FileNotFoundError(f"spTiles translator binary not found: {translator}")

    base_cache = Path(cache_dir).expanduser() if cache_dir is not None else _sptile_cache_dir()
    cache_key = hashlib.sha256(
        f"{tilebc}:{_file_digest(tilebc)}:{translator}:{translator.stat().st_mtime_ns}".encode()
    ).hexdigest()[:16]
    output_dir = base_cache / "spirv" / tilebc.stem / cache_key
    output_path = output_dir / "output.spv"

    if output_path.is_file() and not force:
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(translator), str(tilebc)],
        cwd=output_dir,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "spTiles translation failed\n"
            f"command: {translator} {tilebc}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not output_path.is_file():
        raise RuntimeError(
            "spTiles translation finished without producing output.spv\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return output_path


def resolve_spirv_path(
    *,
    kernel_name: str,
    spirv_path: str | Path | None = None,
    tilebc_path: str | Path | None = None,
    translator_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force_retranslate: bool = False,
) -> Path:
    if spirv_path is not None:
        return Path(spirv_path).expanduser()

    tilebc = (
        Path(tilebc_path).expanduser()
        if tilebc_path is not None
        else compile_tilebc(kernel_name, cache_dir=cache_dir, force=force_retranslate)
    )
    return translate_tilebc_to_spirv(
        tilebc,
        translator_path=translator_path,
        cache_dir=cache_dir,
        force=force_retranslate,
    )


def pop_runtime_options(
    kwargs: dict[str, Any],
    *,
    default_entry_name: str,
) -> tuple[dict[str, Any], str]:
    runtime_options = {
        "spirv_path": kwargs.pop("spirv_path", None),
        "tilebc_path": kwargs.pop("tilebc_path", None),
        "translator_path": kwargs.pop("translator_path", None),
        "cache_dir": kwargs.pop("sptile_cache_dir", None),
        "force_retranslate": kwargs.pop("force_retranslate", False),
    }
    entry_name = kwargs.pop("entry_name", default_entry_name)
    return runtime_options, entry_name


def is_cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def cached_pinned_cpu_tensor(
    tensor: torch.Tensor,
    role: str,
    *,
    op_name: str = "sptile operation",
) -> torch.Tensor:
    key = (role, tuple(tensor.shape), tensor.dtype, str(tensor.device))
    cached = _PINNED_CPU_CACHE.get(key)
    if cached is not None:
        return cached

    if is_cuda_graph_capture_active():
        raise RuntimeError(
            f"{op_name} needs pinned CPU staging buffers before CUDA graph capture. "
            "Call the operation once with the same shapes before capture starts."
        )

    cached = torch.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device="cpu",
        pin_memory=torch.cuda.is_available(),
    )
    _PINNED_CPU_CACHE[key] = cached
    return cached


def as_pinned_cpu_tensor(
    tensor: torch.Tensor,
    role: str,
    *,
    op_name: str = "sptile operation",
) -> torch.Tensor:
    if tensor.device.type == "cpu":
        cpu_tensor = tensor.detach().contiguous()
        if torch.cuda.is_available() and not cpu_tensor.is_pinned():
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor

    cpu_tensor = cached_pinned_cpu_tensor(tensor, role, op_name=op_name)
    cpu_tensor.copy_(tensor.detach().contiguous(), non_blocking=True)
    if not is_cuda_graph_capture_active():
        torch.cuda.current_stream(tensor.device).synchronize()
    return cpu_tensor


def return_to_original_device(
    result: torch.Tensor,
    device: torch.device,
    *,
    role: str,
    op_name: str = "sptile operation",
) -> torch.Tensor:
    if device.type == "cpu":
        return result

    pinned_result = cached_pinned_cpu_tensor(result, f"{role}:host:{device}", op_name=op_name)
    pinned_result.copy_(result.contiguous())

    key = (role, tuple(result.shape), result.dtype, str(device))
    output = _CUDA_OUTPUT_CACHE.get(key)
    if output is None:
        if is_cuda_graph_capture_active():
            raise RuntimeError(
                f"{op_name} needs a CUDA output tensor before CUDA graph capture. "
                "Call the operation once with the same output shape before capture starts."
            )
        output = torch.empty(result.shape, dtype=result.dtype, device=device)
        _CUDA_OUTPUT_CACHE[key] = output

    output.copy_(pinned_result, non_blocking=True)
    return output


def _import_kompute() -> Any:
    try:
        import kp
    except ImportError as exc:
        raise RuntimeError("sptile requires the Kompute Python package (`kp`).") from exc
    return kp


def check_kompute_api() -> None:
    kp = _import_kompute()
    missing = [name for name in ("Manager", "OpAlgoDispatch") if not hasattr(kp, name)]
    if missing:
        raise RuntimeError(
            "This Kompute Python package is too old for the cuTile SPIR-V ABI. "
            f"Missing: {', '.join(missing)}. Install a recent kp package from "
            "the Kompute repository."
        )


def read_spirv(path: str | Path, entry_name: str = "main") -> bytes:
    data = Path(path).read_bytes()
    if not data or len(data) % 4 != 0:
        raise ValueError(f"Invalid SPIR-V file size for {path}")
    words = list(struct.unpack(f"<{len(data) // 4}I", data))
    if words[0] != 0x07230203:
        raise ValueError(f"Invalid SPIR-V magic number for {path}")
    patched = rename_entry_point_for_kompute(words, entry_name)
    return struct.pack(f"<{len(patched)}I", *patched)


def rename_entry_point_for_kompute(words: list[int], entry_name: str) -> list[int]:
    if entry_name == "main":
        return words

    op_entry_point = 15
    index = 5
    found_requested_entry = False
    found_main_entry = False
    patched = words.copy()

    while index < len(patched):
        instruction = patched[index]
        word_count = instruction >> 16
        opcode = instruction & 0xFFFF
        if word_count == 0:
            raise ValueError("Invalid SPIR-V instruction with zero word count.")

        if opcode == op_entry_point and word_count >= 4:
            name_word_start = index + 3
            max_name_words = word_count - 3
            name_bytes = b"".join(
                word.to_bytes(4, "little")
                for word in patched[name_word_start : name_word_start + max_name_words]
            )
            nul = name_bytes.find(b"\0")
            if nul == -1:
                raise ValueError("Invalid SPIR-V OpEntryPoint without a null-terminated name.")

            current_name = name_bytes[:nul].decode("utf-8")
            name_word_count = (nul + 4) // 4
            if current_name == "main":
                found_main_entry = True
            elif current_name == entry_name:
                found_requested_entry = True
                replacement = b"main\0".ljust(name_word_count * 4, b"\0")
                patched[name_word_start : name_word_start + name_word_count] = struct.unpack(
                    f"<{name_word_count}I",
                    replacement,
                )

        index += word_count

    if found_main_entry or found_requested_entry:
        return patched
    raise ValueError(f"Entry point {entry_name!r} was not found in the SPIR-V module.")


def _make_tensor(manager: Any, data: np.ndarray) -> Any:
    if not hasattr(manager, "tensor_t"):
        raise RuntimeError(
            "Installed Kompute does not expose Manager.tensor_t. A newer Kompute "
            "Python build is required because this runner binds float32 and int32 buffers."
        )
    return manager.tensor_t(np.ascontiguousarray(data))


def tensor_2d_descriptor(
    tensor: np.ndarray,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
    index_dtype: np.dtype[Any] | type[np.generic] = np.int32,
) -> list[np.ndarray]:
    data = np.ascontiguousarray(tensor, dtype=dtype)
    dim0, dim1 = data.shape
    stride0 = data.strides[0] // data.itemsize
    stride1 = data.strides[1] // data.itemsize
    return [
        data.reshape(-1),
        np.array([dim0], dtype=index_dtype),
        np.array([dim1], dtype=index_dtype),
        np.array([stride0], dtype=index_dtype),
        np.array([stride1], dtype=index_dtype),
    ]


def _sync_device_op(kp: Any, params: list[Any]) -> Any:
    op_type = getattr(kp, "OpTensorSyncDevice", None) or getattr(kp, "OpSyncDevice", None)
    if op_type is None:
        raise RuntimeError("Installed Kompute does not expose a device sync operation.")
    return op_type(params)


def _sync_local_op(kp: Any, params: list[Any]) -> Any:
    op_type = getattr(kp, "OpTensorSyncLocal", None) or getattr(kp, "OpSyncLocal", None)
    if op_type is None:
        raise RuntimeError("Installed Kompute does not expose a local sync operation.")
    return op_type(params)


def _read_output_tensor(output_tensor: Any) -> np.ndarray:
    if hasattr(output_tensor, "vector"):
        return np.array(output_tensor.vector(), dtype=np.float32, copy=True)
    if hasattr(output_tensor, "data"):
        return np.array(output_tensor.data(), dtype=np.float32, copy=True)
    raise RuntimeError("Unable to read data back from the Kompute output tensor.")


def run_spirv_kernel(
    *,
    spirv_path: str | Path,
    entry_name: str,
    parameter_arrays: Iterable[np.ndarray],
    output_indices: Iterable[int],
    dispatch: tuple[int, int, int],
) -> list[np.ndarray]:
    prepared = PreparedSpirvKernel(
        spirv_path=spirv_path,
        entry_name=entry_name,
        parameter_arrays=parameter_arrays,
        output_indices=output_indices,
        dispatch=dispatch,
    )
    return prepared.dispatch_and_read()


class PreparedSpirvKernel:
    def __init__(
        self,
        *,
        spirv_path: str | Path,
        entry_name: str,
        parameter_arrays: Iterable[np.ndarray],
        output_indices: Iterable[int],
        dispatch: tuple[int, int, int],
    ) -> None:
        check_kompute_api()
        self.kp = _import_kompute()
        self.manager = self.kp.Manager()
        self.params = [_make_tensor(self.manager, array) for array in parameter_arrays]
        self.output_tensors = [self.params[index] for index in output_indices]
        self.algorithm = self.manager.algorithm(self.params, read_spirv(spirv_path, entry_name), dispatch)

        self.manager.sequence().record(_sync_device_op(self.kp, self.params)).eval()

    def dispatch(self) -> None:
        self.manager.sequence().record(self.kp.OpAlgoDispatch(self.algorithm)).eval()

    def dispatch_and_read(self) -> list[np.ndarray]:
        sequence = self.manager.sequence()
        (
            sequence.record(self.kp.OpAlgoDispatch(self.algorithm))
            .record(_sync_local_op(self.kp, self.output_tensors))
            .eval()
        )
        return [_read_output_tensor(output_tensor) for output_tensor in self.output_tensors]
