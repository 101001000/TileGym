# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""TileGym ops module - contains all operation interfaces and backend implementations"""

from tilegym.backend import is_backend_available

# Backend implementations
# Import interface modules
from . import activation
from . import attn_interface

# Make cutile optional - only import if backend is fully available
if is_backend_available("cutile"):
    try:
        from . import cutile
    except (ImportError, RuntimeError):
        import warnings

        warnings.warn("Cutile backend import failed, cutile operations will not be available")
        cutile = None  # type: ignore
else:
    cutile = None  # type: ignore

from . import moe_interface

# Make triton optional - only import if the backend is enabled
# (disabled by default; set TILEGYM_ENABLE_TRITON=1 to opt in)
if is_backend_available("triton"):
    try:
        from . import triton
    except (ImportError, RuntimeError):
        import warnings

        warnings.warn("Triton backend import failed, triton operations will not be available")
        triton = None  # type: ignore
else:
    triton = None  # type: ignore

# Import CUDA Tile C++ backend if available
if is_backend_available("tilecpp"):
    from . import tilecpp

# Import cutile-rs (Rust FFI) backend if available
if is_backend_available("cutile-rs"):
    try:
        from . import cutile_rs
    except (ImportError, RuntimeError):
        import warnings

        warnings.warn(
            "cutile-rs backend import failed, cutile-rs operations will not be available",
            stacklevel=2,
        )
        cutile_rs = None  # type: ignore
        # Backend isn't usable if its ops failed to import — keep the selector in sync.
        from tilegym.backend.selector import _AVAILABLE_BACKENDS

        _AVAILABLE_BACKENDS.discard("cutile-rs")
else:
    cutile_rs = None  # type: ignore

# Import tiletranslation (cuTile -> Triton) backend if available
if is_backend_available("tiletranslation"):
    try:
        from . import tiletranslation
    except (ImportError, RuntimeError):
        import warnings

        warnings.warn(
            "tiletranslation backend import failed, tiletranslation operations will not be available",
            stacklevel=2,
        )
        tiletranslation = None  # type: ignore
        # Backend isn't usable if its ops failed to import — keep the selector in sync.
        from tilegym.backend.selector import _AVAILABLE_BACKENDS

        _AVAILABLE_BACKENDS.discard("tiletranslation")
else:
    tiletranslation = None  # type: ignore

# Import droptokens (cuTile IR + cuda-tile-drop-tokens pass) backend if available
if is_backend_available("droptokens"):
    try:
        from . import droptokens
    except (ImportError, RuntimeError):
        import warnings

        warnings.warn(
            "droptokens backend import failed, droptokens operations will not be available",
            stacklevel=2,
        )
        droptokens = None  # type: ignore
        # Backend isn't usable if its ops failed to import — keep the selector in sync.
        from tilegym.backend.selector import _AVAILABLE_BACKENDS

        _AVAILABLE_BACKENDS.discard("droptokens")
else:
    droptokens = None  # type: ignore

# Re-export key interfaces
from .attn_interface import attention_sink_interface
from .attn_interface import fmha_interface
from .attn_interface import get_attention_sink_interface
from .attn_interface import get_fmha_gemma3_interface
from .attn_interface import get_fmha_interface
from .attn_interface import mla_decoding_interface
from .attn_interface import mla_interface
from .moe_interface import fused_moe
from .ops import *

__all__ = [
    # Export all operations from ops module
    # Interface modules
    "attn_interface",
    "moe_interface",
    # Re-exported submodules
    # Key interfaces
    "attention_sink_interface",
    "fmha_interface",
    "get_attention_sink_interface",
    "get_fmha_interface",
    "get_fmha_gemma3_interface",
    "mla_interface",
    "mla_decoding_interface",
    "fused_moe",
]

# Add cutile to exports only if successfully imported
if cutile is not None:
    __all__.append("cutile")

# Add tilecpp to exports only if it was imported above
if is_backend_available("tilecpp"):
    __all__.append("tilecpp")

# Add tiletranslation to exports only if it was imported above
if tiletranslation is not None:
    __all__.append("tiletranslation")

# Add droptokens to exports only if it was imported above
if droptokens is not None:
    __all__.append("droptokens")
