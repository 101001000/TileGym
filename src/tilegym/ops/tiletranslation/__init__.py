# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""tiletranslation backend: cuTile kernels translated to Triton."""

from tilegym.backend import is_backend_available

# Only import if tiletranslation backend is available
if is_backend_available("tiletranslation"):
    from . import swiglu
    from . import matmul
    from . import silu_and_mul
    from . import rope
    from . import rms_norm

    # Import specific functions for direct access
    from .swiglu import get_swiglu
    from .rope import apply_rope_base
    from .rope import get_apply_rope_func
    from .rms_norm import get_rms_norm_module
    from .rms_norm import rms_norm

    __all__ = [
        "swiglu",
        "get_swiglu",
        "matmul",
        "silu_and_mul",
        "apply_rope_base",
        "get_apply_rope_func",
        "rms_norm",
        "get_rms_norm_module",
    ]
else:
    __all__ = []
