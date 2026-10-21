# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""droptokens backend: cuTile kernels executed after the cuda-tile-drop-tokens pass."""

from tilegym.backend import is_backend_available

# Only import if droptokens backend is available
if is_backend_available("droptokens"):
    from . import rope
    from . import swiglu

    # Import specific functions for direct access
    from .rope import apply_rope_base, get_apply_rope_func
    from .swiglu import get_swiglu

    __all__ = [
        "rope",
        "swiglu",
        "apply_rope_base",
        "get_apply_rope_func",
        "get_swiglu",
    ]
else:
    __all__ = []
