# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""tiletranslation backend: cuTile kernels translated to Triton."""

from tilegym.backend import is_backend_available

# Only import if tiletranslation backend is available
if is_backend_available("tiletranslation"):
    from . import swiglu

    # Import specific functions for direct access
    from .swiglu import get_swiglu

    __all__ = [
        "swiglu",
        "get_swiglu",
    ]
else:
    __all__ = []
