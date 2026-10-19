# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Shared on-disk artifact cache for TileGym backends.

Layout: ``<cache root>/<backend>/<op>/<artifacts>``

- The cache root is ``$TILEGYM_CACHE_DIR`` if set, otherwise
  ``$XDG_CACHE_HOME/tilegym`` or ``~/.cache/tilegym`` (XDG convention, matching
  the tilecpp backend's cache).
- Each backend gets its own folder ("their own cache folder"); artifacts are
  sorted by op below it, and new ops cache the same way by passing their op
  name.
"""

import os
from pathlib import Path


def get_cache_dir(*subdirs: str) -> Path:
    """Return (and create) a cache directory under the TileGym cache root.

    Args:
        *subdirs: Path segments below the root, e.g. ``("tiletranslation",
            "swiglu")`` for the swiglu op of the tiletranslation backend.

    Returns:
        The created ``Path`` (``mkdir(parents=True, exist_ok=True)``).
    """
    env_dir = os.environ.get("TILEGYM_CACHE_DIR", "")
    if env_dir:
        cache_dir = Path(env_dir)
    else:
        # Use XDG cache dir or fallback to ~/.cache
        xdg_cache = os.environ.get("XDG_CACHE_HOME", "")
        if xdg_cache:
            cache_dir = Path(xdg_cache) / "tilegym"
        else:
            cache_dir = Path.home() / ".cache" / "tilegym"
    cache_dir = cache_dir.joinpath(*subdirs) if subdirs else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
