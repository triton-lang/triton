# Copyright © 2024 BAAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications:
# - 2025-06-03:
#   - init version: e9c7aa71832eb2f897a49ce787e42d5377404a72
#

import functools
import os
import shutil
from pathlib import Path


@functools.lru_cache(maxsize=None)  # this is the same as functools.cache in Python 3.9+
def cache_dir_path() -> Path:
    """Return the cache directory for generated files in flaggems."""
    _cache_dir = os.environ.get("FLAGGEMS_CACHE_DIR")
    if _cache_dir is None:
        _cache_dir = Path.home() / ".flaggems"
    else:
        _cache_dir = Path(_cache_dir)
    return _cache_dir


def cache_dir() -> Path:
    """Return cache directory for generated files in flaggems. Create it if it does not exist."""
    _cache_dir = cache_dir_path()
    os.makedirs(_cache_dir, exist_ok=True)
    return _cache_dir


def code_cache_dir() -> Path:
    _code_cache_dir = cache_dir() / "code_cache"
    os.makedirs(_code_cache_dir, exist_ok=True)
    return _code_cache_dir


def config_cache_dir() -> Path:
    _config_cache_dir = cache_dir() / "config_cache"
    os.makedirs(_config_cache_dir, exist_ok=True)
    return _config_cache_dir


def clear_cache():
    """Clear the cache directory for code cache."""
    _cache_dir = cache_dir_path()
    shutil.rmtree(_cache_dir)
