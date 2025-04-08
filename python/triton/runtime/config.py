from __future__ import annotations

import os

from dataclasses import field, dataclass
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import CacheManager, FileCacheManager

EnvVal = str | None


@dataclass
class _RedisConfig:
    key_format: str = os.getenv("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
    host: str = os.getenv("TRITON_REDIS_HOST", "localhost")
    port: int = int(os.getenv("TRITON_REDIS_PORT", 6379))


@dataclass
class _CacheConfig:
    remote_backend: EnvVal =
    manager_class_name: EnvVal = os.getenv("TRITON_CACHE_MANAGER")
    manager_class: Type[CacheManager] = FileCacheManager


@dataclass
class _TritonConfig:
    cc: str | None = os.getenv("CC")

