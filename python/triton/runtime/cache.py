import importlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


def default_cache_dir():
    return os.path.join(Path.home(), ".triton", "cache")


def default_override_dir():
    return os.path.join(Path.home(), ".triton", "override")


def default_dump_dir():
    return os.path.join(Path.home(), ".triton", "dump")


class CacheManager(ABC):

    def __init__(self, key):
        pass

    @abstractmethod
    def get_file(self, filename) -> Optional[str]:
        pass

    @abstractmethod
    def put(self, data, filename, binary=True) -> str:
        pass

    @abstractmethod
    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        pass

    @abstractmethod
    def put_group(self, filename: str, group: Dict[str, str]):
        pass


class FileCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()
            if self.cache_dir:
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")

    def _make_path(self, filename) -> str:
        return os.path.join(self.cache_dir, filename)

    def has_file(self, filename) -> bool:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        return os.path.exists(self._make_path(filename))

    def get_file(self, filename) -> Optional[str]:
        if self.has_file(filename):
            return self._make_path(filename)
        else:
            return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        grp_filename = f"__grp__{filename}"
        if not self.has_file(grp_filename):
            return None
        grp_filepath = self._make_path(grp_filename)
        with open(grp_filepath) as f:
            grp_data = json.load(f)
        child_paths = grp_data.get("child_paths", None)
        # Invalid group data.
        if child_paths is None:
            return None
        result = {}
        for c, p in child_paths.items():
            if os.path.exists(p):
                result[c] = p
        return result

    # Note a group of pushed files as being part of a group
    def put_group(self, filename: str, group: Dict[str, str]) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        grp_contents = json.dumps({"child_paths": group})
        grp_filename = f"__grp__{filename}"
        return self.put(grp_contents, grp_filename, binary=False)

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = str(uuid.uuid4())
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use tempfile to be robust against program interruptions
        temp_path = f"{filepath}.tmp.pid_{pid}_{rnd_id}"
        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        return filepath


class RemoteCacheBackend:
    """
    A backend implementation for accessing a remote/distributed cache.
    """

    def __init__(self, key: str):
        pass

    @abstractmethod
    def get(self, filenames: List[str]) -> Dict[str, bytes]:
        pass

    @abstractmethod
    def put(self, filename: str, data: bytes):
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend):

    def __init__(self, key):
        import redis
        self._key = key
        self._key_fmt = os.environ.get("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
        self._redis = redis.Redis(
            host=os.environ.get("TRITON_REDIS_HOST", "localhost"),
            port=int(os.environ.get("TRITON_REDIS_PORT", 6379)),
        )

    def _get_key(self, filename: str) -> str:
        return self._key_fmt.format(key=self._key, filename=filename)

    def get(self, filenames: List[str]) -> Dict[str, str]:
        results = self._redis.mget([self._get_key(f) for f in filenames])
        return {filename: result for filename, result in zip(filenames, results) if result is not None}

    def put(self, filename: str, data: bytes) -> Dict[str, bytes]:
        self._redis.set(self._get_key(filename), data)


class RemoteCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        # Setup backend pointed too by `TRITON_REMOTE_CACHE_BACKEND`.
        remote_cache_manager = os.environ["TRITON_REMOTE_CACHE_BACKEND"]
        module_path, clz_nme = remote_cache_manager.split(":")
        module = importlib.import_module(module_path)
        remote_cache_cls = getattr(module, clz_nme)
        self._backend = remote_cache_cls(key)

        self._override = override
        self._dump = dump

        # Use a `FileCacheManager` to materialize remote cache paths locally.
        self._file_cache_manager = FileCacheManager(key, override=override, dump=dump)

    def _materialize(self, filename: str, data: bytes):
        # We use a backing `FileCacheManager` to provide the materialized data.
        return self._file_cache_manager.put(data, filename, binary=True)

    def get_file(self, filename: str) -> Optional[str]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_file(filename)

        # We always check the remote cache backend -- even if our internal file-
        # based cache has the item -- to make sure LRU accounting works as
        # expected.
        results = self._backend.get([filename])
        if len(results) == 0:
            return None
        (_, data), = results.items()
        return self._materialize(filename, data)

    def put(self, data, filename: str, binary=True) -> str:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.put(data, filename, binary=binary)

        if not isinstance(data, bytes):
            data = str(data).encode("utf-8")
        self._backend.put(filename, data)
        return self._materialize(filename, data)

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_group(filename)

        grp_filename = f"__grp__{filename}"
        grp_filepath = self.get_file(grp_filename)
        if grp_filepath is None:
            return None
        with open(grp_filepath) as f:
            grp_data = json.load(f)
        child_paths = grp_data.get("child_paths", None)

        result = None

        # Found group data.
        if child_paths is not None:
            result = {}
            for child_path, data in self._backend.get(child_paths).items():
                result[child_path] = self._materialize(child_path, data)

        return result

    def put_group(self, filename: str, group: Dict[str, str]):
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.put_group(filename, group)

        grp_contents = json.dumps({"child_paths": sorted(list(group.keys()))})
        grp_filename = f"__grp__{filename}"
        return self.put(grp_contents, grp_filename)


__cache_cls = FileCacheManager
__cache_cls_nme = "DEFAULT"


def get_cache_manager(key) -> CacheManager:
    import os

    user_cache_manager = os.environ.get("TRITON_CACHE_MANAGER", None)
    global __cache_cls
    global __cache_cls_nme

    if user_cache_manager is not None and user_cache_manager != __cache_cls_nme:
        module_path, clz_nme = user_cache_manager.split(":")
        module = importlib.import_module(module_path)
        __cache_cls = getattr(module, clz_nme)
        __cache_cls_nme = user_cache_manager

    return __cache_cls(key)


def get_override_manager(key) -> CacheManager:
    return __cache_cls(key, override=True)


def get_dump_manager(key) -> CacheManager:
    return __cache_cls(key, dump=True)


def make_so_cache_key(version_hash, signature, constants, ids, **kwargs):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}-{constants}-{ids}"
    for kw in kwargs:
        key = f"{key}-{kwargs.get(kw)}"
    key = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return key
