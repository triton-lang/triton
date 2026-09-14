"""SQLite persistence with process-lifetime files for the path-based cache API."""

import atexit
from contextlib import contextmanager
import functools
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from pathlib import Path

from triton import knobs
from .cache import CacheManager, FileCacheManager


@functools.lru_cache(None)
def _runtime_dir(pid):
    directory = tempfile.mkdtemp(prefix="triton-sqlite-")

    def cleanup():
        # An ordinary fork must not remove files still used by its parent.
        if os.getpid() == pid:
            shutil.rmtree(directory, ignore_errors=True)

    atexit.register(cleanup)
    return directory


class SQLiteCacheManager(CacheManager):
    """Local-only cache. Returned paths remain valid until process exit.

    No connection is retained across operations, threads, or forks. SQLite's
    rollback journal and busy timeout serialize concurrent local writers.
    """

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self._files = FileCacheManager(key, override=override, dump=dump) if override or dump else None
        self.database = os.path.abspath(os.path.join(knobs.cache.dir, "triton-cache-v1.sqlite3"))

    @contextmanager
    def _connect(self, *, initialize=True):
        connection = None
        try:
            if not initialize:
                # Open only an existing database, while allowing hot-journal recovery.
                connection = sqlite3.connect(Path(self.database).as_uri() + "?mode=rw", uri=True, timeout=30)
            else:
                os.makedirs(os.path.dirname(self.database), exist_ok=True)
                connection = sqlite3.connect(self.database, timeout=30)
            with connection:
                if initialize:
                    connection.execute("CREATE TABLE IF NOT EXISTS entries ("
                                       "cache_key TEXT NOT NULL, filename TEXT NOT NULL, data BLOB NOT NULL, "
                                       "PRIMARY KEY (cache_key, filename)) WITHOUT ROWID")
                yield connection
        except sqlite3.Error as error:
            raise RuntimeError(
                f"SQLite cache failure at {self.database}: {error}. "
                "Use writable local storage for compilation; stop all users before clearing a corrupt cache."
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def _materialize(self, filename, data):
        if not filename or os.path.basename(filename) != filename or filename in (".", ".."):
            raise ValueError(f"Invalid cache filename: {filename!r}")
        # Content-specific paths keep previously returned files valid even if
        # another writer replaces the same logical cache entry.
        digest = hashlib.sha256(data).hexdigest()
        directory = os.path.join(_runtime_dir(os.getpid()), digest)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            temporary = os.path.join(directory, str(uuid.uuid4()))
            try:
                with open(temporary, "wb") as stream:
                    stream.write(data)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return path

    def get_file(self, filename):
        if self._files is not None:
            return self._files.get_file(filename)
        if not os.path.exists(self.database):
            return None
        with self._connect(initialize=False) as connection:
            if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entries'").fetchone() is None:
                return None
            row = connection.execute("SELECT data FROM entries WHERE cache_key = ? AND filename = ?",
                                     (self.key, filename)).fetchone()
        return None if row is None else self._materialize(filename, row[0])

    def put(self, data, filename, binary=True):
        if self._files is not None:
            return self._files.put(data, filename, binary=binary)
        if not isinstance(data, bytes):
            data = str(data).encode("utf-8")
        path = self._materialize(filename, data)
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO entries VALUES (?, ?, ?)", (self.key, filename, data))
        return path

    def put_group(self, filename, group):
        if self._files is not None:
            return self._files.put_group(filename, group)
        # Persist names, never process-local paths. Publish only complete groups.
        data = json.dumps({"children": sorted(group)}).encode("utf-8")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for child in group:
                if connection.execute("SELECT 1 FROM entries WHERE cache_key = ? AND filename = ?",
                                      (self.key, child)).fetchone() is None:
                    raise ValueError(f"Missing cache group member: {child}")
            connection.execute("INSERT OR REPLACE INTO entries VALUES (?, ?, ?)",
                               (self.key, "__grp__" + filename, data))
        return self._materialize("__grp__" + filename, data)

    def get_group(self, filename):
        if self._files is not None:
            return self._files.get_group(filename)
        if not os.path.exists(self.database):
            return None
        with self._connect(initialize=False) as connection:
            connection.execute("BEGIN")
            if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entries'").fetchone() is None:
                return None
            row = connection.execute("SELECT data FROM entries WHERE cache_key = ? AND filename = ?",
                                     (self.key, "__grp__" + filename)).fetchone()
            if row is None:
                return None
            try:
                children = json.loads(row[0])["children"]
                if not isinstance(children, list) or any(not isinstance(child, str) for child in children):
                    return None
            except (ValueError, KeyError, TypeError):
                return None
            result = {}
            for child in children:
                row = connection.execute("SELECT data FROM entries WHERE cache_key = ? AND filename = ?",
                                         (self.key, child)).fetchone()
                if row is None:
                    return None
                result[child] = row[0]
        return {child: self._materialize(child, data) for child, data in result.items()}
