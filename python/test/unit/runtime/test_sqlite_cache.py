import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from triton import knobs
from triton.runtime.cache import FileCacheManager, get_cache_manager, get_dump_manager, get_override_manager
from triton.runtime.sqlite_cache import SQLiteCacheManager


@pytest.fixture
def cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))
    monkeypatch.setattr(knobs.cache, "manager_class", None)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TRITON_CACHE_MANAGER", raising=False)
    return tmp_path


@pytest.mark.parametrize("manager_class", [FileCacheManager, SQLiteCacheManager])
def test_roundtrip(cache_root, manager_class):
    manager = manager_class("key")
    assert manager.get_file("absent") is None
    for name, data in [("text.txt", "hello λ\n"), ("binary.bin", b"\x00\xff\x01")]:
        path = manager.put(data, name)
        expected = data.encode() if isinstance(data, str) else data
        assert Path(path).read_bytes() == expected
        assert Path(manager_class("key").get_file(name)).read_bytes() == expected
        assert manager_class("other").get_file(name) is None


def test_sqlite_missing_lookup_creates_nothing(cache_root, monkeypatch):
    root = cache_root / "absent"
    monkeypatch.setattr(knobs.cache, "dir", str(root))
    manager = SQLiteCacheManager("key")
    assert manager.get_file("missing") is None
    assert manager.get_group("missing") is None
    assert not root.exists()


def test_sqlite_database_name_isolated_from_inductor(cache_root):
    other = cache_root / "inductor-cache-v1.sqlite3"
    with sqlite3.connect(other) as connection:
        connection.execute("CREATE TABLE entries (namespace TEXT, key TEXT, data BLOB)")
    before = other.read_bytes()
    manager = SQLiteCacheManager("key")
    assert Path(manager.put(b"value", "x.bin")).read_bytes() == b"value"
    assert other.read_bytes() == before


def test_sqlite_lookup_during_schema_initialization(cache_root):
    manager = SQLiteCacheManager("key")
    connection = sqlite3.connect(manager.database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE entries (cache_key TEXT, filename TEXT, data BLOB)")
        assert manager.get_file("x.bin") is None
        assert manager.get_group("group") is None
        connection.rollback()
    finally:
        connection.close()
    assert Path(manager.put(b"value", "x.bin")).read_bytes() == b"value"


def test_sqlite_readonly_lookup_and_connection_error(cache_root, monkeypatch):
    manager = SQLiteCacheManager("key")
    manager.put_group("group", {"x.bin": manager.put(b"value", "x.bin")})
    database = Path(manager.database)
    before = database.read_bytes()
    original_connect = sqlite3.connect

    def connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=rw")
        connection = original_connect(database_uri, **kwargs)
        connection.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in (
            sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_TRANSACTION) else sqlite3.SQLITE_DENY)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        assert Path(manager.get_file("x.bin")).read_bytes() == b"value"
        assert Path(manager.get_group("group")["x.bin"]).read_bytes() == b"value"
    assert database.read_bytes() == before
    database.chmod(0o444)
    try:
        assert manager.get_group("group") is not None
    finally:
        database.chmod(0o600)
    assert list(cache_root.iterdir()) == [database]

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("cannot open")

    monkeypatch.setattr(sqlite3, "connect", fail)
    with pytest.raises(RuntimeError, match="SQLite cache failure"):
        manager.get_file("x.bin")


def test_selection_and_overrides(cache_root, monkeypatch):
    monkeypatch.setattr(knobs.cache, "backend", "file")
    assert isinstance(get_cache_manager("ab"), FileCacheManager)
    assert not (cache_root / "triton-cache-v1.sqlite3").exists()
    monkeypatch.setattr(knobs.cache, "backend", "sqlite")
    assert isinstance(get_cache_manager("ab"), SQLiteCacheManager)
    monkeypatch.setattr(knobs.cache, "dump_dir", str(cache_root / "dump"))
    monkeypatch.setattr(knobs.cache, "override_dir", str(cache_root / "override"))
    assert isinstance(get_dump_manager("ab"), FileCacheManager)
    assert isinstance(get_override_manager("ab"), FileCacheManager)
    monkeypatch.setattr(knobs.cache, "backend", "unknown")
    with pytest.raises(ValueError, match="TRITON_CACHE_BACKEND"):
        get_cache_manager("ab")
    monkeypatch.setattr(knobs.cache, "manager_class", FileCacheManager)
    assert isinstance(get_cache_manager("ab"), FileCacheManager)


def test_group_snapshot_and_missing_member(cache_root):
    manager = SQLiteCacheManager("key")
    paths = {name: manager.put(data, name) for name, data in [("x.json", "{}"), ("x.bin", b"binary")]}
    manager.put_group("x.json", paths)
    assert manager.get_group("x.json") == paths
    with sqlite3.connect(manager.database) as connection:
        data = connection.execute("SELECT data FROM entries WHERE filename = '__grp__x.json'").fetchone()[0]
        assert json.loads(data) == {"children": ["x.bin", "x.json"]}
        connection.execute("DELETE FROM entries WHERE filename = 'x.bin'")
    assert manager.get_group("x.json") is None
    with pytest.raises(ValueError, match="Missing cache group member"):
        manager.put_group("incomplete", {"missing": "unused"})
    assert manager.get_group("incomplete") is None


@pytest.mark.parametrize("data", [b"not json", b"null", b'{"children": 1}', b'{"children": [null]}'])
def test_corrupt_group_is_miss(cache_root, data):
    manager = SQLiteCacheManager("key")
    manager.put(data, "__grp__x.json")
    assert manager.get_group("x.json") is None


def test_corrupt_database_reports_failure(cache_root):
    (cache_root / "triton-cache-v1.sqlite3").write_bytes(b"not a database")
    with pytest.raises(RuntimeError, match="SQLite cache failure"):
        SQLiteCacheManager("key").get_file("x")
    assert list(cache_root.iterdir()) == [cache_root / "triton-cache-v1.sqlite3"]


def test_materialization_recreated_and_old_path_unchanged(cache_root):
    manager = SQLiteCacheManager("key")
    old = Path(manager.put(b"old", "x.bin"))
    new = Path(manager.put(b"new", "x.bin"))
    assert old.read_bytes() == b"old"
    assert new.read_bytes() == b"new"
    new.unlink()
    assert Path(manager.get_file("x.bin")).read_bytes() == b"new"
    assert not old.is_relative_to(cache_root)
    with pytest.raises(ValueError, match="filename"):
        manager.put(b"x", "../escape")


def test_restart_and_runtime_cleanup(cache_root):
    script = """
import json
from triton.runtime.sqlite_cache import SQLiteCacheManager
m = SQLiteCacheManager('restart')
p = m.put(b'payload', 'x.bin')
m.put_group('x.json', {'x.bin': p})
print(json.dumps(p))
"""
    previous = Path(json.loads(subprocess.check_output([sys.executable, "-c", script], text=True)))
    assert not previous.exists()
    group = SQLiteCacheManager("restart").get_group("x.json")
    assert Path(group["x.bin"]).read_bytes() == b"payload"
    assert len(list(cache_root.rglob("*"))) == 1


def test_concurrent_processes_and_rollback(cache_root):
    script = """
import sys
from triton.runtime.sqlite_cache import SQLiteCacheManager
for i in range(30):
    m = SQLiteCacheManager(str(i))
    m.put(bytes([int(sys.argv[1])]) * 4096, 'shared.bin')
    m.put_group('group', {'shared.bin': 'unused'})
"""
    workers = [subprocess.Popen([sys.executable, "-c", script, str(i)]) for i in range(4)]
    for worker in workers:
        assert worker.wait(timeout=120) == 0
    for i in range(30):
        group = SQLiteCacheManager(str(i)).get_group("group")
        data = Path(group["shared.bin"]).read_bytes()
        assert data in [bytes([n]) * 4096 for n in range(4)]
    database = str(cache_root / "triton-cache-v1.sqlite3")
    script = """
import os, sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute('PRAGMA cache_size=5')
c.execute('BEGIN IMMEDIATE')
c.execute('DELETE FROM entries')
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, database], check=True, timeout=30)
    assert SQLiteCacheManager("0").get_group("group") is not None
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok", )
        assert connection.execute("SELECT count(*) FROM entries").fetchone()[0] == 60


def test_persistent_inode_reduction(cache_root):
    file_root, sqlite_root = cache_root / "file", cache_root / "sqlite"
    knobs.cache.dir = str(file_root)
    for i in range(100):
        manager = FileCacheManager(str(i))
        for j in range(3):
            manager.put(b"payload", f"{j}.bin")
    knobs.cache.dir = str(sqlite_root)
    for i in range(100):
        manager = SQLiteCacheManager(str(i))
        for j in range(3):
            manager.put(b"payload", f"{j}.bin")
    assert len(list(file_root.rglob("*"))) == 400
    assert len(list(sqlite_root.rglob("*"))) == 1


def test_threads_and_database_recreation(cache_root):
    manager = SQLiteCacheManager("threads")

    def put(i):
        path = manager.put(str(i), f"{i}.txt")
        return Path(path).read_text()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(put, range(40))) == [str(i) for i in range(40)]
    Path(manager.database).unlink()
    assert manager.get_file("0.txt") is None
    assert Path(manager.put(b"recovered", "new")).read_bytes() == b"recovered"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork requires POSIX")
def test_fork_cleanup_does_not_remove_parent_paths(cache_root):
    script = """
import os, sys
from pathlib import Path
from triton.runtime.sqlite_cache import SQLiteCacheManager
m = SQLiteCacheManager('fork')
p = m.put(b'parent', 'x.bin')
pid = os.fork()
if pid == 0:
    assert Path(m.get_file('x.bin')).read_bytes() == b'parent'
    sys.exit(0)
assert os.waitpid(pid, 0)[1] == 0
assert Path(p).read_bytes() == b'parent'
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
