"""Regression tests for triton-lang/triton#11512 signature 1.

`FileCacheManager.put()` writes into a private per-writer temp directory,
atomically `os.replace`s the file to its final name, and then cleans up that
directory. Two defects lived in that cleanup step:

  * it sat in the failure path of an already-successful publish, so a filesystem
    that refused the leaf `rmdir` with EBUSY (observed on distributed
    filesystems) crashed a compilation whose real work had completed;
  * it used `os.removedirs`, which keeps walking up after the leaf and prunes the
    shared cache key directory and its ancestors whenever they are empty.

The contract these tests pin down:

  * a cleanup failure after a successful publish must not fail `put()`, and the
    published file must remain readable with the right content;
  * cleanup stays inside this writer's own temp directory;
  * genuine failures — a failed temp write or a failed publish — still raise;
  * the semantics callers rely on (`put` of text/bytes, `put_group`, distinct
    keys, cache hits) are unchanged;
  * many processes publishing the same key concurrently all succeed.

Fault injection is deliberately narrow: the wrapper fails only the leaf `rmdir`
of this test's own writer temp directory, only after the published file already
exists, and only once; every other path reaches the real `os.rmdir`.
"""
import errno
import json
import multiprocessing as mp
import os
import pathlib
import uuid

import pytest

from triton import knobs
from triton.runtime.cache import FileCacheManager, _base32, get_cache_manager


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_manager(cache_dir, key=None):
    """A FileCacheManager rooted at `cache_dir`, built through the real factory.

    Goes through `get_cache_manager` so the production construction path is what
    the tests exercise, and asserts the factory returned the file-backed manager
    (a TRITON_CACHE_MANAGER override would otherwise make every test below pass
    vacuously).
    """
    key = key or uuid.uuid4().hex
    with knobs.cache.scope():
        knobs.cache.dir = str(cache_dir)
        mgr = get_cache_manager(key)
    assert type(mgr) is FileCacheManager, f"unexpected cache manager {type(mgr)}"
    return mgr


class _BusyOnPublishCleanup:
    """Fail the post-publish leaf rmdir of the writer's own temp dir with EBUSY.

    Armed only once the file has been published (the final path exists), so the
    injected failure lands exactly on the cleanup step described in the issue and
    never on an earlier stage of `put()`.
    """

    def __init__(self, cache_dir, published_name, errno_value=errno.EBUSY):
        self.cache_dir = str(cache_dir)
        self.published_name = published_name
        self.errno_value = errno_value
        self.real_rmdir = os.rmdir
        self.failures_injected = 0
        self.removals = []

    def __enter__(self):
        def fake_rmdir(path, *args, **kwargs):
            target = os.fspath(path)
            self.removals.append(target)
            published = os.path.join(self.cache_dir, self.published_name)
            if (self.failures_injected == 0 and os.path.exists(published)
                    and target.startswith(self.cache_dir)
                    and os.path.basename(target).startswith("tmp.pid_")):
                self.failures_injected += 1
                raise OSError(self.errno_value, os.strerror(self.errno_value), target)
            return self.real_rmdir(path, *args, **kwargs)

        os.rmdir = fake_rmdir
        return self

    def __exit__(self, *exc):
        os.rmdir = self.real_rmdir
        return False


class _FailOnReplace:
    """Make `os.replace` raise a chosen OSError."""

    def __init__(self, errno_value=errno.EACCES):
        self.errno_value = errno_value
        self.real_replace = os.replace
        self.calls = 0

    def __enter__(self):
        def fake_replace(*args, **kwargs):
            self.calls += 1
            src = args[0] if args else ""
            raise OSError(self.errno_value, os.strerror(self.errno_value), src)
        os.replace = fake_replace
        return self

    def __exit__(self, *exc):
        os.replace = self.real_replace
        return False


# --------------------------------------------------------------------------- #
# signature 1: cleanup must not fail a successful publish
# --------------------------------------------------------------------------- #
def test_cache_put_preserves_published_file_on_cleanup_ebusy(tmp_path: pathlib.Path):
    """EBUSY from the post-publish cleanup must not fail `put()` or lose data."""
    payload = "kernel-metadata-payload"
    mgr = _make_manager(tmp_path)
    with _BusyOnPublishCleanup(mgr.cache_dir, "payload.txt") as inj:
        path = mgr.put(payload, "payload.txt", binary=False)

    assert inj.failures_injected == 1, "the EBUSY injection did not reach the cleanup step"
    assert path is not None
    assert os.path.isfile(path), "put() returned a path that does not exist"
    assert pathlib.Path(path).read_text() == payload, "published content was lost"


def test_cache_put_binary_preserves_published_file_on_cleanup_ebusy(tmp_path: pathlib.Path):
    payload = bytes(range(256))
    mgr = _make_manager(tmp_path)
    with _BusyOnPublishCleanup(mgr.cache_dir, "blob.bin") as inj:
        path = mgr.put(payload, "blob.bin", binary=True)

    assert inj.failures_injected == 1
    assert pathlib.Path(path).read_bytes() == payload


def test_cache_put_group_preserves_published_metadata_on_cleanup_ebusy(tmp_path: pathlib.Path):
    """`put_group` goes through the same write path, so it gets the same behaviour.

    This does not claim transactional atomicity for a multi-file group.
    """
    group = {"cubin": "/somewhere/kernel.cubin", "json": "/somewhere/kernel.json"}
    mgr = _make_manager(tmp_path)
    with _BusyOnPublishCleanup(mgr.cache_dir, "__grp__kernel.json") as inj:
        path = mgr.put_group("kernel.json", group)

    assert inj.failures_injected == 1
    assert os.path.isfile(path)
    assert json.loads(pathlib.Path(path).read_text()) == {"child_paths": group}


def test_cache_put_survives_repeated_cleanup_failures(tmp_path: pathlib.Path):
    """A filesystem that keeps refusing cleanup must not poison later puts."""
    mgr = _make_manager(tmp_path)
    for i in range(5):
        name = f"f{i}.txt"
        with _BusyOnPublishCleanup(mgr.cache_dir, name) as inj:
            path = mgr.put(f"value-{i}", name, binary=False)
        assert inj.failures_injected == 1
        assert pathlib.Path(path).read_text() == f"value-{i}"

    for i in range(5):
        assert pathlib.Path(mgr.get_file(f"f{i}.txt")).read_text() == f"value-{i}"


def test_cache_put_cleanup_stays_inside_writer_temp_dir(tmp_path: pathlib.Path):
    """Cleanup must not walk up into the shared key directory or the cache root."""
    mgr = _make_manager(tmp_path)
    key_dir = pathlib.Path(mgr.cache_dir)
    removals = []
    real_rmdir = os.rmdir

    def record_rmdir(path, *args, **kwargs):
        removals.append(os.fspath(path))
        return real_rmdir(path, *args, **kwargs)

    os.rmdir = record_rmdir
    try:
        mgr.put("content", "kept.txt", binary=False)
    finally:
        os.rmdir = real_rmdir

    assert removals, "put() performed no directory cleanup at all"
    for target in removals:
        assert os.path.dirname(target) == str(key_dir), \
            f"cleanup escaped the writer temp dir and touched {target}"
        assert os.path.basename(target).startswith("tmp.pid_"), \
            f"cleanup touched a non-temp path {target}"
    assert key_dir.is_dir(), "the shared cache key directory was pruned"
    assert pathlib.Path(tmp_path).is_dir(), "the cache root was pruned"


def test_cache_put_propagates_unexpected_cleanup_error(tmp_path: pathlib.Path):
    """Only concurrent-writer errnos are ignorable; anything else must raise.

    The cleanup guard ignores EBUSY/ENOTEMPTY/ENOENT because another process on
    the same key directory can produce them after our publish has succeeded.
    EACCES is not one of those, so swallowing it would hide a real problem.
    """
    mgr = _make_manager(tmp_path)
    with _BusyOnPublishCleanup(mgr.cache_dir, "denied.txt",
                               errno_value=errno.EACCES) as inj:
        with pytest.raises(OSError) as excinfo:
            mgr.put("content", "denied.txt", binary=False)

    assert inj.failures_injected == 1
    assert excinfo.value.errno == errno.EACCES


# --------------------------------------------------------------------------- #
# genuine failures still propagate
# --------------------------------------------------------------------------- #
def test_cache_put_propagates_publish_failure(tmp_path: pathlib.Path):
    """A failing publish must still raise, and must not be masked by cleanup."""
    mgr = _make_manager(tmp_path)
    with _FailOnReplace(errno.EACCES) as fail:
        with pytest.raises(OSError) as excinfo:
            mgr.put("data", "not-published.txt", binary=False)

    assert fail.calls == 1
    assert excinfo.value.errno == errno.EACCES
    assert mgr.get_file("not-published.txt") is None, "a failed publish left a visible file"


def test_cache_put_propagates_write_failure(tmp_path: pathlib.Path, monkeypatch):
    """A failing temp write must still raise."""
    mgr = _make_manager(tmp_path)
    real_open = open

    def failing_open(path, *args, **kwargs):
        if os.path.basename(os.fspath(path)) == "unwritable.txt":
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), os.fspath(path))
        return real_open(path, *args, **kwargs)

    armed = {"on": True}

    def failing_open(path, *args, **kwargs):
        if armed["on"] and os.path.basename(os.fspath(path)) == "unwritable.txt":
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)
    with pytest.raises(OSError) as excinfo:
        mgr.put("data", "unwritable.txt", binary=False)

    assert excinfo.value.errno == errno.ENOSPC
    # Disarm before asking whether the entry is usable: the injected fault is a
    # write failure, and a read-back check is a separate contract (signature 2).
    armed["on"] = False
    assert mgr.get_file("unwritable.txt") is None


# --------------------------------------------------------------------------- #
# semantics callers rely on
# --------------------------------------------------------------------------- #
def test_cache_put_text_and_bytes_round_trip(tmp_path: pathlib.Path):
    mgr = _make_manager(tmp_path)
    text_path = mgr.put("plain text", "t.txt", binary=False)
    bin_path = mgr.put(b"\x00\x01\x02", "b.bin", binary=True)

    assert pathlib.Path(text_path).read_text() == "plain text"
    assert pathlib.Path(bin_path).read_bytes() == b"\x00\x01\x02"
    assert mgr.get_file("t.txt") == text_path
    assert mgr.get_file("b.bin") == bin_path
    assert mgr.get_file("missing.txt") is None


def test_cache_put_distinct_keys_do_not_interfere(tmp_path: pathlib.Path):
    a, b = _make_manager(tmp_path), _make_manager(tmp_path)
    assert a.cache_dir != b.cache_dir
    pa = a.put("value-a", "same-name.txt", binary=False)
    pb = b.put("value-b", "same-name.txt", binary=False)

    assert pathlib.Path(pa).read_text() == "value-a"
    assert pathlib.Path(pb).read_text() == "value-b"
    assert a.get_file("same-name.txt") != b.get_file("same-name.txt")


def test_cache_put_leaves_no_temp_dir_after_success(tmp_path: pathlib.Path):
    mgr = _make_manager(tmp_path)
    mgr.put("x", "clean.txt", binary=False)
    leftovers = [p for p in os.listdir(mgr.cache_dir) if p.startswith("tmp.pid_")]
    assert leftovers == [], f"successful put left temp dirs behind: {leftovers}"


# --------------------------------------------------------------------------- #
# concurrency: the situation the issue actually describes
# --------------------------------------------------------------------------- #
def _concurrent_writer(cache_dir, key, name, payload, barrier, queue, index, rounds):
    """Publish the same key from a fresh process, barrier-aligned per round."""
    import traceback as _tb

    failures, completed = [], 0
    try:
        from triton import knobs as _knobs
        from triton.runtime.cache import FileCacheManager as mgr_cls, _base32 as base32

        with _knobs.cache.scope():
            _knobs.cache.dir = cache_dir
            mgr = mgr_cls(base32(key))
            for _ in range(rounds):
                barrier.wait(timeout=300)
                try:
                    path = mgr.put(payload, name, binary=False)
                    # Consume what was just published, the way CompiledKernel does.
                    with open(path) as handle:
                        if handle.read() != payload:
                            failures.append("read-back mismatch")
                        else:
                            completed += 1
                except OSError as exc:
                    failures.append(f"OSError errno={exc.errno}: {exc}")
                except Exception:
                    failures.append(_tb.format_exc())
    except Exception:
        failures.append(_tb.format_exc())
    queue.put((index, completed, failures))


@pytest.mark.parametrize("nproc,rounds", [(4, 8), (8, 5)])
def test_cache_concurrent_same_key_multiprocess(tmp_path: pathlib.Path, nproc, rounds):
    """Many processes publishing one key concurrently must all succeed.

    This is the workload from the issue without the proprietary filesystem: a
    shared cache directory and barrier-aligned same-key publishes. Every worker
    must report back and exit 0, so the success count has a trustworthy
    denominator.
    """
    cache_dir = str(tmp_path / "shared_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key, name, payload = "deadbeef" * 8, "shared.txt", "shared-payload"

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(nproc)
    queue = ctx.Queue()
    procs = [ctx.Process(target=_concurrent_writer,
                         args=(cache_dir, key, name, payload, barrier, queue, i, rounds))
             for i in range(nproc)]
    for p in procs:
        p.start()
    results = [queue.get(timeout=600) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert not p.is_alive(), f"worker {p.pid} did not finish"
        assert p.exitcode == 0, f"worker {p.pid} exited {p.exitcode}"

    assert len(results) == nproc, f"only {len(results)}/{nproc} workers reported"
    total_ok = sum(ok for _, ok, _ in results)
    all_failures = [f for _, _, fl in results for f in fl]
    assert not all_failures, f"concurrent publishes failed: {all_failures[:3]}"
    assert total_ok == nproc * rounds, f"{total_ok} != {nproc * rounds} successful publishes"

    final = os.path.join(cache_dir, _base32(key), name)
    assert pathlib.Path(final).read_text() == payload
