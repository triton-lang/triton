Optional SQLite local cache
===========================

Set ``TRITON_CACHE_BACKEND=sqlite`` before starting Python to store compiler
artifacts in a single SQLite database instead of persistent per-artifact files::

    export TRITON_CACHE_BACKEND=sqlite
    export TRITON_CACHE_DIR=/path/to/local/cache/triton
    export TMPDIR=/path/to/local/runtime
    python train.py

This experimental backend is intended for inode-constrained local caches.
``file`` remains the default backend; unknown names raise an error. The database
is ``TRITON_CACHE_DIR/triton-cache-v1.sqlite3``. The existing default directory and
``TRITON_HOME`` behavior apply when ``TRITON_CACHE_DIR`` is unset. No existing
file entries are migrated. ``TRITON_CACHE_MANAGER`` explicitly selects a custom
manager and takes precedence over the backend setting. Remote managers retain
their existing behavior. ``TRITON_STORE_BINARY_ONLY`` still controls the set of
compiler artifacts stored. Dump and override directories remain file-based.

SQLite stores text as UTF-8 and binary data as bytes. Group metadata stores
member names, never runtime paths. Groups are published transactionally after
checking membership and read in a single database snapshot. Incomplete or corrupt
group metadata is treated as a miss.

Paths and temporary inode use
----------------------------

The cache manager API returns paths. Compiler consumers, native loaders, and
instrumentation hooks still require these files. ``put``, ``get_file``, and
``get_group`` materialize the requested objects below Python's temporary directory
(``TMPDIR`` on Unix). Content-specific paths remain valid until process exit,
even when a logical cache entry is subsequently replaced. Normal exit removes
the process's materializations; forked children do not remove their parent's files.

Persistent inode use is approximately one database plus a transient rollback
journal. **Temporary inode use is not bounded** and scales with artifacts touched
by live processes. Place temporary storage outside the constrained persistent
cache and provision enough inodes for it. Abnormal termination may leave
``triton-sqlite-*`` directories; remove only directories of stopped processes.
Tools that scan the old cache layout directly, including filesystem-based
artifact bundlers, cannot discover SQLite entries; use the cache-manager API.
There is no TTL, LRU, size cap, or remote synchronization in this initial backend.

Concurrency and recovery
------------------------

Use local storage for the database and temporary files. Sharing a database across
hosts on NFS, Lustre, or another distributed filesystem is unsupported. SQLite's
rollback journal, transactions, and 30-second busy timeout coordinate local
processes. Connections close after each operation and are not shared across
threads or inherited by forked workers. A ``triton-cache-v1.sqlite3-journal`` file can
exist while a transaction is active.

Database errors are reported without an automatic fallback to per-artifact
persistent files. To clear the cache, stop all users, then remove
``triton-cache-v1.sqlite3`` and any associated journal files in the configured directory.
Never delete a journal while a writer is active. Old filesystem caches and
explicit dump files can be cleared separately after their users have stopped.

Shared storage and read-only snapshots
-------------------------------------

Several processes on one node may share a locally stored database. A network
filesystem does not acquire local locking/synchronization guarantees merely
because only one node currently uses it. This backend does not detect filesystem
types. See `SQLite's network-filesystem guidance <https://sqlite.org/useovernet.html>`_.

Lookups do not create a missing database or initialize its schema. SQLite can
still require writable access to recover a hot rollback journal after a writer
is interrupted. Compilation calls ``put`` and ``put_group`` on misses, so a complete
read-only compiler cache is not supported. Read-only permissions on one client
do not stop writers on another client. The backend does not use ``immutable=1``:
that SQLite option disables locking and change detection and is only safe when
the file cannot change at all.

For multi-node jobs, publish a consistent snapshot to shared storage and stage a
writable copy into each node's local cache before launching workers. Create the
snapshot with all writers stopped and connections closed, or use SQLite's backup
API. Do not copy a live database without its transactional state. Keep runtime
materializations local too. There is no automatic read-only base/local-write
overlay, nor any merging of caches written independently on different nodes.

``v1`` in the filename denotes the backend schema, not the Triton or SQLite
release. The ``triton-`` prefix prevents collisions with Inductor's distinct
schema if both cache directory variables point to the same directory. Earlier
experimental ``cache-v1.sqlite3`` files are ignored; entries are rebuilt. Remove
old databases only after their users have stopped.

Measuring storage overhead
-------------------------

Run ``python python/test/benchmarks/bench_sqlite_cache.py --entries 2000 --directory /path/to/local/storage``
to compare 2,000 synthetic kernels with four artifacts and one group record each.
One local run measured:

.. list-table:: Storage-only measurements for 10,000 logical records
   :header-rows: 1

   * - Backend
     - Persistent objects
     - Write seconds
     - Restart-read seconds
     - Live temporary objects
   * - File
     - 12,000
     - 0.70
     - 0.21
     - 0
   * - SQLite
     - 1
     - 6.44
     - 1.26
     - 16,003

Temporary objects returned to zero after normal process exit. Counts exclude the
benchmark's containing directories. Timings exclude imports and GPU compilation;
they are not an end-to-end performance prediction. SQLite durability and
materialization add overhead. Re-run on the intended local filesystem.
