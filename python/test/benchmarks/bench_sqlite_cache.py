"""Compare cache persistence and restart hits without compiling GPU kernels."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def worker(args):
    from triton.runtime.cache import get_cache_manager

    start = time.perf_counter()
    for i in range(args.entries):
        manager = get_cache_manager(f"{i:064x}")
        if args.worker == "write":
            group = {}
            for extension in ("ttir", "ptx", "cubin", "json"):
                name = "kernel." + extension
                group[name] = manager.put(f"{i}:{extension}:".encode() + b"x" * 1024, name)
            manager.put_group("kernel.json", group)
        else:
            group = manager.get_group("kernel.json")
            assert group is not None and len(group) == 4
            for name, path in group.items():
                assert Path(path).read_bytes().startswith(f"{i}:{name.split('.')[-1]}:".encode())
    seconds = time.perf_counter() - start
    runtime_objects = len(list(Path(os.environ["TMPDIR"]).rglob("*")))
    print(json.dumps({"seconds": seconds, "runtime_objects": runtime_objects}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=1000)
    parser.add_argument("--directory", help="Local parent directory for temporary benchmark storage")
    parser.add_argument("--worker", choices=("write", "read"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    results = {}
    with tempfile.TemporaryDirectory(dir=args.directory) as temporary:
        for backend in ("file", "sqlite"):
            root = Path(temporary) / backend
            cache, runtime = root / "cache", root / "runtime"
            runtime.mkdir(parents=True)
            env = dict(os.environ, TRITON_CACHE_BACKEND=backend,
                       TRITON_CACHE_DIR=str(cache), TMPDIR=str(runtime))
            env.pop("TRITON_CACHE_MANAGER", None)
            phases = {}
            for phase in ("write", "read"):
                output = subprocess.check_output([sys.executable, __file__, "--worker", phase,
                                                  "--entries", str(args.entries)], env=env, text=True)
                phases[phase] = json.loads(output)
            phases["persistent_objects"] = len(list(cache.rglob("*")))
            phases["runtime_objects_after_exit"] = len(list(runtime.rglob("*")))
            results[backend] = phases
    print(json.dumps({"kernels": args.entries, "logical_entries": args.entries * 5, "results": results}, indent=2))


if __name__ == "__main__":
    main()
