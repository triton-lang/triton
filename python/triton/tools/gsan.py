from __future__ import annotations

import argparse
import contextlib
import runpy
import sys
from pathlib import Path
from typing import Sequence

import triton
import torch
from triton.experimental.gsan._allocator import create_mem_pool


def _parse_args(argv: Sequence[str] | None = None) -> tuple[Path, list[str], str]:
    parser = argparse.ArgumentParser(description="Run a Python script with Triton Global Memory Sanitizer.")
    parser.add_argument("script", help="Python script to execute")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the target script")
    args = parser.parse_args(argv)

    script_args = list(args.script_args)
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]

    return Path(args.script), script_args, args.allocator_mode


@contextlib.contextmanager
def _script_context(script_path: Path, script_args: Sequence[str]):
    original_argv = sys.argv[:]
    original_path = sys.path[:]

    sys.argv = [str(script_path), *script_args]
    sys.path.insert(0, str(script_path.parent))
    try:
        yield script_path
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def main(argv: Sequence[str] | None = None) -> int:
    script_path, script_args, allocator_mode = _parse_args(argv)

    script_path = script_path.resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    triton.knobs.compilation.instrumentation = "gsan"

    with torch.use_mem_pool(create_mem_pool()), _script_context(script_path):
        runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
