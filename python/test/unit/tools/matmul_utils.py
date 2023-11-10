import glob
import os
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

import triton

from .matmul_configs import MATMUL_ARGS, MATMUL_CONSTANTS

FIXTURES_DIR = Path(__file__).parent.absolute() / "fixtures"


# ------------------------------------------------------------------------------------------------------------ #


"""
Utilities for generating reference AOT kernels 
"""


# Copied from test/unittest/tools/test_aot.py
class AOTScriptRunner:
    """Wrapper around `triton.tools` for AOT compilation

    Runs `triton.tools.compile` and `triton.tools.link` in subprocesses
    """

    @staticmethod
    def compile_kernel(
        *, dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path
    ):
        compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

        subprocess.run(
            [
                sys.executable,
                compiler_path,
                "-n",
                kernel_name,
                "--signature",
                signature,
                "--out-name",
                out_name,
                "-o",
                out_path,
                "-w",
                str(num_warps),
                "-g",
                grid,
                # Save args passed to `triton.compiler.compile`
                "--save_args",
                # Save params dict used to hydrate kernel template
                "--save_params",
                kernel_path,
            ],
            check=True,
            cwd=dir,
        )

    @staticmethod
    def link_aot_kernels(dir, kernel_name):
        linker_path = os.path.join(triton.tools.__path__[0], "link.py")

        # link all desired configs
        h_files = glob.glob(os.path.join(dir, "*.h"))
        subprocess.run(
            [sys.executable, linker_path] + h_files + ["-o", kernel_name],
            check=True,
            cwd=dir,
        )

    @staticmethod
    def compile_matmul_kernels(
        kernel_name,
        signatures,
        num_warps,
        grids,
        out_dir=None,
        kernel_path=FIXTURES_DIR / "kernels" / "matmul_kernel.py",
    ):
        if isinstance(signatures, str):
            signatures = [signatures]
        if isinstance(num_warps, int):
            num_warps = [num_warps] * len(signatures)
        if isinstance(grids, str):
            grids = [grids] * len(signatures)
        assert len(signatures) == len(num_warps) == len(grids)

        out_dir = out_dir or FIXTURES_DIR / "aot_reference_kernels"

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        for s, w, g in zip(signatures, num_warps, grids):
            AOTScriptRunner.compile_kernel(
                dir=out_dir,
                signature=s,
                kernel_name=kernel_name,
                out_name=kernel_name,
                out_path=kernel_name,
                num_warps=w,
                grid=g,
                kernel_path=kernel_path,
            )
            AOTScriptRunner.link_aot_kernels(out_dir, kernel_name)

    @staticmethod
    def generate_signature(
        dtypes: OrderedDict,
        hints: OrderedDict,
        constant_vals: OrderedDict,
    ):
        assert set(dtypes.keys()) == set(MATMUL_ARGS)
        assert set(hints.keys()) == set(MATMUL_ARGS)

        args = []
        for arg in MATMUL_ARGS:
            dtype = dtypes[arg]
            hint = hints[arg]
            if hint:
                args.append(f"{dtype}:{str(hint)}")
            else:
                args.append(f"{dtype}")

        args_str = ", ".join(args)
        consts = []
        for const in MATMUL_CONSTANTS:
            consts.append(f"{constant_vals[const]}")
        consts_str = ", ".join(consts)
        signature = ", ".join([args_str, consts_str])
        return signature


@dataclass
class AOTScriptResult(dict):
    kernel_headers: List[Path]
    kernel_sources: List[Path]
    linked_header: List[Path]
    linked_source: List[Path]
    jit_args: List[Path]
    compiler_params: List[Path]

    def __post_init__(self):
        self.update(self.__dict__)


@dataclass
class MatmulTestConfig:
    dtypes: OrderedDict
    hints: OrderedDict
    constants: OrderedDict
    num_warps: int
    grid: str


def tt_to_torch(tt):
    if "16" in tt:
        return torch.float16
    elif "32" in tt:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype {tt}")
