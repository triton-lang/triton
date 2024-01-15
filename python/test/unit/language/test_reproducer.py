import os
import shutil

import pytest

import torch
import triton


@triton.jit
def triton_():
    return


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_reproducer():
    tmpdir = ".tmp"
    reproducer = 'triton-reproducer.mlir'
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
    if os.path.exists(reproducer):
        os.remove(reproducer)
    os.environ["TRITON_CACHE_DIR"] = tmpdir
    os.environ["TRITON_REPRODUCER_PATH"] = reproducer
    triton_[(1, )]()
    foundPipeline = ""
    with open(reproducer, 'r') as f:
        line = f.read()
        if 'pipeline:' in line:
            foundPipeline = line
    if 0 == len(foundPipeline):
        raise Exception("Failed to find pipeline info in reproducer file.")
    if "convert-triton-gpu-to-llvm" not in foundPipeline:
        raise Exception("Failed to find triton passes in pipeline")
    # cleanup
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
    if os.path.exists(reproducer):
        os.remove(reproducer)
