import torch
import triton
import triton.language as tl
import triton.profiler as proton
import json
import tempfile
import os


@triton.jit
def foo(x, y, size: tl.constexpr):
    offs = tl.arange(0, size)
    tl.store(y + offs, tl.load(x + offs))


with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "test_trace")
    proton.start(path, data="trace")
    with proton.scope("init"):
        x = torch.ones((1024, ), device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)
    with proton.scope("test"):
        foo[(1, )](x, y, x.size()[0], num_warps=4)
    proton.finalize()
    with open(path + ".chrome_trace") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))
