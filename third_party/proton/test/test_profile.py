import torch
import triton
import triton.profiler as proton
import tempfile
import json
import pytest
from typing import NamedTuple

import triton.language as tl


@pytest.mark.parametrize("context", ["shadow", "python"])
def test_torch(context):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0], context=context)
        proton.enter_scope("test")
        torch.ones((2, 2), device="cuda")
        proton.exit_scope()
        proton.finalize()
        data = json.load(f)
        if context == "shadow":
            assert len(data[0]["children"]) == 1
            assert data[0]["children"][0]["frame"]["name"] == "test"
        elif context == "python":
            assert len(data[0]["children"]) == 1
            # The last frame is the torch kernel
            prev_frame = data
            curr_frame = data[0]["children"]
            while len(curr_frame) > 0:
                prev_frame = curr_frame
                curr_frame = curr_frame[0]["children"]
            assert "elementwise_kernel" in prev_frame[0]["frame"]["name"]


def test_triton():

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device="cuda")
    y = torch.zeros_like(x)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0])
        with proton.scope("test0"):
            with proton.scope("test1"):
                foo[(1, )](x, y)
        with proton.scope("test2"):
            foo[(1, )](x, y)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 2
        assert data[0]["children"][0]["frame"]["name"] == "test0"
        assert len(data[0]["children"][0]["children"]) == 1
        assert data[0]["children"][0]["children"][0]["frame"]["name"] == "test1"
        assert data[0]["children"][1]["frame"]["name"] == "test2"


def test_metrics():

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device="cuda")
    y = torch.zeros_like(x)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0])
        with proton.scope("test0", {"foo": 1.0}):
            foo[(1, )](x, y)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 1
        assert data[0]["children"][0]["frame"]["name"] == "test0"
        assert data[0]["children"][0]["metrics"]["foo"] == 1.0


def test_metrics_ignore():

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device="cuda")
    y = torch.zeros_like(x)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        session_id = proton.start(f.name.split(".")[0])
        proton.deactivate(session_id)
        with proton.scope("test0", {"foo": 1.0}):
            foo[(1, )](x, y)
        proton.activate(session_id)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 0


def test_scope_backward():
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0])
        with proton.scope("ones1"):
            a = torch.ones((100, 100), device="cuda", requires_grad=True)
        with proton.scope("plus"):
            a2 = a * a * a
        with proton.scope("ones2"):
            loss = torch.ones_like(a2)

        # Backward triggers two kernels in a single scope
        with proton.scope("backward"):
            a2.backward(loss)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 4


def test_hook():

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        # get arg's element size
        element_size = args["x"].element_size()  # non-const
        size = args["size"]  # const
        key = "flops" + str(element_size * 8)
        num_ctas = metadata.num_ctas
        return {"name": f"foo_test_{num_ctas}ctas_{size}elems", key: 1.0}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.tensor([2], device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0], hook="triton")
        with proton.scope("test0"):
            foo[(1, )](x, 1, y, num_warps=4)
        proton.finalize()
        data = json.load(f)
        assert len(data[0]["children"]) == 1
        assert data[0]["children"][0]["frame"]["name"] == "test0"
        assert data[0]["children"][0]["children"][0]["frame"]["name"] == "foo_test_1ctas_1elems"
        assert data[0]["children"][0]["children"][0]["metrics"]["flops32"] == 1.0
