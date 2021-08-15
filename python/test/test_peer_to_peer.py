import torch
import triton
import pytest
import subprocess
import triton.language as tl


@pytest.fixture(scope="module", autouse=True)
def print_p2p_matrix():
    stdout = subprocess.check_output(["nvidia-smi", "topo", "-p2p", "n"])
    print(stdout)


@triton.jit
def _copy(from_ptr, to_ptr, N, **meta):
    pid = tl.program_id(0)
    offsets = pid * meta['BLOCK'] + tl.arange(0, meta['BLOCK'])
    values = tl.load(from_ptr + offsets, mask=offsets < N)
    tl.store(to_ptr + offsets, values, mask=offsets < N)


@pytest.mark.parametrize("N, device_kernel, device_from, device_to, stream_from, stream_to",
                         [(N, device_kernel, device_from, device_to, stream_from, stream_to)
                          for N in [512, 857, 1871, 2089, 8573, 31000]
                          for device_kernel in ['cuda:0', 'cuda:1']
                          for device_from in ['cuda:0', 'cuda:1']
                          for device_to in ['cuda:0', 'cuda:1']
                          for stream_from in ['default', 'custom']
                          for stream_to in ['default', 'custom']
                          ])
def test_op(N, device_kernel, device_from, device_to, stream_from, stream_to):
    torch.cuda.set_device(device_kernel)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)

    with torch.cuda.stream(None if stream_from == 'default' else torch.cuda.Stream(device_from)):
        x_from = torch.randn(N, dtype=torch.float32, device=device_from)
    with torch.cuda.stream(None if stream_to == 'default' else torch.cuda.Stream(device_to)):
        x_to = torch.empty(N, dtype=torch.float32, device=device_to)

    _copy[grid](x_from, x_to, N, BLOCK=1024)
    assert torch.allclose(x_from, x_to.to(device_from))
