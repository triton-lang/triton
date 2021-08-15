import torch
import triton
import pytest
import subprocess
import triton.language as tl
import numpy as np


@pytest.fixture(scope="module", autouse=True)
def print_p2p_matrix():
    stdout = subprocess.check_output(["nvidia-smi", "topo", "-p2p", "n"])
    print(stdout)


def get_p2p_matrix():
    stdout = subprocess.check_output(["nvidia-smi", "topo", "-p2p", "n"]).decode("ascii")
    lines = stdout.split("Legend")[0].split('\n')[1:]
    return np.array([line.split('\t')[1:-1] for line in lines][:-2])


def get_p2p_devices():
    matrix = get_p2p_matrix()
    idx = np.where(matrix == "OK")
    return f"cuda:{idx[0][0]}", f"cuda:{idx[1][0]}"


def get_non_p2p_devices():
    matrix = get_p2p_matrix()
    idx = np.where(matrix == "NS")
    return f"cuda:{idx[0][0]}", f"cuda:{idx[1][0]}"


p2p_devices = get_p2p_devices()
non_p2p_devices = get_non_p2p_devices()


@triton.jit
def _copy(from_ptr, to_ptr, N, **meta):
    pid = tl.program_id(0)
    offsets = pid * meta['BLOCK'] + tl.arange(0, meta['BLOCK'])
    values = tl.load(from_ptr + offsets, mask=offsets < N)
    tl.store(to_ptr + offsets, values, mask=offsets < N)


@pytest.mark.parametrize("device_kernel, device_from, device_to, stream_from, stream_to",
                         [(device_kernel, device_from, device_to, stream_from, stream_to)
                          for device_kernel in p2p_devices
                          for device_from in p2p_devices
                          for device_to in p2p_devices
                          for stream_from in ['default', 'custom']
                          for stream_to in ['default', 'custom']
                          ])
def test_p2p(device_kernel, device_from, device_to, stream_from, stream_to):
    if device_to == device_from:
        return
    torch.cuda.set_device(device_kernel)
    N = 512
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)

    with torch.cuda.stream(None if stream_from == 'default' else torch.cuda.Stream(device_from)):
        x_from = torch.randn(N, dtype=torch.float32, device=device_from)
    with torch.cuda.stream(None if stream_to == 'default' else torch.cuda.Stream(device_to)):
        x_to = torch.empty(N, dtype=torch.float32, device=device_to)

    _copy[grid](x_from, x_to, N, BLOCK=1024)
    assert torch.allclose(x_from, x_to.to(device_from))


@pytest.mark.parametrize("device_kernel, device_from, device_to, stream_from, stream_to",
                         [(device_kernel, device_from, device_to, stream_from, stream_to)
                          for device_kernel in non_p2p_devices
                          for device_from in non_p2p_devices
                          for device_to in non_p2p_devices
                          for stream_from in ['default', 'custom']
                          for stream_to in ['default', 'custom']
                          ])
def test_non_p2p(device_kernel, device_from, device_to, stream_from, stream_to):
    if device_to == device_from:
        return
    with pytest.raises(RuntimeError):
        torch.cuda.set_device(device_kernel)
        N = 512
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)

        with torch.cuda.stream(None if stream_from == 'default' else torch.cuda.Stream(device_from)):
            x_from = torch.randn(N, dtype=torch.float32, device=device_from)
        with torch.cuda.stream(None if stream_to == 'default' else torch.cuda.Stream(device_to)):
            x_to = torch.empty(N, dtype=torch.float32, device=device_to)

        _copy[grid](x_from, x_to, N, BLOCK=1024)
