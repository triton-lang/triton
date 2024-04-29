import triton
import triton.profiler as proton
import triton.language as tl

import torch
import sys


@triton.jit
def custom_add(a_ptr):
    tl.store(a_ptr, 1.0)


def main():
    a = torch.zeros(1, device="cuda")
    with proton.scope("test"):
        custom_add[(1, )](a)


def test_main():
    main()


if __name__ == "__main__":
    if sys.argv[1] == "test":
        main()
