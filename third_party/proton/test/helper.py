import triton
import triton.language as tl

import torch


@triton.jit
def custom_add(a_ptr):
    tl.store(a_ptr, 1.0)


def main():
    a = torch.zeros(1)
    custom_add(a)


def test_main():
    main()


if __name__ == "__main__":
    main()
