import pytest
import triton

from triton._filecheck import run_filecheck_test


@triton.jit
def anchor(v):
    pass


# Smoke test to make sure filecheck is working correctly.
def test_filecheck_positive():

    @triton.jit
    def test_kernel():
        # CHECK-LABEL: test_kernel
        scalar = 42
        # CHECK: %c42_i32 = arith.constant 42 : i32
        # CHECK-NEXT: call @{{.*}}anchor{{.*}}(%c42_i32) : (i32) -> ()
        anchor(scalar)

    run_filecheck_test(test_kernel)


def test_filecheck_negative():

    @triton.jit
    def test_kernel():
        # CHECK-LABEL: test_kernel
        scalar = 11
        # CHECK: %c42_i32
        anchor(scalar)

    with pytest.raises(ValueError, match="expected string not found in input\n # CHECK: %c42_i32"):
        run_filecheck_test(test_kernel)
