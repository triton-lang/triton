
from aot_compile.compile_metadata import jit, constexpr

# triton kernel
@jit
def kernel(X, stride_xm,
           Z, stride_zn,
           BLOCK_M: constexpr, BLOCK_N: constexpr):
           """ Dummy copy data """
           pass