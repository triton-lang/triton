import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


def test_compile_only() -> None:

    @triton.jit
    def kernel_add(a, b, c):
        idx = tl.arange(0, 32)
        tl.store(c + idx, tl.load(a + idx) + tl.load(b + idx))

    k = triton.compile(triton.compiler.ASTSource(fn=kernel_add, signature="*fp32,*fp32,*fp32", constants={}),
                       target=GPUTarget("cuda", 80, 32))
    print(k.asm["ttgir"])


test_compile_only()
