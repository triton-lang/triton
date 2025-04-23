import triton
import triton.language as tl
from triton.compiler import ASTSource

target = triton.runtime.driver.active.get_current_target()


def test_signature_ordering():
    """
    Checks that ASTSource always uses the argument order from
    fn.arg_names and not the signature.
    """

    @triton.jit
    def kernel(a, o, N: tl.constexpr):
        tl.store(o + N, tl.load(a + N))

    # Add the arguments so the order always differs
    # from the order in fn.arg_names.
    signature = {}
    signature["N"] = "constexpr"
    signature["a"] = "*fp32"
    signature["o"] = "*fp32"
    src = ASTSource(
        fn=kernel,
        constexprs={"N": 32},
        signature=signature,
    )
    triton.compile(src=src, target=target)
