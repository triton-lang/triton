import triton
import triton.language as tl
import triton._C.libtriton.triton as _triton


@triton.jit
def foo(a, b):
  max, min = maxmin(a, b)
  return max, min

@triton.jit
def maxmin(a, b):
  max = tl.maximum(a, b)
  min = tl.minimum(a, b)
  return max, min


mod, ctx = foo.compile_to_ttir(3, 4, grid=(1,))
assert mod.verify()
mod.dump()


pm = _triton.ir.pass_manager(ctx)
pm.add_inliner_pass()
pm.run(mod)
assert mod.verify()
mod.dump()
