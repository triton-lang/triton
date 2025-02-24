import triton
import pathlib

ir = """
tt.func @kernel() {
  ttg.warp_specialize()
  default {
    tt.print "hello world" {hex = false, isSigned = array<i32>}
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}
"""

tmp = pathlib.Path("test.ttir")
tmp.write_text(ir)

kernel = triton.compile(str(tmp))
print(kernel.asm["ttgir"])

kernel[(1,1,1)]()
