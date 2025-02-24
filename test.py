import triton
import pathlib
import torch

ir = """
tt.func @kernel(%arg0: !tt.ptr<i32>) {
  %c0 = arith.constant 42 : i32
  ttg.warp_specialize(%arg0)
  default {
    tt.store %arg0, %c0 : !tt.ptr<i32>
    ttg.warp_yield
  }
  partition0(%arg1: !tt.ptr<i32>) num_warps(1) {
    ttg.warp_return
  } : (!tt.ptr<i32>) -> ()
  tt.return
}
"""

tmp = pathlib.Path("test.ttir")
tmp.write_text(ir)

kernel = triton.compile(str(tmp))
print(kernel.asm["ttgir"])

k = torch.empty(2, dtype=torch.int32, device='cuda')
kernel[(1,1,1)](k)
print(k)
