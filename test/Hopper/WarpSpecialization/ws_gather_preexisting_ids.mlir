// RUN: not triton-opt %s --nvgpu-warp-specialization=num-stages=2 -o /dev/null 2>&1 | FileCheck %s

// CHECK: warp specialization cannot fall back from unsupported gather in warp-specialized function with preexisting async_task_id attributes

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @preexisting_unsupported_gather(%input: tensor<128x64xf16, #blocked>, %indices: tensor<128x64xi32, #blocked>, %output: tensor<128x64x!tt.ptr<f16>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      %gathered = tt.gather %input[%indices] {async_task_id = array<i32: 0>, axis = 1 : i32} : (tensor<128x64xf16, #blocked>, tensor<128x64xi32, #blocked>) -> tensor<128x64xf16, #blocked>
      tt.store %output, %gathered : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
