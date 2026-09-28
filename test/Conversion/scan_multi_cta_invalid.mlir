// RUN: not triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=80 2>&1 | FileCheck %s
// RUN: not triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=120 2>&1 | FileCheck %s

// CHECK: error: tt.scan across CTAs requires distributed shared memory support

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @scan_axis_cta_unsupported(%v: tensor<32x32xf32, #blocked>) {
    %r = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      "tt.scan.return"(%s) : (f32) -> ()
    }) : (tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked>
    tt.return
  }
}
