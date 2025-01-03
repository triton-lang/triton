// RUN: triton-opt %s --split-input-file --set-specific-allocation-size=arch=gfx942 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: atomic_lds_reduction
  tt.func @atomic_lds_reduction(%arg0 : tensor<128x!tt.ptr<f32>, #blocked>, %arg2 : tensor<128xf32, #blocked>) {
    // CHECK: tt.atomic_rmw fadd, relaxed, gpu{{.*}}allocation.size = 512 : i32
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2 : (tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.return
  }
}
