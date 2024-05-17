// RUN: triton-opt %s -split-input-file -tritongpu-combine-tensor-select-and-if | FileCheck %s
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @select_if_combine
  tt.func public @select_if_combine(%arg0: tensor<64xf32, #blocked>, %dst_ptr: tensor<64x!tt.ptr<f32>, #blocked>, %cnd: i1) attributes {noinline = false} {
    // CHECK: %[[CST0:.*]] = arith.constant dense<0.000000e+00>
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    // CHECK: %[[CST1:.*]] = arith.constant dense<1.000000e+00>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    // CHECK-NOT: arith.select
    %sel = arith.select %cnd, %cst, %cst_1 : tensor<64xf32, #blocked>
    // CHECK: %[[IF_RES:.*]] = scf.if
    scf.if %cnd {
      tt.store %dst_ptr, %arg0 : tensor<64x!tt.ptr<f32>, #blocked>
    // CHECK: scf.yield %[[CST0]]
    }
    // CHECK: else
    // CHECK: scf.yield %[[CST1]]
    // CHECK: tt.store %{{.*}}, %[[IF_RES]]
    tt.store %dst_ptr, %sel : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
