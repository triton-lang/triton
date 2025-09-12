// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="arch-generation-name=gfx1100" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
// CHECK{LITERAL}: #smem = #ttg.shared_memory
// CHECK-LABEL: test_trans_global_load
// CHECK: %[[LOAD:.+]] = tt.load {{.*}} :  tensor<128x16x!tt.ptr<f32>, #blocked>
// CHECK: %[[ALLOC:.+]] = ttg.local_alloc %[[LOAD]] : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared, #smem>
// CHECK: %[[LOCAL_LOAD_TRANS:.+]] = ttg.local_load %[[ALLOC]] : !ttg.memdesc<128x16xf32, #shared, #smem> -> tensor<128x16xf32, #blocked>
// CHECK: %[[LOCAL_LOAD_DIRECT:.+]] = ttg.local_load %[[ALLOC]] : !ttg.memdesc<128x16xf32, #shared, #smem> -> tensor<128x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
// CHECK: tt.dot {{.+}}, %[[LOCAL_LOAD_DIRECT]], {{.+}} : tensor<16x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<128x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<16x16xf32, #blocked1>
// CHECK: %[[TRANS:.+]] = tt.trans %[[LOCAL_LOAD_TRANS]] {order = array<i32: 1, 0>} : tensor<128x16xf32, #blocked> -> tensor<16x128xf32, #blocked3>
// CHECK: %[[TRANS_DOT:.+]] = ttg.convert_layout %[[TRANS]] : tensor<16x128xf32, #blocked3> -> tensor<16x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
// CHECK: tt.dot {{.+}}, %[[TRANS_DOT]], {{.+}} : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<16x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<16x128xf32, #blocked2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_trans_global_load(
    %arg0 : tensor<128x16x!tt.ptr<f32>, #blocked2>,
    %out0 : tensor<16x16x!tt.ptr<f32>, #blocked3>,
    %out1 : tensor<16x128x!tt.ptr<f32>, #blocked>
  ) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked3>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked>

    %0 = tt.load %arg0 : tensor<128x16x!tt.ptr<f32>, #blocked2>
    %1 = ttg.convert_layout %0 : tensor<128x16xf32, #blocked2> -> tensor<128x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>>
    %2 = tt.dot %cst_0, %1, %cst_1 : tensor<16x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>> * tensor<128x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<16x16xf32, #blocked3>
    %3 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<128x16xf32, #blocked2> -> tensor<16x128xf32, #blocked4>
    %4 = ttg.convert_layout %3 : tensor<16x128xf32, #blocked4> -> tensor<16x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %5 = tt.dot %cst_2, %4, %cst_3 : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x128xf32, #blocked>

    tt.store %out0, %2 : tensor<16x16x!tt.ptr<f32>, #blocked3>
    tt.store %out1, %5 : tensor<16x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
