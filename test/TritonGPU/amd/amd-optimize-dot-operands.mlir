// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="arch-generation-name=gfx950" | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="arch-generation-name=gfx942" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
// CHECK{LITERAL}: #smem = #ttg.shared_memory
// CHECK-LABEL: test_local_load_transposed
// CHECK: %[[LOAD:.+]] = tt.load {{.*}} : tensor<64x16x!tt.ptr<f16>, #blocked>
// CHECK: %[[ALLOC:.+]] = ttg.local_alloc %[[LOAD]] : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared, #smem>
// CHECK: %[[LOCAL_LOAD_TRANS:.+]] = ttg.local_load %[[ALLOC]] : !ttg.memdesc<64x16xf16, #shared, #smem> -> tensor<64x16xf16, #linear>
// CHECK: %[[LOCAL_LOAD_DIRECT:.+]] = ttg.local_load %[[ALLOC]] : !ttg.memdesc<64x16xf16, #shared, #smem> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
// CHECK: tt.dot {{.+}}, %[[LOCAL_LOAD_DIRECT]], {{.+}}: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x16xf32, #mma>
// CHECK: %[[TRANS:.+]] = tt.trans %[[LOCAL_LOAD_TRANS]] {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
// CHECK: tt.dot {{.+}}, %[[TRANS]], {{.+}} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x64xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_local_load_transposed(
    %arg0: tensor<64x16x!tt.ptr<f16>, #blocked>,
    %out0 : tensor<128x16x!tt.ptr<f32>, #blocked>,
    %out1 : tensor<128x64x!tt.ptr<f32>, #blocked>
  ) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_1 = arith.constant dense<0.693147182> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>>
    %cst_2 = arith.constant dense<0.581374812> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>

    %0 = tt.load %arg0 : tensor<64x16x!tt.ptr<f16>, #blocked>
    %1 = ttg.convert_layout %0 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #linear>
    %2 = ttg.convert_layout %0 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
    %3 = tt.dot %cst_1, %2, %cst_0 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x16xf32, #mma1>
    %4 = tt.trans %1 {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %5 = tt.dot %cst_2, %4, %cst_3 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>

    %6 = ttg.convert_layout %3 : tensor<128x16xf32, #mma1> -> tensor<128x16xf32, #blocked>
    %7 = ttg.convert_layout %5 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
    tt.store %out0, %6 : tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.store %out1, %7 : tensor<128x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-NOT: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
// CHECK-NOT: #smem = #ttg.shared_memory
// CHECK-LABEL: test_not_local_load_transposed_kWidth_mismatch
// CHECK: tt.load {{.*}} : tensor<64x16x!tt.ptr<f16>, #blocked>
// CHECK-NOT: ttg.local_alloc
// CHECK-NOT: ttg.local_load
// CHECK-NOT: ttg.local_load
// CHECK: tt.dot {{.+}}: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x16xf32, #mma>
// CHECK: tt.trans {{.+}} {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
// CHECK: tt.dot {{.+}} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x64xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_not_local_load_transposed_kWidth_mismatch(
    %arg0: tensor<64x16x!tt.ptr<f16>, #blocked>,
    %out0 : tensor<128x16x!tt.ptr<f32>, #blocked>,
    %out1 : tensor<128x64x!tt.ptr<f32>, #blocked>
  ) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_1 = arith.constant dense<0.693147182> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
    %cst_2 = arith.constant dense<0.581374812> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>

    %0 = tt.load %arg0 : tensor<64x16x!tt.ptr<f16>, #blocked>
    %1 = ttg.convert_layout %0 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #linear>
    %2 = ttg.convert_layout %0 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>>
    %3 = tt.dot %cst_1, %2, %cst_0 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>> -> tensor<128x16xf32, #mma1>
    %4 = tt.trans %1 {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %5 = tt.dot %cst_2, %4, %cst_3 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>

    %6 = ttg.convert_layout %3 : tensor<128x16xf32, #mma1> -> tensor<128x16xf32, #blocked>
    %7 = ttg.convert_layout %5 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
    tt.store %out0, %6 : tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.store %out1, %7 : tensor<128x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-NOT: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
// CHECK-NOT: #smem = #ttg.shared_memory
// CHECK-LABEL: test_not_local_load_transposed_opIdx_mismatch
// CHECK: tt.load {{.*}} : tensor<64x16x!tt.ptr<f16>, #blocked>
// CHECK-NOT: ttg.local_alloc
// CHECK-NOT: ttg.local_load
// CHECK-NOT: ttg.local_load
// CHECK: tt.dot {{.+}}: tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
// CHECK: tt.trans {{.+}} {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
// CHECK: tt.dot {{.+}} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x64xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_not_local_load_transposed_opIdx_mismatch(
    %arg0: tensor<64x16x!tt.ptr<f16>, #blocked>,
    %out0 : tensor<64x64x!tt.ptr<f32>, #blocked>,
    %out1 : tensor<128x64x!tt.ptr<f32>, #blocked>
  ) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma1>
    %cst_1 = arith.constant dense<0.693147182> : tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
    %cst_2 = arith.constant dense<0.581374812> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>

    %0 = tt.load %arg0 : tensor<64x16x!tt.ptr<f16>, #blocked>
    %1 = ttg.convert_layout %0 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #linear>
    %2 = ttg.convert_layout %0 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>>
    %3 = tt.dot %2, %cst_1, %cst_0 : tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<64x64xf32, #mma1>
    %4 = tt.trans %1 {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %5 = tt.dot %cst_2, %4, %cst_3 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>

    %6 = ttg.convert_layout %3 : tensor<64x64xf32, #mma1> -> tensor<64x64xf32, #blocked>
    %7 = ttg.convert_layout %5 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
    tt.store %out0, %6 : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.store %out1, %7 : tensor<128x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
