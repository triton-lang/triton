// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline -canonicalize | FileCheck %s

// Pick a common shared memory layout with vec = max kWidth of all users.
// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
// CHECK-NOT: #ttg.swizzled_shared
// CHECK{LITERAL}: #smem = #ttg.shared_memory
// CHECK-LABEL: test_lds_layout_selection

// CHECK: %[[ALLOC:.+]] = ttg.local_alloc : () -> !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>
// CHECK: %[[MEMDESC_IDX:.+]] = ttg.memdesc_index %[[ALLOC]]

// CHECK: scf.for {{.+}} iter_args({{.*}}, %[[MEMDESC_IDX_ITER:.+]] = %[[MEMDESC_IDX]]) -> ({{.+}})
//  CHECK: %[[LOAD:.+]] = tt.load {{.+}} : tensor<64x16x!tt.ptr<f16>, #blocked>
//  CHECK: %[[LOCAL_LOAD_TRANS:.+]] = ttg.local_load %[[MEMDESC_IDX_ITER]] : {{.+}} -> tensor<64x16xf16, #linear>
//  CHECK: %[[LOCAL_LOAD_DIRECT:.+]] = ttg.local_load %[[MEMDESC_IDX_ITER]] : {{.+}} -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
//  CHECK: tt.dot {{.+}}, %[[LOCAL_LOAD_DIRECT]], {{.+}}
//  CHECK: %[[TRANS:.+]] = tt.trans %[[LOCAL_LOAD_TRANS]] {{.+}} : {{.+}} -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
//  CHECK: tt.dot {{.+}}, %[[TRANS]], {{.+}}
//  CHECK: %[[MEMDESC_IDX:.+]] = ttg.memdesc_index %[[ALLOC]]
//  CHECK: ttg.local_store %[[LOAD]], %[[MEMDESC_IDX]]
//  CHECK: scf.yield

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_lds_layout_selection(
    %arg0: tensor<64x16x!tt.ptr<f16>, #blocked>,
    %out0 : tensor<128x16x!tt.ptr<f32>, #blocked>,
    %out1 : tensor<128x64x!tt.ptr<f32>, #blocked>
  ) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_1 = arith.constant dense<0.693147182> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
    %cst_2 = arith.constant dense<0.581374812> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32

    %0:2 = scf.for %arg1 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg2 = %cst_0, %arg3 = %cst_3) -> (tensor<128x16xf32, #mma1>, tensor<128x64xf32, #mma>)  : i32 {
      %1 = tt.load %arg0 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #linear>
      %3 = ttg.convert_layout %1 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>>
      %4 = tt.dot %cst_1, %3, %arg2 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>> -> tensor<128x16xf32, #mma1>
      %5 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %6 = tt.dot %cst_2, %5, %arg3 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
      scf.yield %4, %6 : tensor<128x16xf32, #mma1>, tensor<128x64xf32, #mma>
    }

    %7 = ttg.convert_layout %0#0 : tensor<128x16xf32, #mma1> -> tensor<128x16xf32, #blocked>
    %8 = ttg.convert_layout %0#1 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
    tt.store %out0, %7 : tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.store %out1, %8 : tensor<128x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// -----

// Verify that a common shared memory layout is chosen for users with different kWidth and opIdx.
// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
// CHECK-NOT: #ttg.swizzled_shared
// CHECK{LITERAL}: #smem = #ttg.shared_memory
// CHECK-LABEL: test_lds_layout_selection_different_opIdx

// CHECK: %[[ALLOC:.+]] = ttg.local_alloc : () -> !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>
// CHECK: %[[MEMDESC_IDX:.+]] = ttg.memdesc_index %[[ALLOC]]

// CHECK: scf.for {{.+}} iter_args({{.*}}, %[[MEMDESC_IDX_ITER:.+]] = %[[MEMDESC_IDX]]) -> ({{.+}})
//  CHECK: %[[LOAD:.+]] = tt.load {{.+}} : tensor<64x16x!tt.ptr<f16>, #blocked>
//  CHECK: %[[LOCAL_LOAD_TRANS:.+]] = ttg.local_load %[[MEMDESC_IDX_ITER]] : {{.+}} -> tensor<64x16xf16, #linear>
//  CHECK: %[[LOCAL_LOAD_DIRECT:.+]] = ttg.local_load %[[MEMDESC_IDX_ITER]] : {{.+}} -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
//  CHECK: tt.dot %[[LOCAL_LOAD_DIRECT]], {{.+}}
//  CHECK: %[[TRANS:.+]] = tt.trans %[[LOCAL_LOAD_TRANS]] {{.+}} : {{.+}} -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
//  CHECK: tt.dot {{.+}}, %[[TRANS]], {{.+}}
//  CHECK: %[[MEMDESC_IDX:.+]] = ttg.memdesc_index %[[ALLOC]]
//  CHECK: ttg.local_store %[[LOAD]], %[[MEMDESC_IDX]]
//  CHECK: scf.yield

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_lds_layout_selection_different_opIdx(
    %arg0: tensor<64x16x!tt.ptr<f16>, #blocked>,
    %out0 : tensor<64x64x!tt.ptr<f32>, #blocked>,
    %out1 : tensor<128x64x!tt.ptr<f32>, #blocked>
  ) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma1>
    %cst_1 = arith.constant dense<0.693147182> : tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>>
    %cst_2 = arith.constant dense<0.581374812> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32

    %0:2 = scf.for %arg1 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg2 = %cst_0, %arg3 = %cst_3) -> (tensor<64x64xf32, #mma1>, tensor<128x64xf32, #mma>)  : i32 {
      %1 = tt.load %arg0 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #linear>
      %3 = ttg.convert_layout %1 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
      %4 = tt.dot %3, %cst_1, %arg2 : tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>> -> tensor<64x64xf32, #mma1>
      %5 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %6 = tt.dot %cst_2, %5, %arg3 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
      scf.yield %4, %6 : tensor<64x64xf32, #mma1>, tensor<128x64xf32, #mma>
    }

    %7 = ttg.convert_layout %0#0 : tensor<64x64xf32, #mma1> -> tensor<64x64xf32, #blocked>
    %8 = ttg.convert_layout %0#1 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
    tt.store %out0, %7 : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.store %out1, %8 : tensor<128x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// WMMA: an operand loaded [N, K] (K contiguous) and fed to the dot through tt.trans is K-contiguous,
// so its LDS buffer is swizzled (it was unswizzled when the transpose was not taken into account).
// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
// CHECK-NOT: #ttg.swizzled_shared
// CHECK{LITERAL}: #smem = #ttg.shared_memory
// CHECK-LABEL: test_lds_layout_selection_wmma_trans

// CHECK: %[[ALLOC:.+]] = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>
// CHECK: scf.for
//  CHECK: %[[LOCAL_LOAD:.+]] = ttg.local_load {{.+}} -> tensor<64x128xbf16, #linear>
//  CHECK: %[[TRANS:.+]] = tt.trans %[[LOCAL_LOAD]] {{.+}} -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
//  CHECK: tt.dot {{.+}}, %[[TRANS]], {{.+}}

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_lds_layout_selection_wmma_trans(
    %arg0: tensor<64x128x!tt.ptr<bf16>, #blocked>,
    %out0 : tensor<128x64x!tt.ptr<f32>, #blocked>
  ) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_1 = arith.constant dense<0.581374812> : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32

    %0 = scf.for %arg1 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg2 = %cst_0) -> (tensor<128x64xf32, #mma>)  : i32 {
      %1 = tt.load %arg0 : tensor<64x128x!tt.ptr<bf16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #linear>
      %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<64x128xbf16, #linear> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %4 = tt.dot %cst_1, %3, %arg2 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
      scf.yield %4 : tensor<128x64xf32, #mma>
    }

    %5 = ttg.convert_layout %0 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
    tt.store %out0, %5 : tensor<128x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
