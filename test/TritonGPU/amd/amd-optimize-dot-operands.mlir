// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="arch-generation-name=gfx950" | FileCheck %s --check-prefixes CHECK,GFX950
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [1, 0, 0], [2, 0, 0], [0, 32, 0], [0, 64, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 8], [0, 0, 16]], warp = [[0, 16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]], warp = [[16, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [1, 0, 0], [2, 0, 0], [0, 0, 32], [0, 0, 64]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 8, 0], [0, 16, 0]], warp = [[0, 0, 16]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 2], instrShape = [16, 16], isTransposed = true}>
// GFX950{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// GFX950-LABEL: test_alloc_shared_mem_for_scaled_upcast
// GFX950: %[[LOAD:.+]] = tt.load
// GFX950: %[[ALLOC:.+]] = ttg.local_alloc %[[LOAD]] : (tensor<128x4xi8, #blocked>) -> !ttg.memdesc<128x4xi8, #shared, #smem>
// GFX950: %[[LOCAL_LOAD:.+]] = ttg.local_load %[[ALLOC]] : !ttg.memdesc<128x4xi8, #shared, #smem> -> tensor<128x4xi8, #linear1>
// GFX950: tt.trans %[[LOCAL_LOAD]] {order = array<i32: 1, 0>}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_alloc_shared_mem_for_scaled_upcast(
    %arg0: tensor<128x4x!tt.ptr<i8>, #blocked>,
    %arg1: tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
    %out: tensor<128x128x!tt.ptr<bf16>, #blocked>,
    %K: i32 {tt.divisibility = 16 : i32}
  ) {
      %c0_i32 = arith.constant 0 : i32
      %c128_i32 = arith.constant 128 : i32
      %cst_0 = arith.constant dense<7> : tensor<4x128xi16, #ttg.slice<{dim = 2, parent = #linear}>>
      %cst_1 = arith.constant dense<0.0> : tensor<128x128xbf16, #blocked>

      %14:1 = scf.for %13 = %c0_i32 to %K step %c128_i32 iter_args(%15 = %cst_1) -> (tensor<128x128xbf16, #blocked>) : i32 {
        %1 = tt.load %arg0 : tensor<128x4x!tt.ptr<i8>, #blocked>
        %2 = ttg.convert_layout %1 : tensor<128x4xi8, #blocked> -> tensor<128x4xi8, #linear1>
        %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<128x4xi8, #linear1> -> tensor<4x128xi8, #ttg.slice<{dim = 2, parent = #linear}>>
        %4 = arith.extui %3 : tensor<4x128xi8, #ttg.slice<{dim = 2, parent = #linear}>> to tensor<4x128xi16, #ttg.slice<{dim = 2, parent = #linear}>>
        %5 = arith.shli %4, %cst_0 : tensor<4x128xi16, #ttg.slice<{dim = 2, parent = #linear}>>
        %6 = tt.bitcast %5 : tensor<4x128xi16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x128xbf16, #ttg.slice<{dim = 2, parent = #linear}>>
        %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<4x128xbf16, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x128x1xbf16, #linear>
        %8 = tt.broadcast %7 : tensor<4x128x1xbf16, #linear> -> tensor<4x128x32xbf16, #linear>
        %9 = tt.trans %8 {order = array<i32: 0, 2, 1>} : tensor<4x128x32xbf16, #linear> -> tensor<4x32x128xbf16, #linear2>
        %10 = tt.reshape %9 : tensor<4x32x128xbf16, #linear2> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
        %11 = amdgpu.scaled_upcast_fp8 %arg1 scale %10 : tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
        %12 = ttg.convert_layout %11 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xbf16, #blocked>
        %16 = arith.addf %15, %12 : tensor<128x128xbf16, #blocked>
        scf.yield %16 : tensor<128x128xbf16, #blocked>
      }
      tt.store %out, %14#0 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      tt.return
  }
}
