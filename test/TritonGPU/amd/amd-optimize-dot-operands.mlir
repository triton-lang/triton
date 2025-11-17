// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="arch-generation-name=gfx950" | FileCheck %s --check-prefixes GFX950

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
        %11 = amdg.scaled_upcast_fp8 %arg1 scale %10 : tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
        %12 = ttg.convert_layout %11 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xbf16, #blocked>
        %16 = arith.addf %15, %12 : tensor<128x128xbf16, #blocked>
        scf.yield %16 : tensor<128x128xbf16, #blocked>
      }
      tt.store %out, %14#0 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      tt.return
  }
}
