// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="arch-generation-name=gfx950" | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="arch-generation-name=gfx942" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
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

// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
// CHECK{LITERAL}: #shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK{LITERAL}: #smem = #ttg.shared_memory
// CHECK-LABEL: attn_bwd
// CHECK-DAG: %[[LOAD:.+]] = tt.load {{.*}} : tensor<64x16x!tt.ptr<f16>, #blocked1>
// CHECK-DAG: %[[ALLOC:.+]] = ttg.local_alloc %[[LOAD]] : (tensor<64x16xf16, #blocked1>) -> !ttg.memdesc<64x16xf16, #shared, #smem>
// CHECK-DAG: %[[LOCAL_LOAD_TRANS:.+]] = ttg.local_load %[[ALLOC]] : !ttg.memdesc<64x16xf16, #shared, #smem> -> tensor<64x16xf16, #linear>
// CHECK-DAG: %[[LOCAL_LOAD_DIRECT:.+]] = ttg.local_load %[[ALLOC]] : !ttg.memdesc<64x16xf16, #shared, #smem> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
// CHECK-DAG: tt.dot {{.+}}, %[[LOCAL_LOAD_DIRECT]], {{.+}}: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x16xf32, #mma1>
// CHECK-DAG: %[[TRANS:.+]] = tt.trans %[[LOCAL_LOAD_TRANS]] {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
// CHECK-DAG: tt.dot {{.+}}, %[[TRANS]], {{.+}} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>

// CHECK-DAG: %[[LOAD1:.+]] = tt.load {{.*}} : tensor<16x64x!tt.ptr<f16>, #blocked>
// CHECK-DAG: %[[ALLOC1:.+]] = ttg.local_alloc %[[LOAD1]] : (tensor<16x64xf16, #blocked>) -> !ttg.memdesc<16x64xf16, #shared1, #smem>
// CHECK-DAG: %[[LOCAL_LOAD_TRANS1:.+]] = ttg.local_load %[[ALLOC1]] : !ttg.memdesc<16x64xf16, #shared1, #smem> -> tensor<16x64xf16, #linear1>
// CHECK-DAG: %[[LOCAL_LOAD_DIRECT1:.+]] = ttg.local_load %[[ALLOC1]] : !ttg.memdesc<16x64xf16, #shared1, #smem> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
// CHECK-DAG: tt.dot {{.+}}, %[[LOCAL_LOAD_DIRECT1]], {{.+}}: tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
// CHECK-DAG: %[[TRANS1:.+]] = tt.trans %[[LOCAL_LOAD_TRANS1]] {order = array<i32: 1, 0>} : tensor<16x64xf16, #linear1> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>>
// CHECK-DAG: tt.dot {{.+}}, %[[TRANS1]], {{.+}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x16xf32, #mma1>

// CHECK-DAG: tt.load
// CHECK-DAG: ttg.local_alloc
// CHECK-DAG: ttg.local_load
// CHECK-DAG: ttg.local_load
// CHECK-DAG: tt.dot
// CHECK-DAG: tt.trans
// CHECK-DAG: tt.dot

// CHECK-DAG: tt.load
// CHECK-DAG: ttg.local_alloc
// CHECK-DAG: ttg.local_load
// CHECK-DAG: ttg.local_load
// CHECK-DAG: tt.dot
// CHECK-DAG: tt.trans
// CHECK-DAG: tt.dot

// CHECK-DAG: tt.load
// CHECK-DAG: ttg.local_alloc
// CHECK-DAG: ttg.local_load
// CHECK-DAG: ttg.local_load
// CHECK-DAG: tt.dot
// CHECK-DAG: tt.trans
// CHECK-DAG: tt.dot

// CHECK-DAG: tt.load
// CHECK-DAG: ttg.local_alloc
// CHECK-DAG: ttg.local_load
// CHECK-DAG: ttg.local_load
// CHECK-DAG: tt.dot
// CHECK-DAG: tt.trans
// CHECK-DAG: tt.dot

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [64, 0]], block = []}>
#loc1 = loc("Q")
#loc2 = loc("K")
#loc3 = loc("V")
#loc4 = loc("sm_scale")
#loc5 = loc("DO")
#loc6 = loc("DQ")
#loc7 = loc("DK")
#loc8 = loc("DV")
#loc9 = loc("M")
#loc10 = loc("D")
#loc11 = loc("stride_z")
#loc12 = loc("stride_h")
#loc13 = loc("stride_tok")
#loc14 = loc("H")
#loc15 = loc("N_CTX")
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @attn_bwd(%Q: !tt.ptr<f16> loc("Q"), %K: !tt.ptr<f16> loc("K"), %V: !tt.ptr<f16> loc("V"), %sm_scale: f32 loc("sm_scale"), %DO: !tt.ptr<f16> loc("DO"), %DQ: !tt.ptr<f16> loc("DQ"), %DK: !tt.ptr<f16> loc("DK"), %DV: !tt.ptr<f16> loc("DV"), %M: !tt.ptr<f32> loc("M"), %D: !tt.ptr<f32> loc("D"), %stride_z: i32 loc("stride_z"), %stride_h: i32 loc("stride_h"), %stride_tok: i32 loc("stride_tok"), %H: i32 loc("H"), %N_CTX: i32 loc("N_CTX")) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #mma> loc(#loc)
    %c128_i32 = arith.constant 128 : i32 loc(#loc)
    %c8_i32 = arith.constant 8 : i32 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c16_i32 = arith.constant 16 : i32 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1> loc(#loc)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma> loc(#loc)
    %cst_2 = arith.constant dense<0.693147182> : tensor<128x64xf32, #mma> loc(#loc)
    %0 = tt.get_program_id z : i32 loc(#loc)
    %1 = arith.muli %0, %N_CTX : i32 loc(#loc)
    %2 = arith.extsi %1 : i32 to i64 loc(#loc)
    %3 = arith.remsi %0, %H : i32 loc(#loc)
    %4 = arith.muli %stride_h, %3 : i32 loc(#loc)
    %5 = arith.divsi %0, %H : i32 loc(#loc)
    %6 = arith.muli %stride_z, %5 : i32 loc(#loc)
    %7 = arith.addi %4, %6 : i32 loc(#loc)
    %8 = arith.extsi %7 : i32 to i64 loc(#loc)
    %9 = tt.get_program_id x : i32 loc(#loc)
    %10 = tt.addptr %Q, %8 : !tt.ptr<f16>, i64 loc(#loc)
    %11 = tt.addptr %K, %8 : !tt.ptr<f16>, i64 loc(#loc)
    %12 = tt.addptr %V, %8 : !tt.ptr<f16>, i64 loc(#loc)
    %13 = tt.addptr %DO, %8 : !tt.ptr<f16>, i64 loc(#loc)
    %14 = tt.addptr %DQ, %8 : !tt.ptr<f16>, i64 loc(#loc)
    %15 = tt.addptr %DK, %8 : !tt.ptr<f16>, i64 loc(#loc)
    %16 = tt.addptr %DV, %8 : !tt.ptr<f16>, i64 loc(#loc)
    %17 = tt.addptr %M, %2 : !tt.ptr<f32>, i64 loc(#loc)
    %18 = tt.addptr %D, %2 : !tt.ptr<f32>, i64 loc(#loc)
    %19 = arith.muli %9, %c128_i32 : i32 loc(#loc)
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %24 = tt.splat %19 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %25 = tt.splat %19 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %26 = tt.splat %19 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %27 = tt.splat %19 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %28 = arith.addi %24, %20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %29 = arith.addi %25, %21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %30 = arith.addi %26, %22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %31 = arith.addi %27, %23 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %32 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc)
    %33 = tt.expand_dims %29 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xi32, #mma1> loc(#loc)
    %34 = tt.splat %stride_tok : i32 -> tensor<128x1xi32, #blocked> loc(#loc)
    %35 = arith.muli %32, %34 : tensor<128x1xi32, #blocked> loc(#loc)
    %36 = tt.splat %11 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %37 = tt.addptr %36, %35 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked> loc(#loc)
    %38 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %39 = tt.expand_dims %38 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked> loc(#loc)
    %40 = tt.broadcast %37 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %41 = tt.broadcast %39 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked> loc(#loc)
    %42 = tt.addptr %40, %41 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc)
    %43 = tt.load %42 : tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %44 = tt.splat %12 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %45 = tt.addptr %44, %35 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked> loc(#loc)
    %46 = tt.broadcast %45 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %47 = tt.addptr %46, %41 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc)
    %48 = tt.load %47 : tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %49 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %50 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
    %51 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %52 = tt.splat %19 : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %53 = tt.splat %19 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %54 = arith.addi %52, %49 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %55 = arith.addi %53, %51 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %56 = tt.expand_dims %54 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1> loc(#loc)
    %57 = tt.splat %stride_tok : i32 -> tensor<1x16xi32, #blocked1> loc(#loc)
    %58 = arith.muli %56, %57 : tensor<1x16xi32, #blocked1> loc(#loc)
    %59 = tt.splat %10 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked1> loc(#loc)
    %60 = tt.addptr %59, %58 : tensor<1x16x!tt.ptr<f16>, #blocked1>, tensor<1x16xi32, #blocked1> loc(#loc)
    %61 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %62 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %63 = tt.expand_dims %61 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1> loc(#loc)
    %64 = tt.expand_dims %62 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1> loc(#loc)
    %65 = tt.broadcast %60 : tensor<1x16x!tt.ptr<f16>, #blocked1> -> tensor<64x16x!tt.ptr<f16>, #blocked1> loc(#loc)
    %66 = tt.broadcast %63 : tensor<64x1xi32, #blocked1> -> tensor<64x16xi32, #blocked1> loc(#loc)
    %67 = tt.addptr %65, %66 : tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16xi32, #blocked1> loc(#loc)
    %68 = tt.expand_dims %55 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked> loc(#loc)
    %69 = tt.splat %stride_tok : i32 -> tensor<16x1xi32, #blocked> loc(#loc)
    %70 = arith.muli %68, %69 : tensor<16x1xi32, #blocked> loc(#loc)
    %71 = tt.splat %13 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %72 = tt.addptr %71, %70 : tensor<16x1x!tt.ptr<f16>, #blocked>, tensor<16x1xi32, #blocked> loc(#loc)
    %73 = tt.broadcast %72 : tensor<16x1x!tt.ptr<f16>, #blocked> -> tensor<16x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %74 = tt.broadcast %39 : tensor<1x64xi32, #blocked> -> tensor<16x64xi32, #blocked> loc(#loc)
    %75 = tt.addptr %73, %74 : tensor<16x64x!tt.ptr<f16>, #blocked>, tensor<16x64xi32, #blocked> loc(#loc)
    %76 = tt.splat %17 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
    %77 = tt.broadcast %33 : tensor<128x1xi32, #mma1> -> tensor<128x16xi32, #mma1> loc(#loc)
    %78 = tt.splat %18 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
    %79 = arith.muli %stride_tok, %c16_i32 : i32 loc(#loc)
    %80 = tt.splat %79 : i32 -> tensor<64x16xi32, #blocked1> loc(#loc)
    %81 = tt.splat %79 : i32 -> tensor<16x64xi32, #blocked> loc(#loc)
    %curr_m:5 = scf.for %curr_m_5 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %cst_1, %arg17 = %cst_1, %arg18 = %75, %arg19 = %19, %arg20 = %67) -> (tensor<128x64xf32, #mma>, tensor<128x64xf32, #mma>, tensor<16x64x!tt.ptr<f16>, #blocked>, i32, tensor<64x16x!tt.ptr<f16>, #blocked1>)  : i32 {
      %198 = tt.load %arg20 : tensor<64x16x!tt.ptr<f16>, #blocked1> loc(#loc)
      %199 = ttg.convert_layout %198 : tensor<64x16xf16, #blocked1> -> tensor<64x16xf16, #linear> loc(#loc)
      %200 = tt.splat %arg19 : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
      %201 = arith.addi %200, %50 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
      %202 = tt.addptr %76, %201 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma1}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
      %203 = tt.load %202 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
      %204 = ttg.convert_layout %43 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> loc(#loc)
      %205 = ttg.convert_layout %198 : tensor<64x16xf16, #blocked1> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> loc(#loc)
      %206 = tt.dot %204, %205, %cst_0 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x16xf32, #mma1> loc(#loc)
      %207 = tt.expand_dims %203 {axis = 0 : i32} : tensor<16xf32, #ttg.slice<{dim = 0, parent = #mma1}>> -> tensor<1x16xf32, #mma1> loc(#loc)
      %208 = tt.broadcast %207 : tensor<1x16xf32, #mma1> -> tensor<128x16xf32, #mma1> loc(#loc)
      %209 = arith.subf %206, %208 : tensor<128x16xf32, #mma1> loc(#loc)
      %210 = math.exp2 %209 : tensor<128x16xf32, #mma1> loc(#loc)
      %211 = tt.expand_dims %201 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> -> tensor<1x16xi32, #mma1> loc(#loc)
      %212 = tt.broadcast %211 : tensor<1x16xi32, #mma1> -> tensor<128x16xi32, #mma1> loc(#loc)
      %213 = arith.cmpi sge, %212, %77 : tensor<128x16xi32, #mma1> loc(#loc)
      %214 = arith.select %213, %210, %cst_0 : tensor<128x16xi1, #mma1>, tensor<128x16xf32, #mma1> loc(#loc)
      %215 = tt.load %arg18 : tensor<16x64x!tt.ptr<f16>, #blocked> loc(#loc)
      %216 = ttg.convert_layout %215 : tensor<16x64xf16, #blocked> -> tensor<16x64xf16, #linear1> loc(#loc)
      %217 = arith.truncf %214 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1> loc(#loc)
      %218 = ttg.convert_layout %217 : tensor<128x16xf16, #mma1> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %219 = ttg.convert_layout %215 : tensor<16x64xf16, #blocked> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %220 = tt.dot %218, %219, %arg17 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma> loc(#loc)
      %221 = tt.addptr %78, %201 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma1}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
      %222 = tt.load %221 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
      %223 = tt.trans %216 {order = array<i32: 1, 0>} : tensor<16x64xf16, #linear1> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> loc(#loc)
      %224 = ttg.convert_layout %48 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> loc(#loc)
      %225 = tt.dot %224, %223, %cst_0 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x16xf32, #mma1> loc(#loc)
      %226 = tt.expand_dims %222 {axis = 0 : i32} : tensor<16xf32, #ttg.slice<{dim = 0, parent = #mma1}>> -> tensor<1x16xf32, #mma1> loc(#loc)
      %227 = tt.broadcast %226 : tensor<1x16xf32, #mma1> -> tensor<128x16xf32, #mma1> loc(#loc)
      %228 = arith.subf %225, %227 : tensor<128x16xf32, #mma1> loc(#loc)
      %229 = arith.mulf %214, %228 : tensor<128x16xf32, #mma1> loc(#loc)
      %230 = arith.truncf %229 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1> loc(#loc)
      %231 = tt.trans %199 {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %232 = ttg.convert_layout %230 : tensor<128x16xf16, #mma1> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %233 = tt.dot %232, %231, %arg16 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma> loc(#loc)
      %234 = arith.addi %arg19, %c16_i32 : i32 loc(#loc)
      %235 = tt.addptr %arg20, %80 : tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16xi32, #blocked1> loc(#loc)
      %236 = tt.addptr %arg18, %81 : tensor<16x64x!tt.ptr<f16>, #blocked>, tensor<16x64xi32, #blocked> loc(#loc)
      scf.yield %233, %220, %236, %234, %235 : tensor<128x64xf32, #mma>, tensor<128x64xf32, #mma>, tensor<16x64x!tt.ptr<f16>, #blocked>, i32, tensor<64x16x!tt.ptr<f16>, #blocked1> loc(#loc)
    } loc(#loc29)
    %82 = arith.addi %19, %c128_i32 : i32 loc(#loc)
    %83 = arith.subi %N_CTX, %82 : i32 loc(#loc)
    %84 = arith.divsi %83, %c32_i32 : i32 loc(#loc)
    %85 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %86 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
    %87 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %88 = tt.splat %82 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %89 = tt.splat %82 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %90 = arith.addi %88, %85 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %91 = arith.addi %89, %87 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %92 = tt.expand_dims %90 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1> loc(#loc)
    %93 = tt.splat %stride_tok : i32 -> tensor<1x32xi32, #blocked1> loc(#loc)
    %94 = arith.muli %92, %93 : tensor<1x32xi32, #blocked1> loc(#loc)
    %95 = tt.splat %10 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked1> loc(#loc)
    %96 = tt.addptr %95, %94 : tensor<1x32x!tt.ptr<f16>, #blocked1>, tensor<1x32xi32, #blocked1> loc(#loc)
    %97 = tt.broadcast %96 : tensor<1x32x!tt.ptr<f16>, #blocked1> -> tensor<64x32x!tt.ptr<f16>, #blocked1> loc(#loc)
    %98 = tt.broadcast %64 : tensor<64x1xi32, #blocked1> -> tensor<64x32xi32, #blocked1> loc(#loc)
    %99 = tt.addptr %97, %98 : tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32xi32, #blocked1> loc(#loc)
    %100 = tt.expand_dims %91 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked> loc(#loc)
    %101 = tt.splat %stride_tok : i32 -> tensor<32x1xi32, #blocked> loc(#loc)
    %102 = arith.muli %100, %101 : tensor<32x1xi32, #blocked> loc(#loc)
    %103 = tt.splat %13 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %104 = tt.addptr %103, %102 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked> loc(#loc)
    %105 = tt.broadcast %104 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %106 = tt.broadcast %39 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked> loc(#loc)
    %107 = tt.addptr %105, %106 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked> loc(#loc)
    %108 = tt.splat %17 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
    %109 = tt.splat %18 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
    %110 = arith.muli %stride_tok, %c32_i32 : i32 loc(#loc)
    %111 = tt.splat %110 : i32 -> tensor<64x32xi32, #blocked1> loc(#loc)
    %112 = tt.splat %110 : i32 -> tensor<32x64xi32, #blocked> loc(#loc)
    %curr_m_3:5 = scf.for %curr_m_5 = %c0_i32 to %84 step %c1_i32 iter_args(%curr_m_6 = %curr_m#0, %curr_m_7 = %curr_m#1, %arg18 = %107, %arg19 = %82, %arg20 = %99) -> (tensor<128x64xf32, #mma>, tensor<128x64xf32, #mma>, tensor<32x64x!tt.ptr<f16>, #blocked>, i32, tensor<64x32x!tt.ptr<f16>, #blocked1>)  : i32 {
      %198 = tt.load %arg20 : tensor<64x32x!tt.ptr<f16>, #blocked1> loc(#loc)
      %199 = ttg.convert_layout %198 : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #linear2> loc(#loc)
      %200 = tt.splat %arg19 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
      %201 = arith.addi %200, %86 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
      %202 = tt.addptr %108, %201 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
      %203 = tt.load %202 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
      %204 = ttg.convert_layout %43 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %205 = ttg.convert_layout %198 : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %206 = tt.dot %204, %205, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma> loc(#loc)
      %207 = tt.expand_dims %203 {axis = 0 : i32} : tensor<32xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xf32, #mma> loc(#loc)
      %208 = tt.broadcast %207 : tensor<1x32xf32, #mma> -> tensor<128x32xf32, #mma> loc(#loc)
      %209 = arith.subf %206, %208 : tensor<128x32xf32, #mma> loc(#loc)
      %210 = math.exp2 %209 : tensor<128x32xf32, #mma> loc(#loc)
      %211 = tt.load %arg18 : tensor<32x64x!tt.ptr<f16>, #blocked> loc(#loc)
      %212 = ttg.convert_layout %211 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #linear3> loc(#loc)
      %213 = arith.truncf %210 : tensor<128x32xf32, #mma> to tensor<128x32xf16, #mma> loc(#loc)
      %214 = ttg.convert_layout %213 : tensor<128x32xf16, #mma> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %215 = ttg.convert_layout %211 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %216 = tt.dot %214, %215, %curr_m_7 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma> loc(#loc)
      %217 = tt.addptr %109, %201 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
      %218 = tt.load %217 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>> loc(#loc)
      %219 = tt.trans %212 {order = array<i32: 1, 0>} : tensor<32x64xf16, #linear3> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %220 = ttg.convert_layout %48 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %221 = tt.dot %220, %219, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma> loc(#loc)
      %222 = tt.expand_dims %218 {axis = 0 : i32} : tensor<32xf32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xf32, #mma> loc(#loc)
      %223 = tt.broadcast %222 : tensor<1x32xf32, #mma> -> tensor<128x32xf32, #mma> loc(#loc)
      %224 = arith.subf %221, %223 : tensor<128x32xf32, #mma> loc(#loc)
      %225 = arith.mulf %210, %224 : tensor<128x32xf32, #mma> loc(#loc)
      %226 = arith.truncf %225 : tensor<128x32xf32, #mma> to tensor<128x32xf16, #mma> loc(#loc)
      %227 = tt.trans %199 {order = array<i32: 1, 0>} : tensor<64x32xf16, #linear2> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %228 = ttg.convert_layout %226 : tensor<128x32xf16, #mma> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %229 = tt.dot %228, %227, %curr_m_6 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma> loc(#loc)
      %230 = arith.addi %arg19, %c32_i32 : i32 loc(#loc)
      %231 = tt.addptr %arg20, %111 : tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32xi32, #blocked1> loc(#loc)
      %232 = tt.addptr %arg18, %112 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked> loc(#loc)
      scf.yield %229, %216, %232, %230, %231 : tensor<128x64xf32, #mma>, tensor<128x64xf32, #mma>, tensor<32x64x!tt.ptr<f16>, #blocked>, i32, tensor<64x32x!tt.ptr<f16>, #blocked1> loc(#loc)
    } loc(#loc29)
    %113 = tt.splat %16 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %114 = tt.addptr %113, %35 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked> loc(#loc)
    %115 = tt.broadcast %114 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %116 = tt.addptr %115, %41 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc)
    %117 = arith.truncf %curr_m_3#1 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma> loc(#loc)
    %118 = ttg.convert_layout %116 : tensor<128x64x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #mma> loc(#loc)
    %119 = ttg.convert_layout %118 : tensor<128x64x!tt.ptr<f16>, #mma> -> tensor<128x64x!tt.ptr<f16>, #linear4> loc(#loc)
    %120 = ttg.convert_layout %117 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #linear4> loc(#loc)
    tt.store %119, %120 : tensor<128x64x!tt.ptr<f16>, #linear4> loc(#loc)
    %121 = tt.splat %sm_scale : f32 -> tensor<128x64xf32, #mma> loc(#loc)
    %122 = arith.mulf %curr_m_3#0, %121 : tensor<128x64xf32, #mma> loc(#loc)
    %123 = tt.splat %15 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %124 = tt.addptr %123, %35 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked> loc(#loc)
    %125 = tt.broadcast %124 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %126 = tt.addptr %125, %41 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc)
    %127 = arith.truncf %122 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma> loc(#loc)
    %128 = ttg.convert_layout %126 : tensor<128x64x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #mma> loc(#loc)
    %129 = ttg.convert_layout %128 : tensor<128x64x!tt.ptr<f16>, #mma> -> tensor<128x64x!tt.ptr<f16>, #linear4> loc(#loc)
    %130 = ttg.convert_layout %127 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #linear4> loc(#loc)
    tt.store %129, %130 : tensor<128x64x!tt.ptr<f16>, #linear4> loc(#loc)
    %131 = tt.splat %10 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %132 = tt.addptr %131, %35 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked> loc(#loc)
    %133 = tt.broadcast %132 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %134 = tt.addptr %133, %41 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc)
    %135 = tt.load %134 : tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %136 = tt.splat %13 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %137 = tt.addptr %136, %35 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked> loc(#loc)
    %138 = tt.broadcast %137 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %139 = tt.addptr %138, %41 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc)
    %140 = tt.load %139 : tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %141 = tt.splat %17 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %142 = tt.splat %17 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %143 = tt.addptr %141, %30 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %144 = tt.addptr %142, %31 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %145 = tt.load %143 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %146 = tt.load %144 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %147 = tt.expand_dims %145 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xf32, #mma1> loc(#loc)
    %148 = tt.expand_dims %146 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma> loc(#loc)
    %149 = tt.splat %11 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked1> loc(#loc)
    %150 = tt.addptr %149, %58 : tensor<1x16x!tt.ptr<f16>, #blocked1>, tensor<1x16xi32, #blocked1> loc(#loc)
    %151 = tt.broadcast %150 : tensor<1x16x!tt.ptr<f16>, #blocked1> -> tensor<64x16x!tt.ptr<f16>, #blocked1> loc(#loc)
    %152 = tt.addptr %151, %66 : tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16xi32, #blocked1> loc(#loc)
    %153 = tt.splat %12 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked1> loc(#loc)
    %154 = tt.addptr %153, %58 : tensor<1x16x!tt.ptr<f16>, #blocked1>, tensor<1x16xi32, #blocked1> loc(#loc)
    %155 = tt.broadcast %154 : tensor<1x16x!tt.ptr<f16>, #blocked1> -> tensor<64x16x!tt.ptr<f16>, #blocked1> loc(#loc)
    %156 = tt.addptr %155, %66 : tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16xi32, #blocked1> loc(#loc)
    %157 = tt.splat %18 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %158 = tt.splat %18 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %159 = tt.addptr %157, %30 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %160 = tt.addptr %158, %31 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %161 = tt.load %159 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma1}>> loc(#loc)
    %162 = tt.load %160 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma}>> loc(#loc)
    %163 = tt.broadcast %147 : tensor<128x1xf32, #mma1> -> tensor<128x16xf32, #mma1> loc(#loc)
    %164 = tt.broadcast %33 : tensor<128x1xi32, #mma1> -> tensor<128x16xi32, #mma1> loc(#loc)
    %165 = tt.expand_dims %161 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xf32, #mma1> loc(#loc)
    %166 = tt.broadcast %165 : tensor<128x1xf32, #mma1> -> tensor<128x16xf32, #mma1> loc(#loc)
    %167 = arith.muli %stride_tok, %c16_i32 : i32 loc(#loc)
    %168 = tt.splat %167 : i32 -> tensor<64x16xi32, #blocked1> loc(#loc)
    %curr_n:4 = scf.for %curr_n_5 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %cst_1, %arg17 = %19, %arg18 = %152, %arg19 = %156) -> (tensor<128x64xf32, #mma>, i32, tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16x!tt.ptr<f16>, #blocked1>)  : i32 {
      %198 = tt.load %arg18 : tensor<64x16x!tt.ptr<f16>, #blocked1> loc(#loc)
      %199 = ttg.convert_layout %198 : tensor<64x16xf16, #blocked1> -> tensor<64x16xf16, #linear> loc(#loc)
      %200 = tt.load %arg19 : tensor<64x16x!tt.ptr<f16>, #blocked1> loc(#loc)
      %201 = ttg.convert_layout %135 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> loc(#loc)
      %202 = ttg.convert_layout %198 : tensor<64x16xf16, #blocked1> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> loc(#loc)
      %203 = tt.dot %201, %202, %cst_0 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x16xf32, #mma1> loc(#loc)
      %204 = arith.subf %203, %163 : tensor<128x16xf32, #mma1> loc(#loc)
      %205 = math.exp2 %204 : tensor<128x16xf32, #mma1> loc(#loc)
      %206 = tt.splat %arg17 : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
      %207 = arith.addi %206, %50 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> loc(#loc)
      %208 = tt.expand_dims %207 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> -> tensor<1x16xi32, #mma1> loc(#loc)
      %209 = tt.broadcast %208 : tensor<1x16xi32, #mma1> -> tensor<128x16xi32, #mma1> loc(#loc)
      %210 = arith.cmpi sge, %164, %209 : tensor<128x16xi32, #mma1> loc(#loc)
      %211 = arith.select %210, %205, %cst_0 : tensor<128x16xi1, #mma1>, tensor<128x16xf32, #mma1> loc(#loc)
      %212 = ttg.convert_layout %140 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> loc(#loc)
      %213 = ttg.convert_layout %200 : tensor<64x16xf16, #blocked1> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> loc(#loc)
      %214 = tt.dot %212, %213, %cst_0 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 8}>> -> tensor<128x16xf32, #mma1> loc(#loc)
      %215 = arith.subf %214, %166 : tensor<128x16xf32, #mma1> loc(#loc)
      %216 = arith.mulf %211, %215 : tensor<128x16xf32, #mma1> loc(#loc)
      %217 = arith.truncf %216 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1> loc(#loc)
      %218 = tt.trans %199 {order = array<i32: 1, 0>} : tensor<64x16xf16, #linear> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %219 = ttg.convert_layout %217 : tensor<128x16xf16, #mma1> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %220 = tt.dot %219, %218, %arg16 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma> loc(#loc)
      %221 = arith.addi %arg17, %c16_i32 : i32 loc(#loc)
      %222 = tt.addptr %arg18, %168 : tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16xi32, #blocked1> loc(#loc)
      %223 = tt.addptr %arg19, %168 : tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16xi32, #blocked1> loc(#loc)
      scf.yield %220, %221, %222, %223 : tensor<128x64xf32, #mma>, i32, tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16x!tt.ptr<f16>, #blocked1> loc(#loc)
    } loc(#loc28)
    %169 = arith.divsi %19, %c32_i32 : i32 loc(#loc)
    %170 = arith.muli %169, %c32_i32 : i32 loc(#loc)
    %171 = arith.subi %19, %170 : i32 loc(#loc)
    %172 = tt.splat %171 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %173 = arith.addi %172, %85 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %174 = tt.expand_dims %173 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1> loc(#loc)
    %175 = arith.muli %174, %93 : tensor<1x32xi32, #blocked1> loc(#loc)
    %176 = tt.splat %11 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked1> loc(#loc)
    %177 = tt.addptr %176, %175 : tensor<1x32x!tt.ptr<f16>, #blocked1>, tensor<1x32xi32, #blocked1> loc(#loc)
    %178 = tt.broadcast %177 : tensor<1x32x!tt.ptr<f16>, #blocked1> -> tensor<64x32x!tt.ptr<f16>, #blocked1> loc(#loc)
    %179 = tt.addptr %178, %98 : tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32xi32, #blocked1> loc(#loc)
    %180 = tt.splat %12 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked1> loc(#loc)
    %181 = tt.addptr %180, %175 : tensor<1x32x!tt.ptr<f16>, #blocked1>, tensor<1x32xi32, #blocked1> loc(#loc)
    %182 = tt.broadcast %181 : tensor<1x32x!tt.ptr<f16>, #blocked1> -> tensor<64x32x!tt.ptr<f16>, #blocked1> loc(#loc)
    %183 = tt.addptr %182, %98 : tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32xi32, #blocked1> loc(#loc)
    %184 = tt.broadcast %148 : tensor<128x1xf32, #mma> -> tensor<128x32xf32, #mma> loc(#loc)
    %185 = tt.expand_dims %162 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma> loc(#loc)
    %186 = tt.broadcast %185 : tensor<128x1xf32, #mma> -> tensor<128x32xf32, #mma> loc(#loc)
    %187 = arith.muli %stride_tok, %c32_i32 : i32 loc(#loc)
    %188 = tt.splat %187 : i32 -> tensor<64x32xi32, #blocked1> loc(#loc)
    %curr_n_4:3 = scf.for %curr_n_5 = %c0_i32 to %169 step %c1_i32 iter_args(%curr_n_6 = %curr_n#0, %arg17 = %179, %arg18 = %183) -> (tensor<128x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32x!tt.ptr<f16>, #blocked1>)  : i32 {
      %198 = tt.load %arg17 : tensor<64x32x!tt.ptr<f16>, #blocked1> loc(#loc)
      %199 = ttg.convert_layout %198 : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #linear2> loc(#loc)
      %200 = tt.load %arg18 : tensor<64x32x!tt.ptr<f16>, #blocked1> loc(#loc)
      %201 = ttg.convert_layout %135 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %202 = ttg.convert_layout %198 : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %203 = tt.dot %201, %202, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma> loc(#loc)
      %204 = arith.subf %203, %184 : tensor<128x32xf32, #mma> loc(#loc)
      %205 = math.exp2 %204 : tensor<128x32xf32, #mma> loc(#loc)
      %206 = ttg.convert_layout %140 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %207 = ttg.convert_layout %200 : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %208 = tt.dot %206, %207, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x32xf32, #mma> loc(#loc)
      %209 = arith.subf %208, %186 : tensor<128x32xf32, #mma> loc(#loc)
      %210 = arith.mulf %205, %209 : tensor<128x32xf32, #mma> loc(#loc)
      %211 = arith.truncf %210 : tensor<128x32xf32, #mma> to tensor<128x32xf16, #mma> loc(#loc)
      %212 = tt.trans %199 {order = array<i32: 1, 0>} : tensor<64x32xf16, #linear2> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> loc(#loc)
      %213 = ttg.convert_layout %211 : tensor<128x32xf16, #mma> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> loc(#loc)
      %214 = tt.dot %213, %212, %curr_n_6 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma> loc(#loc)
      %215 = tt.addptr %arg17, %188 : tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32xi32, #blocked1> loc(#loc)
      %216 = tt.addptr %arg18, %188 : tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32xi32, #blocked1> loc(#loc)
      scf.yield %214, %215, %216 : tensor<128x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked1>, tensor<64x32x!tt.ptr<f16>, #blocked1> loc(#loc)
    } loc(#loc26)
    %189 = tt.splat %14 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked> loc(#loc)
    %190 = tt.addptr %189, %35 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked> loc(#loc)
    %191 = tt.broadcast %190 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked> loc(#loc)
    %192 = tt.addptr %191, %41 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked> loc(#loc)
    %193 = arith.mulf %curr_n_4#0, %cst_2 : tensor<128x64xf32, #mma> loc(#loc)
    %194 = arith.truncf %193 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma> loc(#loc)
    %195 = ttg.convert_layout %192 : tensor<128x64x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #mma> loc(#loc)
    %196 = ttg.convert_layout %195 : tensor<128x64x!tt.ptr<f16>, #mma> -> tensor<128x64x!tt.ptr<f16>, #linear4> loc(#loc)
    %197 = ttg.convert_layout %194 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #linear4> loc(#loc)
    tt.store %196, %197 : tensor<128x64x!tt.ptr<f16>, #linear4> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc16 = loc("dk")
#loc17 = loc("dq")
#loc18 = loc("dv"(#loc16))
#loc19 = loc("offs_n"(#loc17))
#loc20 = loc("kT_ptrs"(#loc17))
#loc21 = loc("offs_m"(#loc18))
#loc22 = loc("kT_ptrs"(#loc19))
#loc23 = loc("vT_ptrs"(#loc20))
#loc24 = loc("qT_ptrs"(#loc21))
#loc25 = loc("vT_ptrs"(#loc22))
#loc26 = loc("curr_n"(#loc23))
#loc27 = loc("do_ptrs"(#loc24))
#loc28 = loc("curr_n"(#loc25))
#loc29 = loc("curr_m"(#loc27))
