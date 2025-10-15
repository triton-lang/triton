// RUN: triton-opt %s -split-input-file --tritonamdgpu-implicit-convert-layout --tritongpu-remove-layout-conversions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 8], threadsPerWarp = [1, 4, 16, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 32, 2], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [4, 0], [8, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 4], [0, 8]], warp = [[1, 0], [2, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 2, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]], lane = [[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 0, 4], [0, 1, 0, 0]], warp = [[1, 0, 0, 0], [2, 0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0, 0], [0, 2, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0]], lane = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 4, 0, 0], [1, 0, 0, 0]], warp = [[0, 0, 1, 0], [0, 0, 2, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 64, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 4], [0, 0, 8]], warp = [[0, 16, 0], [0, 32, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 64]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 4, 0], [0, 8, 0]], warp = [[0, 0, 16], [0, 0, 32]], block = []}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg7: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg8: f32, %arg9: f32, %arg10: f32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<1x1x16x1xi32, #blocked>
    %cst_0 = arith.constant dense<8> : tensor<16x1xi32, #mma>
    %cst_1 = arith.constant dense<16> : tensor<16x1xi32, #linear>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #mma>
    %cst_4 = arith.constant dense<128> : tensor<1x128xi32, #blocked1>
    %cst_5 = arith.constant dense<1.44269502> : tensor<16x256xf32, #mma>
    %cst_6 = arith.constant dense<0xFF800000> : tensor<16x256xf32, #mma>
    %cst_7 = arith.constant dense<8> : tensor<16xi32, #blocked2>
    %c15_i32 = arith.constant 15 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_8 = arith.constant dense<0> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %cst_9 = arith.constant dense<0> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %cst_10 = arith.constant dense<8> : tensor<16x1xi32, #blocked1>
    %cst_11 = arith.constant dense<128> : tensor<1x128xi32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = tt.addptr %arg7, %0 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = arith.muli %2, %c256_i32 : i32
    %6 = arith.cmpi sge, %5, %4 : i32
    cf.cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %7 = arith.addi %5, %c256_i32 : i32
    %8 = arith.minsi %7, %4 : i32
    %9 = arith.subi %8, %5 : i32
    %10 = arith.addi %9, %c15_i32 : i32
    %11 = arith.divsi %10, %c16_i32 : i32
    %12 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %13 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %14 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked2>
    %15 = tt.splat %11 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %16 = tt.splat %11 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %17 = arith.cmpi slt, %12, %15 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %18 = arith.cmpi slt, %13, %16 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %19 = arith.select %17, %12, %cst_8 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %20 = arith.select %18, %13, %cst_9 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %21 = arith.muli %2, %c16_i32 : i32
    %22 = arith.muli %0, %arg27 : i32
    %23 = arith.addi %22, %21 : i32
    %24 = tt.splat %23 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %25 = tt.splat %23 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %26 = arith.addi %19, %24 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %27 = arith.addi %20, %25 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %28 = amdgpu.buffer_load %arg6[%26] : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %29 = amdgpu.buffer_load %arg6[%27] : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %30 = arith.muli %0, %arg18 : i32
    %31 = arith.muli %1, %c8_i32 : i32
    %32 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %33 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %34 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %35 = tt.expand_dims %32 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<16x1xi32, #mma>
    %36 = tt.expand_dims %33 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1>
    %37 = tt.expand_dims %34 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<16x1xi32, #linear>
    %38 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %39 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %40 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>>
    %41 = tt.expand_dims %38 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
    %42 = tt.expand_dims %39 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %43 = arith.cmpi slt, %35, %cst_0 : tensor<16x1xi32, #mma>
    %44 = arith.cmpi slt, %36, %cst_10 : tensor<16x1xi32, #blocked1>
    %45 = arith.cmpi slt, %41, %cst_11 : tensor<1x128xi32, #mma>
    %46 = arith.cmpi slt, %42, %cst_4 : tensor<1x128xi32, #blocked1>
    %47 = tt.broadcast %43 : tensor<16x1xi1, #mma> -> tensor<16x128xi1, #mma>
    %48 = tt.broadcast %44 : tensor<16x1xi1, #blocked1> -> tensor<16x128xi1, #blocked1>
    %49 = tt.broadcast %45 : tensor<1x128xi1, #mma> -> tensor<16x128xi1, #mma>
    %50 = tt.broadcast %46 : tensor<1x128xi1, #blocked1> -> tensor<16x128xi1, #blocked1>
    %51 = arith.andi %47, %49 : tensor<16x128xi1, #mma>
    %52 = arith.andi %48, %50 : tensor<16x128xi1, #blocked1>
    %53 = arith.muli %31, %arg19 : i32
    %54 = tt.splat %arg19 : i32 -> tensor<16x1xi32, #blocked1>
    %55 = arith.muli %36, %54 : tensor<16x1xi32, #blocked1>
    %56 = arith.addi %30, %53 : i32
    %57 = tt.broadcast %55 : tensor<16x1xi32, #blocked1> -> tensor<16x128xi32, #blocked1>
    %58 = tt.broadcast %42 : tensor<1x128xi32, #blocked1> -> tensor<16x128xi32, #blocked1>
    %59 = arith.addi %57, %58 : tensor<16x128xi32, #blocked1>
    %60 = tt.splat %56 : i32 -> tensor<16x128xi32, #blocked1>
    %61 = arith.addi %60, %59 : tensor<16x128xi32, #blocked1>
    %62 = amdgpu.buffer_load %arg3[%61], %52 : tensor<16x128xbf16, #blocked1>
    %63 = arith.extf %62 : tensor<16x128xbf16, #blocked1> to tensor<16x128xf32, #blocked1>
    %64 = tt.splat %arg8 : f32 -> tensor<16x128xf32, #blocked1>
    %65 = arith.mulf %63, %64 : tensor<16x128xf32, #blocked1>
    %66 = arith.truncf %65 : tensor<16x128xf32, #blocked1> to tensor<16x128xbf16, #blocked1>
    %67 = arith.muli %1, %arg21 : i32
    %68 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %69 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %70 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked3}>}>>
    %71 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>>
    %72 = tt.expand_dims %68 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x16xi32, #linear>
    %73 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>}>>
    %74 = tt.splat %21 : i32 -> tensor<16x1xi32, #linear>
    %75 = arith.addi %74, %37 : tensor<16x1xi32, #linear>
    %76 = arith.muli %75, %cst_1 : tensor<16x1xi32, #linear>
    %77 = tt.broadcast %76 : tensor<16x1xi32, #linear> -> tensor<16x16xi32, #linear>
    %78 = tt.broadcast %72 : tensor<1x16xi32, #linear> -> tensor<16x16xi32, #linear>
    %79 = arith.addi %77, %78 : tensor<16x16xi32, #linear>
    %80 = tt.expand_dims %29 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>> -> tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>>
    %81 = tt.expand_dims %80 {axis = 2 : i32} : tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>> -> tensor<16x1x1xi32, #ttg.slice<{dim = 3, parent = #blocked}>>
    %82 = tt.expand_dims %81 {axis = 3 : i32} : tensor<16x1x1xi32, #ttg.slice<{dim = 3, parent = #blocked}>> -> tensor<16x1x1x1xi32, #blocked>
    %83 = tt.splat %arg20 : i32 -> tensor<16x1x1x1xi32, #blocked>
    %84 = arith.muli %82, %83 : tensor<16x1x1x1xi32, #blocked>
    %85 = tt.broadcast %84 : tensor<16x1x1x1xi32, #blocked> -> tensor<16x16x1x1xi32, #blocked>
    %86 = tt.expand_dims %69 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>>
    %87 = tt.expand_dims %86 {axis = 2 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>> -> tensor<1x16x1xi32, #ttg.slice<{dim = 3, parent = #blocked}>>
    %88 = tt.expand_dims %87 {axis = 3 : i32} : tensor<1x16x1xi32, #ttg.slice<{dim = 3, parent = #blocked}>> -> tensor<1x16x1x1xi32, #blocked>
    %89 = tt.splat %arg22 : i32 -> tensor<1x16x1x1xi32, #blocked>
    %90 = arith.muli %88, %89 : tensor<1x16x1x1xi32, #blocked>
    %91 = tt.broadcast %90 : tensor<1x16x1x1xi32, #blocked> -> tensor<16x16x1x1xi32, #blocked>
    %92 = arith.addi %85, %91 : tensor<16x16x1x1xi32, #blocked>
    %93 = tt.broadcast %92 : tensor<16x16x1x1xi32, #blocked> -> tensor<16x16x16x1xi32, #blocked>
    %94 = tt.expand_dims %71 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>>
    %95 = tt.expand_dims %94 {axis = 1 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #blocked}>}>> -> tensor<1x1x16xi32, #ttg.slice<{dim = 3, parent = #blocked}>>
    %96 = tt.expand_dims %95 {axis = 3 : i32} : tensor<1x1x16xi32, #ttg.slice<{dim = 3, parent = #blocked}>> -> tensor<1x1x16x1xi32, #blocked>
    %97 = arith.muli %96, %cst : tensor<1x1x16x1xi32, #blocked>
    %98 = tt.broadcast %97 : tensor<1x1x16x1xi32, #blocked> -> tensor<16x16x16x1xi32, #blocked>
    %99 = arith.addi %93, %98 : tensor<16x16x16x1xi32, #blocked>
    %100 = tt.broadcast %99 : tensor<16x16x16x1xi32, #blocked> -> tensor<16x16x16x8xi32, #blocked>
    %101 = tt.expand_dims %73 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>}>> -> tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %102 = tt.expand_dims %101 {axis = 1 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x1x8xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %103 = tt.expand_dims %102 {axis = 2 : i32} : tensor<1x1x8xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x1x1x8xi32, #blocked>
    %104 = tt.broadcast %103 : tensor<1x1x1x8xi32, #blocked> -> tensor<16x16x16x8xi32, #blocked>
    %105 = arith.addi %100, %104 : tensor<16x16x16x8xi32, #blocked>
    %106 = tt.splat %67 : i32 -> tensor<16x16x16x8xi32, #blocked>
    %107 = arith.addi %106, %105 : tensor<16x16x16x8xi32, #blocked>
    %108 = amdgpu.buffer_load %arg4[%107] : tensor<16x16x16x8xbf16, #blocked>
    %109 = ttg.convert_layout %108 : tensor<16x16x16x8xbf16, #blocked> -> tensor<16x16x16x8xbf16, #linear1>
    %110 = tt.trans %109 {order = array<i32: 1, 3, 0, 2>} : tensor<16x16x16x8xbf16, #linear1> -> tensor<16x8x16x16xbf16, #linear2>
    %111 = tt.reshape %110 : tensor<16x8x16x16xbf16, #linear2> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %112 = ttg.local_alloc %66 : (tensor<16x128xbf16, #blocked1>) -> !ttg.memdesc<16x128xbf16, #shared, #smem>
    %113 = ttg.local_load %112 : !ttg.memdesc<16x128xbf16, #shared, #smem> -> tensor<16x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %114 = tt.dot %113, %111, %cst_2, inputPrecision = tf32 : tensor<16x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x256xf32, #mma>
    %115 = tt.reshape %79 : tensor<16x16xi32, #linear> -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %116 = tt.expand_dims %115 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi32, #mma>
    %117 = tt.splat %4 : i32 -> tensor<1x256xi32, #mma>
    %118 = arith.cmpi slt, %116, %117 : tensor<1x256xi32, #mma>
    %119 = tt.broadcast %43 : tensor<16x1xi1, #mma> -> tensor<16x256xi1, #mma>
    %120 = tt.broadcast %118 : tensor<1x256xi1, #mma> -> tensor<16x256xi1, #mma>
    %121 = arith.andi %119, %120 : tensor<16x256xi1, #mma>
    %122 = arith.select %121, %114, %cst_6 : tensor<16x256xi1, #mma>, tensor<16x256xf32, #mma>
    %123 = "tt.reduce"(%122) <{axis = 1 : i32}> ({
    ^bb0(%arg28: f32, %arg29: f32):
      %185 = arith.maxnumf %arg28, %arg29 : f32
      tt.reduce.return %185 : f32
    }) : (tensor<16x256xf32, #mma>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %124 = tt.expand_dims %123 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<16x1xf32, #mma>
    %125 = tt.broadcast %124 : tensor<16x1xf32, #mma> -> tensor<16x256xf32, #mma>
    %126 = arith.subf %122, %125 : tensor<16x256xf32, #mma>
    %127 = arith.mulf %126, %cst_5 : tensor<16x256xf32, #mma>
    %128 = math.exp2 %127 : tensor<16x256xf32, #mma>
    %129 = arith.truncf %128 : tensor<16x256xf32, #mma> to tensor<16x256xbf16, #mma>
    %130 = "tt.reduce"(%129) <{axis = 1 : i32}> ({
    ^bb0(%arg28: bf16, %arg29: bf16):
      %185 = arith.addf %arg28, %arg29 : bf16
      tt.reduce.return %185 : bf16
    }) : (tensor<16x256xbf16, #mma>) -> tensor<16xbf16, #ttg.slice<{dim = 1, parent = #mma}>>
    %131 = arith.muli %1, %arg25 : i32
    %132 = tt.expand_dims %28 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>> -> tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #blocked3}>>
    %133 = tt.expand_dims %132 {axis = 2 : i32} : tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<16x1x1xi32, #blocked3>
    %134 = tt.splat %arg24 : i32 -> tensor<16x1x1xi32, #blocked3>
    %135 = arith.muli %133, %134 : tensor<16x1x1xi32, #blocked3>
    %136 = tt.broadcast %135 : tensor<16x1x1xi32, #blocked3> -> tensor<16x128x1xi32, #blocked3>
    %137 = tt.expand_dims %40 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked3}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 2, parent = #blocked3}>>
    %138 = tt.expand_dims %137 {axis = 2 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<1x128x1xi32, #blocked3>
    %139 = tt.splat %arg26 : i32 -> tensor<1x128x1xi32, #blocked3>
    %140 = arith.muli %138, %139 : tensor<1x128x1xi32, #blocked3>
    %141 = tt.broadcast %140 : tensor<1x128x1xi32, #blocked3> -> tensor<16x128x1xi32, #blocked3>
    %142 = arith.addi %136, %141 : tensor<16x128x1xi32, #blocked3>
    %143 = tt.broadcast %142 : tensor<16x128x1xi32, #blocked3> -> tensor<16x128x16xi32, #blocked3>
    %144 = tt.expand_dims %70 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked3}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %145 = tt.expand_dims %144 {axis = 1 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<1x1x16xi32, #blocked3>
    %146 = tt.broadcast %145 : tensor<1x1x16xi32, #blocked3> -> tensor<16x128x16xi32, #blocked3>
    %147 = arith.addi %143, %146 : tensor<16x128x16xi32, #blocked3>
    %148 = tt.splat %131 : i32 -> tensor<16x128x16xi32, #blocked3>
    %149 = arith.addi %148, %147 : tensor<16x128x16xi32, #blocked3>
    %150 = amdgpu.buffer_load %arg5[%149] : tensor<16x128x16xbf16, #blocked3>
    %151 = ttg.convert_layout %150 : tensor<16x128x16xbf16, #blocked3> -> tensor<16x128x16xbf16, #linear3>
    %152 = tt.trans %151 {order = array<i32: 0, 2, 1>} : tensor<16x128x16xbf16, #linear3> -> tensor<16x16x128xbf16, #linear4>
    %153 = tt.reshape %152 : tensor<16x16x128xbf16, #linear4> -> tensor<256x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %154 = arith.muli %0, %arg11 : i32
    %155 = arith.muli %1, %arg12 : i32
    %156 = arith.addi %154, %155 : i32
    %157 = arith.muli %2, %arg13 : i32
    %158 = arith.addi %156, %157 : i32
    %159 = tt.splat %158 : i32 -> tensor<16xi32, #blocked2>
    %160 = arith.cmpi slt, %14, %cst_7 : tensor<16xi32, #blocked2>
    %161 = arith.addi %159, %14 : tensor<16xi32, #blocked2>
    %162 = ttg.convert_layout %123 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<16xf32, #blocked2>
    amdgpu.buffer_store %162, %arg1[%161], %160 : tensor<16xf32, #blocked2>
    %163 = ttg.convert_layout %130 : tensor<16xbf16, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<16xbf16, #blocked2>
    %164 = arith.extf %163 : tensor<16xbf16, #blocked2> to tensor<16xf32, #blocked2>
    amdgpu.buffer_store %164, %arg0[%161], %160 : tensor<16xf32, #blocked2>
    %165 = ttg.local_alloc %129 : (tensor<16x256xbf16, #mma>) -> !ttg.memdesc<16x256xbf16, #shared1, #smem>
    %166 = ttg.local_load %165 : !ttg.memdesc<16x256xbf16, #shared1, #smem> -> tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %167 = tt.dot %166, %153, %cst_3, inputPrecision = tf32 : tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<256x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x128xf32, #mma>
    %168 = tt.expand_dims %130 {axis = 1 : i32} : tensor<16xbf16, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<16x1xbf16, #mma>
    %169 = arith.extf %168 : tensor<16x1xbf16, #mma> to tensor<16x1xf32, #mma>
    %170 = tt.broadcast %169 : tensor<16x1xf32, #mma> -> tensor<16x128xf32, #mma>
    %171 = arith.divf %167, %170 : tensor<16x128xf32, #mma>
    %172 = arith.muli %0, %arg14 : i32
    %173 = arith.muli %1, %arg15 : i32
    %174 = arith.addi %172, %173 : i32
    %175 = arith.muli %2, %arg16 : i32
    %176 = tt.splat %arg17 : i32 -> tensor<16x1xi32, #mma>
    %177 = arith.muli %35, %176 : tensor<16x1xi32, #mma>
    %178 = tt.broadcast %177 : tensor<16x1xi32, #mma> -> tensor<16x128xi32, #mma>
    %179 = tt.broadcast %41 : tensor<1x128xi32, #mma> -> tensor<16x128xi32, #mma>
    %180 = arith.addi %178, %179 : tensor<16x128xi32, #mma>
    %181 = arith.addi %174, %175 : i32
    %182 = tt.splat %181 : i32 -> tensor<16x128xi32, #mma>
    %183 = arith.addi %182, %180 : tensor<16x128xi32, #mma>
    %184 = arith.truncf %171 : tensor<16x128xf32, #mma> to tensor<16x128xbf16, #mma>
    amdgpu.buffer_store %184, %arg2[%183], %51 : tensor<16x128xbf16, #mma>
    tt.return
  }
}
