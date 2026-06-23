// RUN: triton-opt %s --mlir-disable-threading --pass-pipeline='builtin.module(tritonamdgpu-update-async-wait-count{gfx-arch=gfx950}, convert-warp-pipeline{gfx-arch=gfx950}, convert-scf-to-cf{allow-pattern-rollback=true}, gluon-inline, convert-index-to-llvm{index-bitwidth=0}, allocate-amdgpu-shared-memory{arch=gfx950}, tritongpu-global-scratch-memory-allocation, tritongpu-global-scratch-memory-allocation, convert-triton-amdgpu-to-llvm{ftz=true gfx-arch=gfx950}, triton-amdgpu-convert-warp-specialize-to-llvm{gfx-arch=gfx950}, canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, convert-cf-to-llvm{index-bitwidth=0}, convert-arith-to-llvm{index-bitwidth=0}, canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce, enable-line-info, convert-builtin-func-to-llvm{ftz=true gfx-arch=gfx950})' | FileCheck %s

// Regression test for split-soffset uniformity fallback on values created
// during conversion. This is reduced from AITER's MoE routing _topk kernel.
// The store value depends on a reduction containing the split-safe buffer
// load; lowering that store used to recursively query uniformity for a
// detached block argument and crash in Block::isEntryBlock().

// CHECK-LABEL: llvm.func @_topk

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 4, 4], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 2, 2, 2, 2, 2, 2], warpsPerCTA = [8, 1, 1, 1, 1, 1, 1], order = [6, 5, 4, 3, 2, 1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1], threadsPerWarp = [2, 2, 2, 2, 2, 2], warpsPerCTA = [8, 1, 1, 1, 1, 1], order = [5, 4, 3, 2, 1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [4, 2, 2, 2, 2], warpsPerCTA = [8, 1, 1, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [8, 2, 2, 2], warpsPerCTA = [8, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [16, 2, 2], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [], lane = [[0], [0], [1], [0], [0], [0]], warp = [[0], [0], [0]], block = []}>
#linear1 = #ttg.linear<{register = [], lane = [[0], [1], [0], [0], [0], [0]], warp = [[0], [0], [0]], block = []}>
#linear2 = #ttg.linear<{register = [], lane = [[1], [0], [0], [0], [0], [0]], warp = [[0], [0], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_topk(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<i16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: i32, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg10: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0> : tensor<1x4xi32, #blocked>
    %cst_0 = arith.constant dense<2.500000e+00> : tensor<1x4xf32, #blocked>
    %cst_1 = arith.constant dense<32> : tensor<1x4xi32, #blocked>
    %cst_2 = arith.constant dense<1> : tensor<1x4xi32, #blocked>
    %cst_3 = arith.constant dense<0> : tensor<1x4x4xi32, #blocked1>
    %cst_4 = arith.constant dense<0> : tensor<128xi32, #blocked2>
    %cst_5 = arith.constant dense<-1> : tensor<1x4xi32, #blocked>
    %cst_6 = arith.constant dense<-2147483648> : tensor<1x4xi32, #blocked>
    %cst_7 = arith.constant dense<1> : tensor<1x2xi32, #blocked3>
    %cst_8 = arith.constant dense<-1> : tensor<1x128xi32, #blocked4>
    %cst_9 = arith.constant dense<-2147483648> : tensor<1x128xi32, #blocked4>
    %cst_10 = arith.constant dense<0> : tensor<1x128xi32, #blocked4>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<1x128xf32, #blocked4>
    %cst_12 = arith.constant dense<16> : tensor<1x4xi64, #blocked>
    %cst_13 = arith.constant dense<48> : tensor<1x4xi64, #blocked>
    %cst_14 = arith.constant dense<16> : tensor<1x128xi64, #blocked4>
    %cst_15 = arith.constant dense<0xFF800000> : tensor<1x128xf32, #blocked4>
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<1x128xf32, #blocked4>
    %cst_17 = arith.constant dense<1> : tensor<2x1xi32, #blocked3>
    %cst_18 = arith.constant dense<9.99999968E-21> : tensor<1x1xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.cmpi slt, %0, %c1_i32 : i32
    scf.if %1 {
      %298 = arith.muli %0, %c128_i32 : i32
      %299 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
      %300 = tt.splat %298 : i32 -> tensor<128xi32, #blocked2>
      %301 = arith.addi %299, %300 : tensor<128xi32, #blocked2>
      amdg.buffer_store %cst_4, %arg9[%301] {amdgpu.split_soffset_safe} : tensor<128xi32, #blocked2>
    } else {
      %298 = arith.addi %arg11, %c1_i32 : i32
      %299 = arith.cmpi slt, %0, %298 : i32
      scf.if %299 {
        %300 = arith.subi %0, %c1_i32 : i32
        %301 = arith.muli %300, %c128_i32 : i32
        %302 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
        %303 = tt.splat %301 : i32 -> tensor<128xi32, #blocked2>
        %304 = arith.addi %303, %302 : tensor<128xi32, #blocked2>
        %305 = tt.splat %arg12 : i32 -> tensor<128xi32, #blocked2>
        %306 = arith.cmpi slt, %304, %305 : tensor<128xi32, #blocked2>
        amdg.buffer_store %cst_4, %arg10[%304], %306 : tensor<128xi32, #blocked2>
      }
    }
    %2 = arith.cmpi sge, %0, %arg7 : i32
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %3 = arith.cmpi slt, %0, %arg7 : i32
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi32, #blocked4>
    %6 = tt.splat %arg8 : i32 -> tensor<1x128xi32, #blocked4>
    %7 = arith.cmpi slt, %5, %6 : tensor<1x128xi32, #blocked4>
    %8 = arith.muli %0, %arg1 : i32
    %9 = tt.splat %8 : i32 -> tensor<1x128xi32, #blocked4>
    %10 = arith.addi %5, %9 : tensor<1x128xi32, #blocked4>
    %11 = tt.splat %3 : i1 -> tensor<1x128xi1, #blocked4>
    %12 = arith.andi %11, %7 : tensor<1x128xi1, #blocked4>
    %13 = amdg.buffer_load %arg0[%10], %12 : tensor<1x128xf32, #blocked4>
    %14 = arith.maxnumf %13, %cst_16 : tensor<1x128xf32, #blocked4>
    %15 = math.absf %13 : tensor<1x128xf32, #blocked4>
    %16 = arith.negf %15 : tensor<1x128xf32, #blocked4>
    %17 = math.exp %16 : tensor<1x128xf32, #blocked4>
    %18 = arith.addf %17, %cst_11 : tensor<1x128xf32, #blocked4>
    %19 = math.log %18 : tensor<1x128xf32, #blocked4>
    %20 = arith.addf %14, %19 : tensor<1x128xf32, #blocked4>
    %21 = math.sqrt %20 : tensor<1x128xf32, #blocked4>
    %22 = tt.splat %arg8 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %23 = arith.cmpi slt, %4, %22 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %24 = amdg.buffer_load %arg13[%4], %23 {amdgpu.split_soffset_safe} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %25 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xf32, #blocked4>
    %26 = arith.addf %21, %25 : tensor<1x128xf32, #blocked4>
    %27 = arith.select %12, %26, %cst_15 : tensor<1x128xi1, #blocked4>, tensor<1x128xf32, #blocked4>
    %28 = tt.bitcast %27 : tensor<1x128xf32, #blocked4> -> tensor<1x128xi32, #blocked4>
    %29 = arith.andi %28, %cst_9 : tensor<1x128xi32, #blocked4>
    %30 = arith.cmpi ne, %29, %cst_10 : tensor<1x128xi32, #blocked4>
    %31 = arith.select %30, %cst_8, %cst_9 : tensor<1x128xi1, #blocked4>, tensor<1x128xi32, #blocked4>
    %32 = arith.xori %28, %31 : tensor<1x128xi32, #blocked4>
    %33 = arith.extui %32 : tensor<1x128xi32, #blocked4> to tensor<1x128xi64, #blocked4>
    %34 = arith.shli %33, %cst_14 : tensor<1x128xi64, #blocked4>
    %35 = arith.extsi %5 : tensor<1x128xi32, #blocked4> to tensor<1x128xi64, #blocked4>
    %36 = arith.ori %34, %35 : tensor<1x128xi64, #blocked4>
    %37 = tt.reshape %36 : tensor<1x128xi64, #blocked4> -> tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %38 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear>
    %39 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear1>
    %40 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear2>
    %41 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #blocked2>
    %42 = tt.reshape %39 : tensor<2xi32, #linear1> -> tensor<1x1x1x1x1x2x1xi32, #blocked5>
    %43 = "tt.reduce"(%37) <{axis = 6 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2x2x2xi64, #blocked5>) -> tensor<2x2x2x2x2x2xi64, #ttg.slice<{dim = 6, parent = #blocked5}>>
    %44 = tt.expand_dims %43 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi64, #ttg.slice<{dim = 6, parent = #blocked5}>> -> tensor<2x2x2x2x2x2x1xi64, #blocked5>
    %45 = tt.broadcast %44 : tensor<2x2x2x2x2x2x1xi64, #blocked5> -> tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %46 = arith.xori %37, %45 : tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %47 = tt.reshape %40 : tensor<2xi32, #linear2> -> tensor<1x1x1x1x1x1x2xi32, #blocked5>
    %48 = arith.cmpi ugt, %37, %46 : tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %49 = tt.broadcast %42 : tensor<1x1x1x1x1x2x1xi32, #blocked5> -> tensor<1x1x1x1x1x2x2xi32, #blocked5>
    %50 = tt.broadcast %47 : tensor<1x1x1x1x1x1x2xi32, #blocked5> -> tensor<1x1x1x1x1x2x2xi32, #blocked5>
    %51 = arith.xori %49, %50 : tensor<1x1x1x1x1x2x2xi32, #blocked5>
    %52 = arith.extui %48 : tensor<2x2x2x2x2x2x2xi1, #blocked5> to tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %53 = tt.broadcast %51 : tensor<1x1x1x1x1x2x2xi32, #blocked5> -> tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %54 = arith.cmpi ne, %52, %53 : tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %55 = arith.select %54, %46, %37 : tensor<2x2x2x2x2x2x2xi1, #blocked5>, tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %56 = tt.reshape %38 : tensor<2xi32, #linear> -> tensor<1x1x1x1x2x1x1xi32, #blocked5>
    %57 = "tt.reduce"(%55) <{axis = 5 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2x2x2xi64, #blocked5>) -> tensor<2x2x2x2x2x2xi64, #ttg.slice<{dim = 5, parent = #blocked5}>>
    %58 = tt.expand_dims %57 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi64, #ttg.slice<{dim = 5, parent = #blocked5}>> -> tensor<2x2x2x2x2x1x2xi64, #blocked5>
    %59 = tt.broadcast %58 : tensor<2x2x2x2x2x1x2xi64, #blocked5> -> tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %60 = arith.xori %55, %59 : tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %61 = arith.cmpi ugt, %55, %60 : tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %62 = tt.broadcast %56 : tensor<1x1x1x1x2x1x1xi32, #blocked5> -> tensor<1x1x1x1x2x2x1xi32, #blocked5>
    %63 = tt.broadcast %42 : tensor<1x1x1x1x1x2x1xi32, #blocked5> -> tensor<1x1x1x1x2x2x1xi32, #blocked5>
    %64 = arith.xori %62, %63 : tensor<1x1x1x1x2x2x1xi32, #blocked5>
    %65 = arith.extui %61 : tensor<2x2x2x2x2x2x2xi1, #blocked5> to tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %66 = tt.broadcast %64 : tensor<1x1x1x1x2x2x1xi32, #blocked5> -> tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %67 = arith.cmpi ne, %65, %66 : tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %68 = arith.select %67, %60, %55 : tensor<2x2x2x2x2x2x2xi1, #blocked5>, tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %69 = "tt.reduce"(%68) <{axis = 6 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2x2x2xi64, #blocked5>) -> tensor<2x2x2x2x2x2xi64, #ttg.slice<{dim = 6, parent = #blocked5}>>
    %70 = tt.expand_dims %69 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi64, #ttg.slice<{dim = 6, parent = #blocked5}>> -> tensor<2x2x2x2x2x2x1xi64, #blocked5>
    %71 = tt.broadcast %70 : tensor<2x2x2x2x2x2x1xi64, #blocked5> -> tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %72 = arith.xori %68, %71 : tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %73 = arith.cmpi ugt, %68, %72 : tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %74 = tt.broadcast %56 : tensor<1x1x1x1x2x1x1xi32, #blocked5> -> tensor<1x1x1x1x2x1x2xi32, #blocked5>
    %75 = tt.broadcast %47 : tensor<1x1x1x1x1x1x2xi32, #blocked5> -> tensor<1x1x1x1x2x1x2xi32, #blocked5>
    %76 = arith.xori %74, %75 : tensor<1x1x1x1x2x1x2xi32, #blocked5>
    %77 = arith.extui %73 : tensor<2x2x2x2x2x2x2xi1, #blocked5> to tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %78 = tt.broadcast %76 : tensor<1x1x1x1x2x1x2xi32, #blocked5> -> tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %79 = arith.cmpi ne, %77, %78 : tensor<2x2x2x2x2x2x2xi32, #blocked5>
    %80 = arith.select %79, %72, %68 : tensor<2x2x2x2x2x2x2xi1, #blocked5>, tensor<2x2x2x2x2x2x2xi64, #blocked5>
    %81 = "tt.reduce"(%80) <{axis = 4 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.maxui %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2x2x2xi64, #blocked5>) -> tensor<2x2x2x2x2x2xi64, #ttg.slice<{dim = 4, parent = #blocked5}>>
    %82 = ttg.convert_layout %81 : tensor<2x2x2x2x2x2xi64, #ttg.slice<{dim = 4, parent = #blocked5}>> -> tensor<2x2x2x2x2x2xi64, #blocked6>
    %83 = tt.reshape %38 : tensor<2xi32, #linear> -> tensor<1x1x1x2x1x1xi32, #blocked6>
    %84 = "tt.reduce"(%82) <{axis = 4 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2x2xi64, #blocked6>) -> tensor<2x2x2x2x2xi64, #ttg.slice<{dim = 4, parent = #blocked6}>>
    %85 = tt.expand_dims %84 {axis = 4 : i32} : tensor<2x2x2x2x2xi64, #ttg.slice<{dim = 4, parent = #blocked6}>> -> tensor<2x2x2x2x1x2xi64, #blocked6>
    %86 = tt.broadcast %85 : tensor<2x2x2x2x1x2xi64, #blocked6> -> tensor<2x2x2x2x2x2xi64, #blocked6>
    %87 = arith.xori %82, %86 : tensor<2x2x2x2x2x2xi64, #blocked6>
    %88 = tt.reshape %39 : tensor<2xi32, #linear1> -> tensor<1x1x1x1x2x1xi32, #blocked6>
    %89 = arith.cmpi ugt, %82, %87 : tensor<2x2x2x2x2x2xi64, #blocked6>
    %90 = tt.broadcast %83 : tensor<1x1x1x2x1x1xi32, #blocked6> -> tensor<1x1x1x2x2x1xi32, #blocked6>
    %91 = tt.broadcast %88 : tensor<1x1x1x1x2x1xi32, #blocked6> -> tensor<1x1x1x2x2x1xi32, #blocked6>
    %92 = arith.xori %90, %91 : tensor<1x1x1x2x2x1xi32, #blocked6>
    %93 = arith.extui %89 : tensor<2x2x2x2x2x2xi1, #blocked6> to tensor<2x2x2x2x2x2xi32, #blocked6>
    %94 = tt.broadcast %92 : tensor<1x1x1x2x2x1xi32, #blocked6> -> tensor<2x2x2x2x2x2xi32, #blocked6>
    %95 = arith.cmpi ne, %93, %94 : tensor<2x2x2x2x2x2xi32, #blocked6>
    %96 = arith.select %95, %87, %82 : tensor<2x2x2x2x2x2xi1, #blocked6>, tensor<2x2x2x2x2x2xi64, #blocked6>
    %97 = "tt.reduce"(%96) <{axis = 5 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2x2xi64, #blocked6>) -> tensor<2x2x2x2x2xi64, #ttg.slice<{dim = 5, parent = #blocked6}>>
    %98 = tt.expand_dims %97 {axis = 5 : i32} : tensor<2x2x2x2x2xi64, #ttg.slice<{dim = 5, parent = #blocked6}>> -> tensor<2x2x2x2x2x1xi64, #blocked6>
    %99 = tt.broadcast %98 : tensor<2x2x2x2x2x1xi64, #blocked6> -> tensor<2x2x2x2x2x2xi64, #blocked6>
    %100 = arith.xori %96, %99 : tensor<2x2x2x2x2x2xi64, #blocked6>
    %101 = tt.reshape %40 : tensor<2xi32, #linear2> -> tensor<1x1x1x1x1x2xi32, #blocked6>
    %102 = arith.cmpi ugt, %96, %100 : tensor<2x2x2x2x2x2xi64, #blocked6>
    %103 = tt.broadcast %83 : tensor<1x1x1x2x1x1xi32, #blocked6> -> tensor<1x1x1x2x1x2xi32, #blocked6>
    %104 = tt.broadcast %101 : tensor<1x1x1x1x1x2xi32, #blocked6> -> tensor<1x1x1x2x1x2xi32, #blocked6>
    %105 = arith.xori %103, %104 : tensor<1x1x1x2x1x2xi32, #blocked6>
    %106 = arith.extui %102 : tensor<2x2x2x2x2x2xi1, #blocked6> to tensor<2x2x2x2x2x2xi32, #blocked6>
    %107 = tt.broadcast %105 : tensor<1x1x1x2x1x2xi32, #blocked6> -> tensor<2x2x2x2x2x2xi32, #blocked6>
    %108 = arith.cmpi ne, %106, %107 : tensor<2x2x2x2x2x2xi32, #blocked6>
    %109 = arith.select %108, %100, %96 : tensor<2x2x2x2x2x2xi1, #blocked6>, tensor<2x2x2x2x2x2xi64, #blocked6>
    %110 = "tt.reduce"(%109) <{axis = 3 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.maxui %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2x2xi64, #blocked6>) -> tensor<2x2x2x2x2xi64, #ttg.slice<{dim = 3, parent = #blocked6}>>
    %111 = ttg.convert_layout %110 : tensor<2x2x2x2x2xi64, #ttg.slice<{dim = 3, parent = #blocked6}>> -> tensor<2x2x2x2x2xi64, #blocked7>
    %112 = tt.reshape %38 : tensor<2xi32, #linear> -> tensor<1x1x2x1x1xi32, #blocked7>
    %113 = "tt.reduce"(%111) <{axis = 3 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2xi64, #blocked7>) -> tensor<2x2x2x2xi64, #ttg.slice<{dim = 3, parent = #blocked7}>>
    %114 = tt.expand_dims %113 {axis = 3 : i32} : tensor<2x2x2x2xi64, #ttg.slice<{dim = 3, parent = #blocked7}>> -> tensor<2x2x2x1x2xi64, #blocked7>
    %115 = tt.broadcast %114 : tensor<2x2x2x1x2xi64, #blocked7> -> tensor<2x2x2x2x2xi64, #blocked7>
    %116 = arith.xori %111, %115 : tensor<2x2x2x2x2xi64, #blocked7>
    %117 = tt.reshape %39 : tensor<2xi32, #linear1> -> tensor<1x1x1x2x1xi32, #blocked7>
    %118 = arith.cmpi ugt, %111, %116 : tensor<2x2x2x2x2xi64, #blocked7>
    %119 = tt.broadcast %112 : tensor<1x1x2x1x1xi32, #blocked7> -> tensor<1x1x2x2x1xi32, #blocked7>
    %120 = tt.broadcast %117 : tensor<1x1x1x2x1xi32, #blocked7> -> tensor<1x1x2x2x1xi32, #blocked7>
    %121 = arith.xori %119, %120 : tensor<1x1x2x2x1xi32, #blocked7>
    %122 = arith.extui %118 : tensor<2x2x2x2x2xi1, #blocked7> to tensor<2x2x2x2x2xi32, #blocked7>
    %123 = tt.broadcast %121 : tensor<1x1x2x2x1xi32, #blocked7> -> tensor<2x2x2x2x2xi32, #blocked7>
    %124 = arith.cmpi ne, %122, %123 : tensor<2x2x2x2x2xi32, #blocked7>
    %125 = arith.select %124, %116, %111 : tensor<2x2x2x2x2xi1, #blocked7>, tensor<2x2x2x2x2xi64, #blocked7>
    %126 = "tt.reduce"(%125) <{axis = 4 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2xi64, #blocked7>) -> tensor<2x2x2x2xi64, #ttg.slice<{dim = 4, parent = #blocked7}>>
    %127 = tt.expand_dims %126 {axis = 4 : i32} : tensor<2x2x2x2xi64, #ttg.slice<{dim = 4, parent = #blocked7}>> -> tensor<2x2x2x2x1xi64, #blocked7>
    %128 = tt.broadcast %127 : tensor<2x2x2x2x1xi64, #blocked7> -> tensor<2x2x2x2x2xi64, #blocked7>
    %129 = arith.xori %125, %128 : tensor<2x2x2x2x2xi64, #blocked7>
    %130 = tt.reshape %40 : tensor<2xi32, #linear2> -> tensor<1x1x1x1x2xi32, #blocked7>
    %131 = arith.cmpi ugt, %125, %129 : tensor<2x2x2x2x2xi64, #blocked7>
    %132 = tt.broadcast %112 : tensor<1x1x2x1x1xi32, #blocked7> -> tensor<1x1x2x1x2xi32, #blocked7>
    %133 = tt.broadcast %130 : tensor<1x1x1x1x2xi32, #blocked7> -> tensor<1x1x2x1x2xi32, #blocked7>
    %134 = arith.xori %132, %133 : tensor<1x1x2x1x2xi32, #blocked7>
    %135 = arith.extui %131 : tensor<2x2x2x2x2xi1, #blocked7> to tensor<2x2x2x2x2xi32, #blocked7>
    %136 = tt.broadcast %134 : tensor<1x1x2x1x2xi32, #blocked7> -> tensor<2x2x2x2x2xi32, #blocked7>
    %137 = arith.cmpi ne, %135, %136 : tensor<2x2x2x2x2xi32, #blocked7>
    %138 = arith.select %137, %129, %125 : tensor<2x2x2x2x2xi1, #blocked7>, tensor<2x2x2x2x2xi64, #blocked7>
    %139 = "tt.reduce"(%138) <{axis = 2 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.maxui %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2x2xi64, #blocked7>) -> tensor<2x2x2x2xi64, #ttg.slice<{dim = 2, parent = #blocked7}>>
    %140 = ttg.convert_layout %139 : tensor<2x2x2x2xi64, #ttg.slice<{dim = 2, parent = #blocked7}>> -> tensor<2x2x2x2xi64, #blocked8>
    %141 = tt.reshape %38 : tensor<2xi32, #linear> -> tensor<1x2x1x1xi32, #blocked8>
    %142 = "tt.reduce"(%140) <{axis = 2 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2xi64, #blocked8>) -> tensor<2x2x2xi64, #ttg.slice<{dim = 2, parent = #blocked8}>>
    %143 = tt.expand_dims %142 {axis = 2 : i32} : tensor<2x2x2xi64, #ttg.slice<{dim = 2, parent = #blocked8}>> -> tensor<2x2x1x2xi64, #blocked8>
    %144 = tt.broadcast %143 : tensor<2x2x1x2xi64, #blocked8> -> tensor<2x2x2x2xi64, #blocked8>
    %145 = arith.xori %140, %144 : tensor<2x2x2x2xi64, #blocked8>
    %146 = tt.reshape %39 : tensor<2xi32, #linear1> -> tensor<1x1x2x1xi32, #blocked8>
    %147 = arith.cmpi ugt, %140, %145 : tensor<2x2x2x2xi64, #blocked8>
    %148 = tt.broadcast %141 : tensor<1x2x1x1xi32, #blocked8> -> tensor<1x2x2x1xi32, #blocked8>
    %149 = tt.broadcast %146 : tensor<1x1x2x1xi32, #blocked8> -> tensor<1x2x2x1xi32, #blocked8>
    %150 = arith.xori %148, %149 : tensor<1x2x2x1xi32, #blocked8>
    %151 = arith.extui %147 : tensor<2x2x2x2xi1, #blocked8> to tensor<2x2x2x2xi32, #blocked8>
    %152 = tt.broadcast %150 : tensor<1x2x2x1xi32, #blocked8> -> tensor<2x2x2x2xi32, #blocked8>
    %153 = arith.cmpi ne, %151, %152 : tensor<2x2x2x2xi32, #blocked8>
    %154 = arith.select %153, %145, %140 : tensor<2x2x2x2xi1, #blocked8>, tensor<2x2x2x2xi64, #blocked8>
    %155 = "tt.reduce"(%154) <{axis = 3 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2xi64, #blocked8>) -> tensor<2x2x2xi64, #ttg.slice<{dim = 3, parent = #blocked8}>>
    %156 = tt.expand_dims %155 {axis = 3 : i32} : tensor<2x2x2xi64, #ttg.slice<{dim = 3, parent = #blocked8}>> -> tensor<2x2x2x1xi64, #blocked8>
    %157 = tt.broadcast %156 : tensor<2x2x2x1xi64, #blocked8> -> tensor<2x2x2x2xi64, #blocked8>
    %158 = arith.xori %154, %157 : tensor<2x2x2x2xi64, #blocked8>
    %159 = tt.reshape %40 : tensor<2xi32, #linear2> -> tensor<1x1x1x2xi32, #blocked8>
    %160 = arith.cmpi ugt, %154, %158 : tensor<2x2x2x2xi64, #blocked8>
    %161 = tt.broadcast %141 : tensor<1x2x1x1xi32, #blocked8> -> tensor<1x2x1x2xi32, #blocked8>
    %162 = tt.broadcast %159 : tensor<1x1x1x2xi32, #blocked8> -> tensor<1x2x1x2xi32, #blocked8>
    %163 = arith.xori %161, %162 : tensor<1x2x1x2xi32, #blocked8>
    %164 = arith.extui %160 : tensor<2x2x2x2xi1, #blocked8> to tensor<2x2x2x2xi32, #blocked8>
    %165 = tt.broadcast %163 : tensor<1x2x1x2xi32, #blocked8> -> tensor<2x2x2x2xi32, #blocked8>
    %166 = arith.cmpi ne, %164, %165 : tensor<2x2x2x2xi32, #blocked8>
    %167 = arith.select %166, %158, %154 : tensor<2x2x2x2xi1, #blocked8>, tensor<2x2x2x2xi64, #blocked8>
    %168 = "tt.reduce"(%167) <{axis = 1 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.maxui %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2x2xi64, #blocked8>) -> tensor<2x2x2xi64, #ttg.slice<{dim = 1, parent = #blocked8}>>
    %169 = ttg.convert_layout %168 : tensor<2x2x2xi64, #ttg.slice<{dim = 1, parent = #blocked8}>> -> tensor<2x2x2xi64, #blocked9>
    %170 = tt.reshape %38 : tensor<2xi32, #linear> -> tensor<2x1x1xi32, #blocked9>
    %171 = "tt.reduce"(%169) <{axis = 1 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2xi64, #blocked9>) -> tensor<2x2xi64, #ttg.slice<{dim = 1, parent = #blocked9}>>
    %172 = tt.expand_dims %171 {axis = 1 : i32} : tensor<2x2xi64, #ttg.slice<{dim = 1, parent = #blocked9}>> -> tensor<2x1x2xi64, #blocked9>
    %173 = tt.broadcast %172 : tensor<2x1x2xi64, #blocked9> -> tensor<2x2x2xi64, #blocked9>
    %174 = arith.xori %169, %173 : tensor<2x2x2xi64, #blocked9>
    %175 = tt.reshape %39 : tensor<2xi32, #linear1> -> tensor<1x2x1xi32, #blocked9>
    %176 = arith.cmpi ugt, %169, %174 : tensor<2x2x2xi64, #blocked9>
    %177 = tt.broadcast %170 : tensor<2x1x1xi32, #blocked9> -> tensor<2x2x1xi32, #blocked9>
    %178 = tt.broadcast %175 : tensor<1x2x1xi32, #blocked9> -> tensor<2x2x1xi32, #blocked9>
    %179 = arith.xori %177, %178 : tensor<2x2x1xi32, #blocked9>
    %180 = arith.extui %176 : tensor<2x2x2xi1, #blocked9> to tensor<2x2x2xi32, #blocked9>
    %181 = tt.broadcast %179 : tensor<2x2x1xi32, #blocked9> -> tensor<2x2x2xi32, #blocked9>
    %182 = arith.cmpi ne, %180, %181 : tensor<2x2x2xi32, #blocked9>
    %183 = arith.select %182, %174, %169 : tensor<2x2x2xi1, #blocked9>, tensor<2x2x2xi64, #blocked9>
    %184 = "tt.reduce"(%183) <{axis = 2 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2xi64, #blocked9>) -> tensor<2x2xi64, #ttg.slice<{dim = 2, parent = #blocked9}>>
    %185 = tt.expand_dims %184 {axis = 2 : i32} : tensor<2x2xi64, #ttg.slice<{dim = 2, parent = #blocked9}>> -> tensor<2x2x1xi64, #blocked9>
    %186 = tt.broadcast %185 : tensor<2x2x1xi64, #blocked9> -> tensor<2x2x2xi64, #blocked9>
    %187 = arith.xori %183, %186 : tensor<2x2x2xi64, #blocked9>
    %188 = tt.reshape %40 : tensor<2xi32, #linear2> -> tensor<1x1x2xi32, #blocked9>
    %189 = arith.cmpi ugt, %183, %187 : tensor<2x2x2xi64, #blocked9>
    %190 = tt.broadcast %170 : tensor<2x1x1xi32, #blocked9> -> tensor<2x1x2xi32, #blocked9>
    %191 = tt.broadcast %188 : tensor<1x1x2xi32, #blocked9> -> tensor<2x1x2xi32, #blocked9>
    %192 = arith.xori %190, %191 : tensor<2x1x2xi32, #blocked9>
    %193 = arith.extui %189 : tensor<2x2x2xi1, #blocked9> to tensor<2x2x2xi32, #blocked9>
    %194 = tt.broadcast %192 : tensor<2x1x2xi32, #blocked9> -> tensor<2x2x2xi32, #blocked9>
    %195 = arith.cmpi ne, %193, %194 : tensor<2x2x2xi32, #blocked9>
    %196 = arith.select %195, %187, %183 : tensor<2x2x2xi1, #blocked9>, tensor<2x2x2xi64, #blocked9>
    %197 = "tt.reduce"(%196) <{axis = 0 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.maxui %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2x2xi64, #blocked9>) -> tensor<2x2xi64, #ttg.slice<{dim = 0, parent = #blocked9}>>
    %198 = ttg.convert_layout %197 : tensor<2x2xi64, #ttg.slice<{dim = 0, parent = #blocked9}>> -> tensor<2x2xi64, #blocked3>
    %199 = "tt.reduce"(%198) <{axis = 0 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2xi64, #blocked3>) -> tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %200 = tt.expand_dims %199 {axis = 0 : i32} : tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi64, #blocked3>
    %201 = tt.broadcast %200 : tensor<1x2xi64, #blocked3> -> tensor<2x2xi64, #blocked3>
    %202 = arith.xori %198, %201 : tensor<2x2xi64, #blocked3>
    %203 = tt.reshape %39 : tensor<2xi32, #linear1> -> tensor<2x1xi32, #blocked3>
    %204 = arith.cmpi ugt, %198, %202 : tensor<2x2xi64, #blocked3>
    %205 = arith.xori %203, %cst_17 : tensor<2x1xi32, #blocked3>
    %206 = arith.extui %204 : tensor<2x2xi1, #blocked3> to tensor<2x2xi32, #blocked3>
    %207 = tt.broadcast %205 : tensor<2x1xi32, #blocked3> -> tensor<2x2xi32, #blocked3>
    %208 = arith.cmpi ne, %206, %207 : tensor<2x2xi32, #blocked3>
    %209 = arith.select %208, %202, %198 : tensor<2x2xi1, #blocked3>, tensor<2x2xi64, #blocked3>
    %210 = "tt.reduce"(%209) <{axis = 1 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2xi64, #blocked3>) -> tensor<2xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %211 = tt.expand_dims %210 {axis = 1 : i32} : tensor<2xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<2x1xi64, #blocked3>
    %212 = tt.broadcast %211 : tensor<2x1xi64, #blocked3> -> tensor<2x2xi64, #blocked3>
    %213 = arith.xori %209, %212 : tensor<2x2xi64, #blocked3>
    %214 = tt.reshape %41 : tensor<2xi32, #blocked2> -> tensor<1x2xi32, #blocked3>
    %215 = arith.cmpi ugt, %209, %213 : tensor<2x2xi64, #blocked3>
    %216 = arith.xori %214, %cst_7 : tensor<1x2xi32, #blocked3>
    %217 = arith.extui %215 : tensor<2x2xi1, #blocked3> to tensor<2x2xi32, #blocked3>
    %218 = tt.broadcast %216 : tensor<1x2xi32, #blocked3> -> tensor<2x2xi32, #blocked3>
    %219 = arith.cmpi ne, %217, %218 : tensor<2x2xi32, #blocked3>
    %220 = arith.select %219, %213, %209 : tensor<2x2xi1, #blocked3>, tensor<2x2xi64, #blocked3>
    %221 = tt.reshape %220 : tensor<2x2xi64, #blocked3> -> tensor<1x4xi64, #blocked>
    %222 = arith.shli %221, %cst_13 : tensor<1x4xi64, #blocked>
    %223 = arith.shrui %221, %cst_12 : tensor<1x4xi64, #blocked>
    %224 = arith.ori %222, %223 : tensor<1x4xi64, #blocked>
    %225 = tt.reshape %224 : tensor<1x4xi64, #blocked> -> tensor<2x2xi64, #blocked3>
    %226 = "tt.reduce"(%225) <{axis = 1 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2xi64, #blocked3>) -> tensor<2xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %227 = tt.expand_dims %226 {axis = 1 : i32} : tensor<2xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<2x1xi64, #blocked3>
    %228 = tt.broadcast %227 : tensor<2x1xi64, #blocked3> -> tensor<2x2xi64, #blocked3>
    %229 = arith.xori %225, %228 : tensor<2x2xi64, #blocked3>
    %230 = arith.cmpi ugt, %225, %229 : tensor<2x2xi64, #blocked3>
    %231 = tt.broadcast %203 : tensor<2x1xi32, #blocked3> -> tensor<2x2xi32, #blocked3>
    %232 = tt.broadcast %214 : tensor<1x2xi32, #blocked3> -> tensor<2x2xi32, #blocked3>
    %233 = arith.xori %231, %232 : tensor<2x2xi32, #blocked3>
    %234 = arith.extui %230 : tensor<2x2xi1, #blocked3> to tensor<2x2xi32, #blocked3>
    %235 = arith.cmpi ne, %234, %233 : tensor<2x2xi32, #blocked3>
    %236 = arith.select %235, %229, %225 : tensor<2x2xi1, #blocked3>, tensor<2x2xi64, #blocked3>
    %237 = "tt.reduce"(%236) <{axis = 0 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2xi64, #blocked3>) -> tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %238 = tt.expand_dims %237 {axis = 0 : i32} : tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi64, #blocked3>
    %239 = tt.broadcast %238 : tensor<1x2xi64, #blocked3> -> tensor<2x2xi64, #blocked3>
    %240 = arith.xori %236, %239 : tensor<2x2xi64, #blocked3>
    %241 = arith.cmpi ugt, %236, %240 : tensor<2x2xi64, #blocked3>
    %242 = arith.extui %241 : tensor<2x2xi1, #blocked3> to tensor<2x2xi32, #blocked3>
    %243 = arith.cmpi ne, %242, %231 : tensor<2x2xi32, #blocked3>
    %244 = arith.select %243, %240, %236 : tensor<2x2xi1, #blocked3>, tensor<2x2xi64, #blocked3>
    %245 = "tt.reduce"(%244) <{axis = 1 : i32}> ({
    ^bb0(%arg14: i64, %arg15: i64):
      %298 = arith.xori %arg14, %arg15 : i64
      tt.reduce.return %298 : i64
    }) : (tensor<2x2xi64, #blocked3>) -> tensor<2xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %246 = tt.expand_dims %245 {axis = 1 : i32} : tensor<2xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<2x1xi64, #blocked3>
    %247 = tt.broadcast %246 : tensor<2x1xi64, #blocked3> -> tensor<2x2xi64, #blocked3>
    %248 = arith.xori %244, %247 : tensor<2x2xi64, #blocked3>
    %249 = arith.cmpi ugt, %244, %248 : tensor<2x2xi64, #blocked3>
    %250 = arith.extui %249 : tensor<2x2xi1, #blocked3> to tensor<2x2xi32, #blocked3>
    %251 = arith.cmpi ne, %250, %232 : tensor<2x2xi32, #blocked3>
    %252 = arith.select %251, %248, %244 : tensor<2x2xi1, #blocked3>, tensor<2x2xi64, #blocked3>
    %253 = tt.reshape %252 : tensor<2x2xi64, #blocked3> -> tensor<1x4xi64, #blocked>
    %254 = arith.shrui %253, %cst_13 : tensor<1x4xi64, #blocked>
    %255 = arith.trunci %254 : tensor<1x4xi64, #blocked> to tensor<1x4xi32, #blocked>
    %256 = arith.trunci %253 : tensor<1x4xi64, #blocked> to tensor<1x4xi32, #blocked>
    %257 = arith.andi %256, %cst_6 : tensor<1x4xi32, #blocked>
    %258 = arith.cmpi eq, %257, %cst : tensor<1x4xi32, #blocked>
    %259 = arith.select %258, %cst_5, %cst_6 : tensor<1x4xi1, #blocked>, tensor<1x4xi32, #blocked>
    %260 = arith.xori %256, %259 : tensor<1x4xi32, #blocked>
    %261 = tt.bitcast %260 : tensor<1x4xi32, #blocked> -> tensor<1x4xf32, #blocked>
    %262 = amdg.buffer_load %arg13[%255] {amdgpu.split_soffset_safe} : tensor<1x4xf32, #blocked>
    %263 = arith.subf %261, %262 : tensor<1x4xf32, #blocked>
    %264 = "tt.reduce"(%263) <{axis = 1 : i32}> ({
    ^bb0(%arg14: f32, %arg15: f32):
      %298 = arith.addf %arg14, %arg15 : f32
      tt.reduce.return %298 : f32
    }) : (tensor<1x4xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %265 = tt.expand_dims %264 {axis = 1 : i32} : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1xf32, #blocked>
    %266 = arith.addf %265, %cst_18 : tensor<1x1xf32, #blocked>
    %267 = tt.broadcast %266 : tensor<1x1xf32, #blocked> -> tensor<1x4xf32, #blocked>
    %268 = arith.divf %263, %267 : tensor<1x4xf32, #blocked>
    %269 = arith.mulf %268, %cst_0 : tensor<1x4xf32, #blocked>
    %270 = arith.muli %0, %arg4 : i32
    %271 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %272 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked1}>}>>
    %273 = tt.expand_dims %272 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked1}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %274 = tt.splat %270 : i32 -> tensor<1x4xi32, #blocked>
    %275 = tt.expand_dims %271 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi32, #blocked>
    %276 = arith.addi %275, %274 : tensor<1x4xi32, #blocked>
    %277 = tt.splat %3 : i1 -> tensor<1x4xi1, #blocked>
    amdg.buffer_store %269, %arg2[%276], %277 : tensor<1x4xf32, #blocked>
    tt.return
  }
}
