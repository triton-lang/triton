// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s
// Reproducer for https://github.com/pytorch/pytorch/issues/130101
// This is difficult to minimize as it specifically happens when a long slice of
// operations are being rematerialized that reuses the same node with two different
// layouts.

// CHECK: tt.return
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 2, 2], warpsPerCTA = [2, 1, 1], order = [2, 1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 2, 4], warpsPerCTA = [2, 1, 1], order = [2, 1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [2, 2, 8], warpsPerCTA = [2, 1, 1], order = [2, 1, 0]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [2, 1, 1], order = [2, 1, 0]}>
#blocked6 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 1], order = [2, 1, 0]}>
#blocked7 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 2], order = [2, 1, 0]}>
#blocked8 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked9 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [16, 2, 1], warpsPerCTA = [2, 1, 1], order = [2, 1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "cuda:86", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @triton_(%arg0: tensor<1x256xi8, #blocked>, %arg1: tensor<1x256xi8, #blocked>, %arg2: tensor<1x256xi8, #blocked>, %arg3: tensor<1x256xi8, #blocked>, %arg4: tensor<1x256xi8, #blocked>, %arg5: tensor<1x256xi8, #blocked>, %arg6: tensor<1x256xi8, #blocked>, %arg7: tensor<1x256xi8, #blocked>, %arg8: tensor<1x256xi8, #blocked>, %arg9: tensor<1x256xi8, #blocked>, %arg10: tensor<1x256xi8, #blocked>, %arg11: tensor<1x256xi8, #blocked>, %arg12: tensor<1x256xi8, #blocked>, %arg13: tensor<1x256xi8, #blocked>, %arg14: tensor<1x256xi8, #blocked>, %arg15: tensor<1x256xi8, #blocked>, %arg16: tensor<1x256xi8, #blocked>, %arg17: tensor<1x256xi8, #blocked>, %arg18: tensor<1x256xi8, #blocked>, %arg19: tensor<1x256xi8, #blocked>, %arg20: tensor<1x256xi8, #blocked>, %arg21: tensor<1x256xi8, #blocked>, %arg22: tensor<1x256xi8, #blocked>, %arg23: tensor<1x256xi8, #blocked>, %arg24: tensor<1x256xi8, #blocked>, %arg25: tensor<1x256xi8, #blocked>, %arg26: tensor<1x256xi8, #blocked>, %arg27: tensor<1x256xi8, #blocked>, %arg28: tensor<1x256xi8, #blocked>, %arg29: tensor<1x256xi8, #blocked>, %arg30: tensor<1x256xi8, #blocked>, %arg31: tensor<1x256xi8, #blocked>, %arg32: tensor<1x256xi8, #blocked>, %arg33: tensor<1x256xi8, #blocked>, %arg34: tensor<1x256xi8, #blocked>, %arg35: tensor<1x256xi8, #blocked>) -> tensor<1x256xi32, #blocked1> attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<1x2x1xi32, #blocked2>
    %cst_0 = arith.constant dense<1> : tensor<1x2x1xi32, #blocked3>
    %cst_1 = arith.constant dense<1> : tensor<1x2x1xi32, #blocked4>
    %cst_2 = arith.constant dense<1> : tensor<1x2x1xi32, #blocked5>
    %cst_3 = arith.constant dense<1> : tensor<1x2x1xi32, #blocked6>
    %cst_4 = arith.constant dense<1> : tensor<1x2x1xi32, #blocked7>
    %cst_5 = arith.constant dense<0> : tensor<1x256xi32, #blocked8>
    %cst_6 = arith.constant dense<1> : tensor<1x2x1xi32, #blocked9>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked8}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked8}>> -> tensor<1x256xi32, #blocked8>
    %2 = tt.reshape %1 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<128x2x1xi32, #blocked9>
    %3 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked2}>}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked2}>>
    %5 = tt.expand_dims %4 {axis = 2 : i32} : tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked2}>> -> tensor<1x2x1xi32, #blocked2>
    %6 = tt.broadcast %5 : tensor<1x2x1xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %7 = tt.reshape %6 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %8 = arith.trunci %5 : tensor<1x2x1xi32, #blocked2> to tensor<1x2x1xi8, #blocked2>
    %9 = arith.extsi %8 : tensor<1x2x1xi8, #blocked2> to tensor<1x2x1xi32, #blocked2>
    %10 = tt.broadcast %9 : tensor<1x2x1xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %11 = tt.broadcast %8 : tensor<1x2x1xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %12 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked3}>}>>
    %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked3}>}>> -> tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked3}>>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked3}>> -> tensor<1x2x1xi32, #blocked3>
    %15 = tt.broadcast %14 : tensor<1x2x1xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %16 = tt.reshape %15 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %17 = arith.trunci %14 : tensor<1x2x1xi32, #blocked3> to tensor<1x2x1xi8, #blocked3>
    %18 = arith.extsi %17 : tensor<1x2x1xi8, #blocked3> to tensor<1x2x1xi32, #blocked3>
    %19 = tt.broadcast %18 : tensor<1x2x1xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %20 = tt.broadcast %17 : tensor<1x2x1xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %21 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked4}>}>>
    %22 = tt.expand_dims %21 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked4}>}>> -> tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked4}>>
    %23 = tt.expand_dims %22 {axis = 2 : i32} : tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked4}>> -> tensor<1x2x1xi32, #blocked4>
    %24 = tt.broadcast %23 : tensor<1x2x1xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %25 = tt.reshape %24 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %26 = arith.trunci %23 : tensor<1x2x1xi32, #blocked4> to tensor<1x2x1xi8, #blocked4>
    %27 = arith.extsi %26 : tensor<1x2x1xi8, #blocked4> to tensor<1x2x1xi32, #blocked4>
    %28 = tt.broadcast %27 : tensor<1x2x1xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %29 = tt.broadcast %26 : tensor<1x2x1xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %30 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked5}>}>>
    %31 = tt.expand_dims %30 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked5}>}>> -> tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked5}>>
    %32 = tt.expand_dims %31 {axis = 2 : i32} : tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked5}>> -> tensor<1x2x1xi32, #blocked5>
    %33 = tt.broadcast %32 : tensor<1x2x1xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %34 = tt.reshape %33 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %35 = arith.trunci %32 : tensor<1x2x1xi32, #blocked5> to tensor<1x2x1xi8, #blocked5>
    %36 = arith.extsi %35 : tensor<1x2x1xi8, #blocked5> to tensor<1x2x1xi32, #blocked5>
    %37 = tt.broadcast %36 : tensor<1x2x1xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %38 = tt.broadcast %35 : tensor<1x2x1xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %39 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked6}>}>>
    %40 = tt.expand_dims %39 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked6}>}>> -> tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked6}>>
    %41 = tt.expand_dims %40 {axis = 2 : i32} : tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked6}>> -> tensor<1x2x1xi32, #blocked6>
    %42 = tt.broadcast %41 : tensor<1x2x1xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %43 = tt.reshape %42 {allow_reorder = false} : tensor<4x2x32xi32, #blocked6> -> tensor<1x256xi32, #blocked8>
    %44 = arith.trunci %41 : tensor<1x2x1xi32, #blocked6> to tensor<1x2x1xi8, #blocked6>
    %45 = arith.extsi %44 : tensor<1x2x1xi8, #blocked6> to tensor<1x2x1xi32, #blocked6>
    %46 = tt.broadcast %45 : tensor<1x2x1xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %47 = tt.broadcast %44 : tensor<1x2x1xi8, #blocked6> -> tensor<4x2x32xi8, #blocked6>
    %48 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked7}>}>>
    %49 = tt.expand_dims %48 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked7}>}>> -> tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked7}>>
    %50 = tt.expand_dims %49 {axis = 2 : i32} : tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked7}>> -> tensor<1x2x1xi32, #blocked7>
    %51 = tt.broadcast %50 : tensor<1x2x1xi32, #blocked7> -> tensor<2x2x64xi32, #blocked7>
    %52 = tt.reshape %51 {allow_reorder = false} : tensor<2x2x64xi32, #blocked7> -> tensor<1x256xi32, #blocked8>
    %53 = tt.broadcast %50 : tensor<1x2x1xi32, #blocked7> -> tensor<1x2x128xi32, #blocked7>
    %54 = tt.reshape %53 {allow_reorder = false} : tensor<1x2x128xi32, #blocked7> -> tensor<1x256xi32, #blocked8>
    %55 = arith.trunci %50 : tensor<1x2x1xi32, #blocked7> to tensor<1x2x1xi8, #blocked7>
    %56 = arith.extsi %55 : tensor<1x2x1xi8, #blocked7> to tensor<1x2x1xi32, #blocked7>
    %57 = tt.broadcast %56 : tensor<1x2x1xi32, #blocked7> -> tensor<2x2x64xi32, #blocked7>
    %58 = tt.broadcast %56 : tensor<1x2x1xi32, #blocked7> -> tensor<1x2x128xi32, #blocked7>
    %59 = tt.broadcast %55 : tensor<1x2x1xi8, #blocked7> -> tensor<2x2x64xi8, #blocked7>
    %60 = tt.broadcast %55 : tensor<1x2x1xi8, #blocked7> -> tensor<1x2x128xi8, #blocked7>
    %61 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked9}>}>>
    %62 = tt.expand_dims %61 {axis = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #triton_gpu.slice<{dim = 2, parent = #blocked9}>}>> -> tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked9}>>
    %63 = tt.expand_dims %62 {axis = 2 : i32} : tensor<1x2xi32, #triton_gpu.slice<{dim = 2, parent = #blocked9}>> -> tensor<1x2x1xi32, #blocked9>
    %64 = arith.trunci %63 : tensor<1x2x1xi32, #blocked9> to tensor<1x2x1xi8, #blocked9>
    %65 = arith.extsi %64 : tensor<1x2x1xi8, #blocked9> to tensor<1x2x1xi32, #blocked9>
    %66 = tt.broadcast %65 : tensor<1x2x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %67 = arith.muli %2, %66 : tensor<128x2x1xi32, #blocked9>
    %68 = "tt.reduce"(%67) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %69 = tt.expand_dims %68 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %70 = tt.broadcast %69 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %71 = tt.reshape %70 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %72 = tt.broadcast %64 : tensor<1x2x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %73 = arith.subi %cst_6, %65 : tensor<1x2x1xi32, #blocked9>
    %74 = arith.trunci %73 : tensor<1x2x1xi32, #blocked9> to tensor<1x2x1xi8, #blocked9>
    %75 = tt.broadcast %74 : tensor<1x2x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %76 = arith.extsi %74 : tensor<1x2x1xi8, #blocked9> to tensor<1x2x1xi32, #blocked9>
    %77 = tt.broadcast %76 : tensor<1x2x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %78 = arith.muli %2, %77 : tensor<128x2x1xi32, #blocked9>
    %79 = "tt.reduce"(%78) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %80 = tt.expand_dims %79 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %81 = tt.broadcast %80 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %82 = tt.reshape %81 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %83 = arith.cmpi sgt, %82, %71 : tensor<1x256xi32, #blocked8>
    %84 = arith.xori %82, %71 : tensor<1x256xi32, #blocked8>
    %85 = arith.subi %cst_4, %56 : tensor<1x2x1xi32, #blocked7>
    %86 = arith.trunci %85 : tensor<1x2x1xi32, #blocked7> to tensor<1x2x1xi8, #blocked7>
    %87 = tt.broadcast %86 : tensor<1x2x1xi8, #blocked7> -> tensor<2x2x64xi8, #blocked7>
    %88 = arith.extsi %86 : tensor<1x2x1xi8, #blocked7> to tensor<1x2x1xi32, #blocked7>
    %89 = tt.broadcast %88 : tensor<1x2x1xi32, #blocked7> -> tensor<2x2x64xi32, #blocked7>
    %90 = tt.broadcast %88 : tensor<1x2x1xi32, #blocked7> -> tensor<1x2x128xi32, #blocked7>
    %91 = tt.broadcast %86 : tensor<1x2x1xi8, #blocked7> -> tensor<1x2x128xi8, #blocked7>
    %92 = arith.subi %cst_3, %45 : tensor<1x2x1xi32, #blocked6>
    %93 = arith.trunci %92 : tensor<1x2x1xi32, #blocked6> to tensor<1x2x1xi8, #blocked6>
    %94 = tt.broadcast %93 : tensor<1x2x1xi8, #blocked6> -> tensor<4x2x32xi8, #blocked6>
    %95 = arith.extsi %93 : tensor<1x2x1xi8, #blocked6> to tensor<1x2x1xi32, #blocked6>
    %96 = tt.broadcast %95 : tensor<1x2x1xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %97 = arith.subi %cst_2, %36 : tensor<1x2x1xi32, #blocked5>
    %98 = arith.trunci %97 : tensor<1x2x1xi32, #blocked5> to tensor<1x2x1xi8, #blocked5>
    %99 = tt.broadcast %98 : tensor<1x2x1xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %100 = arith.extsi %98 : tensor<1x2x1xi8, #blocked5> to tensor<1x2x1xi32, #blocked5>
    %101 = tt.broadcast %100 : tensor<1x2x1xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %102 = arith.subi %cst_1, %27 : tensor<1x2x1xi32, #blocked4>
    %103 = arith.trunci %102 : tensor<1x2x1xi32, #blocked4> to tensor<1x2x1xi8, #blocked4>
    %104 = tt.broadcast %103 : tensor<1x2x1xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %105 = arith.extsi %103 : tensor<1x2x1xi8, #blocked4> to tensor<1x2x1xi32, #blocked4>
    %106 = tt.broadcast %105 : tensor<1x2x1xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %107 = arith.subi %cst_0, %18 : tensor<1x2x1xi32, #blocked3>
    %108 = arith.trunci %107 : tensor<1x2x1xi32, #blocked3> to tensor<1x2x1xi8, #blocked3>
    %109 = tt.broadcast %108 : tensor<1x2x1xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %110 = arith.extsi %108 : tensor<1x2x1xi8, #blocked3> to tensor<1x2x1xi32, #blocked3>
    %111 = tt.broadcast %110 : tensor<1x2x1xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %112 = arith.subi %cst, %9 : tensor<1x2x1xi32, #blocked2>
    %113 = arith.trunci %112 : tensor<1x2x1xi32, #blocked2> to tensor<1x2x1xi8, #blocked2>
    %114 = tt.broadcast %113 : tensor<1x2x1xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %115 = arith.extsi %113 : tensor<1x2x1xi8, #blocked2> to tensor<1x2x1xi32, #blocked2>
    %116 = tt.broadcast %115 : tensor<1x2x1xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %117 = triton_gpu.convert_layout %arg0 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %118 = tt.reshape %117 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<128x2x1xi8, #blocked9>
    %119 = arith.muli %118, %75 : tensor<128x2x1xi8, #blocked9>
    %120 = "tt.reduce"(%119) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %121 = tt.expand_dims %120 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %122 = tt.broadcast %121 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %123 = tt.reshape %122 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %124 = arith.muli %118, %72 : tensor<128x2x1xi8, #blocked9>
    %125 = "tt.reduce"(%124) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %126 = tt.expand_dims %125 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %127 = tt.broadcast %126 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %128 = tt.reshape %127 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %129 = arith.cmpi slt, %123, %128 : tensor<1x256xi8, #blocked8>
    %130 = arith.cmpi eq, %123, %128 : tensor<1x256xi8, #blocked8>
    %131 = arith.andi %130, %83 : tensor<1x256xi1, #blocked8>
    %132 = arith.ori %129, %131 : tensor<1x256xi1, #blocked8>
    %133 = arith.extui %132 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %134 = arith.xori %133, %7 : tensor<1x256xi32, #blocked8>
    %135 = arith.cmpi ne, %134, %cst_5 : tensor<1x256xi32, #blocked8>
    %136 = arith.select %135, %84, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %137 = arith.xori %1, %136 : tensor<1x256xi32, #blocked8>
    %138 = tt.reshape %137 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<64x2x2xi32, #blocked2>
    %139 = arith.muli %138, %116 : tensor<64x2x2xi32, #blocked2>
    %140 = "tt.reduce"(%139) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %141 = tt.expand_dims %140 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %142 = tt.broadcast %141 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %143 = tt.reshape %142 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %144 = arith.muli %138, %10 : tensor<64x2x2xi32, #blocked2>
    %145 = "tt.reduce"(%144) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %146 = tt.expand_dims %145 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %147 = tt.broadcast %146 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %148 = tt.reshape %147 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %149 = arith.cmpi sgt, %143, %148 : tensor<1x256xi32, #blocked8>
    %150 = arith.xori %143, %148 : tensor<1x256xi32, #blocked8>
    %151 = triton_gpu.convert_layout %arg1 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %152 = tt.reshape %151 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<64x2x2xi8, #blocked2>
    %153 = arith.muli %152, %114 : tensor<64x2x2xi8, #blocked2>
    %154 = "tt.reduce"(%153) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %155 = tt.expand_dims %154 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %156 = tt.broadcast %155 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %157 = tt.reshape %156 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %158 = arith.muli %152, %11 : tensor<64x2x2xi8, #blocked2>
    %159 = "tt.reduce"(%158) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %160 = tt.expand_dims %159 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %161 = tt.broadcast %160 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %162 = tt.reshape %161 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %163 = arith.cmpi slt, %157, %162 : tensor<1x256xi8, #blocked8>
    %164 = arith.cmpi eq, %157, %162 : tensor<1x256xi8, #blocked8>
    %165 = arith.andi %164, %149 : tensor<1x256xi1, #blocked8>
    %166 = arith.ori %163, %165 : tensor<1x256xi1, #blocked8>
    %167 = arith.extui %166 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %168 = arith.xori %167, %16 : tensor<1x256xi32, #blocked8>
    %169 = arith.cmpi ne, %168, %cst_5 : tensor<1x256xi32, #blocked8>
    %170 = arith.select %169, %150, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %171 = arith.xori %137, %170 : tensor<1x256xi32, #blocked8>
    %172 = tt.reshape %171 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<128x2x1xi32, #blocked9>
    %173 = arith.muli %172, %77 : tensor<128x2x1xi32, #blocked9>
    %174 = "tt.reduce"(%173) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %175 = tt.expand_dims %174 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %176 = tt.broadcast %175 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %177 = tt.reshape %176 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %178 = arith.muli %172, %66 : tensor<128x2x1xi32, #blocked9>
    %179 = "tt.reduce"(%178) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %180 = tt.expand_dims %179 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %181 = tt.broadcast %180 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %182 = tt.reshape %181 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %183 = arith.cmpi sgt, %177, %182 : tensor<1x256xi32, #blocked8>
    %184 = arith.xori %177, %182 : tensor<1x256xi32, #blocked8>
    %185 = triton_gpu.convert_layout %arg2 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %186 = tt.reshape %185 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<128x2x1xi8, #blocked9>
    %187 = arith.muli %186, %75 : tensor<128x2x1xi8, #blocked9>
    %188 = "tt.reduce"(%187) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %189 = tt.expand_dims %188 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %190 = tt.broadcast %189 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %191 = tt.reshape %190 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %192 = arith.muli %186, %72 : tensor<128x2x1xi8, #blocked9>
    %193 = "tt.reduce"(%192) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %194 = tt.expand_dims %193 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %195 = tt.broadcast %194 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %196 = tt.reshape %195 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %197 = arith.cmpi slt, %191, %196 : tensor<1x256xi8, #blocked8>
    %198 = arith.cmpi eq, %191, %196 : tensor<1x256xi8, #blocked8>
    %199 = arith.andi %198, %183 : tensor<1x256xi1, #blocked8>
    %200 = arith.ori %197, %199 : tensor<1x256xi1, #blocked8>
    %201 = arith.extui %200 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %202 = arith.xori %201, %16 : tensor<1x256xi32, #blocked8>
    %203 = arith.cmpi ne, %202, %cst_5 : tensor<1x256xi32, #blocked8>
    %204 = arith.select %203, %184, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %205 = arith.xori %171, %204 : tensor<1x256xi32, #blocked8>
    %206 = tt.reshape %205 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<32x2x4xi32, #blocked3>
    %207 = arith.muli %206, %111 : tensor<32x2x4xi32, #blocked3>
    %208 = "tt.reduce"(%207) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %209 = tt.expand_dims %208 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %210 = tt.broadcast %209 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %211 = tt.reshape %210 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %212 = arith.muli %206, %19 : tensor<32x2x4xi32, #blocked3>
    %213 = "tt.reduce"(%212) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %214 = tt.expand_dims %213 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %215 = tt.broadcast %214 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %216 = tt.reshape %215 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %217 = arith.cmpi sgt, %211, %216 : tensor<1x256xi32, #blocked8>
    %218 = arith.xori %211, %216 : tensor<1x256xi32, #blocked8>
    %219 = triton_gpu.convert_layout %arg3 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %220 = tt.reshape %219 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<32x2x4xi8, #blocked3>
    %221 = arith.muli %220, %109 : tensor<32x2x4xi8, #blocked3>
    %222 = "tt.reduce"(%221) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %223 = tt.expand_dims %222 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %224 = tt.broadcast %223 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %225 = tt.reshape %224 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %226 = arith.muli %220, %20 : tensor<32x2x4xi8, #blocked3>
    %227 = "tt.reduce"(%226) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %228 = tt.expand_dims %227 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %229 = tt.broadcast %228 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %230 = tt.reshape %229 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %231 = arith.cmpi slt, %225, %230 : tensor<1x256xi8, #blocked8>
    %232 = arith.cmpi eq, %225, %230 : tensor<1x256xi8, #blocked8>
    %233 = arith.andi %232, %217 : tensor<1x256xi1, #blocked8>
    %234 = arith.ori %231, %233 : tensor<1x256xi1, #blocked8>
    %235 = arith.extui %234 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %236 = arith.xori %235, %25 : tensor<1x256xi32, #blocked8>
    %237 = arith.cmpi ne, %236, %cst_5 : tensor<1x256xi32, #blocked8>
    %238 = arith.select %237, %218, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %239 = arith.xori %205, %238 : tensor<1x256xi32, #blocked8>
    %240 = tt.reshape %239 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<64x2x2xi32, #blocked2>
    %241 = arith.muli %240, %116 : tensor<64x2x2xi32, #blocked2>
    %242 = "tt.reduce"(%241) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %243 = tt.expand_dims %242 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %244 = tt.broadcast %243 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %245 = tt.reshape %244 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %246 = arith.muli %240, %10 : tensor<64x2x2xi32, #blocked2>
    %247 = "tt.reduce"(%246) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %248 = tt.expand_dims %247 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %249 = tt.broadcast %248 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %250 = tt.reshape %249 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %251 = arith.cmpi sgt, %245, %250 : tensor<1x256xi32, #blocked8>
    %252 = arith.xori %245, %250 : tensor<1x256xi32, #blocked8>
    %253 = triton_gpu.convert_layout %arg4 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %254 = tt.reshape %253 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<64x2x2xi8, #blocked2>
    %255 = arith.muli %254, %114 : tensor<64x2x2xi8, #blocked2>
    %256 = "tt.reduce"(%255) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %257 = tt.expand_dims %256 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %258 = tt.broadcast %257 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %259 = tt.reshape %258 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %260 = arith.muli %254, %11 : tensor<64x2x2xi8, #blocked2>
    %261 = "tt.reduce"(%260) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %262 = tt.expand_dims %261 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %263 = tt.broadcast %262 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %264 = tt.reshape %263 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %265 = arith.cmpi slt, %259, %264 : tensor<1x256xi8, #blocked8>
    %266 = arith.cmpi eq, %259, %264 : tensor<1x256xi8, #blocked8>
    %267 = arith.andi %266, %251 : tensor<1x256xi1, #blocked8>
    %268 = arith.ori %265, %267 : tensor<1x256xi1, #blocked8>
    %269 = arith.extui %268 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %270 = arith.xori %269, %25 : tensor<1x256xi32, #blocked8>
    %271 = arith.cmpi ne, %270, %cst_5 : tensor<1x256xi32, #blocked8>
    %272 = arith.select %271, %252, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %273 = arith.xori %239, %272 : tensor<1x256xi32, #blocked8>
    %274 = tt.reshape %273 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<128x2x1xi32, #blocked9>
    %275 = arith.muli %274, %77 : tensor<128x2x1xi32, #blocked9>
    %276 = "tt.reduce"(%275) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %277 = tt.expand_dims %276 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %278 = tt.broadcast %277 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %279 = tt.reshape %278 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %280 = arith.muli %274, %66 : tensor<128x2x1xi32, #blocked9>
    %281 = "tt.reduce"(%280) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %282 = tt.expand_dims %281 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %283 = tt.broadcast %282 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %284 = tt.reshape %283 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %285 = arith.cmpi sgt, %279, %284 : tensor<1x256xi32, #blocked8>
    %286 = arith.xori %279, %284 : tensor<1x256xi32, #blocked8>
    %287 = triton_gpu.convert_layout %arg5 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %288 = tt.reshape %287 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<128x2x1xi8, #blocked9>
    %289 = arith.muli %288, %75 : tensor<128x2x1xi8, #blocked9>
    %290 = "tt.reduce"(%289) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %291 = tt.expand_dims %290 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %292 = tt.broadcast %291 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %293 = tt.reshape %292 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %294 = arith.muli %288, %72 : tensor<128x2x1xi8, #blocked9>
    %295 = "tt.reduce"(%294) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %296 = tt.expand_dims %295 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %297 = tt.broadcast %296 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %298 = tt.reshape %297 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %299 = arith.cmpi slt, %293, %298 : tensor<1x256xi8, #blocked8>
    %300 = arith.cmpi eq, %293, %298 : tensor<1x256xi8, #blocked8>
    %301 = arith.andi %300, %285 : tensor<1x256xi1, #blocked8>
    %302 = arith.ori %299, %301 : tensor<1x256xi1, #blocked8>
    %303 = arith.extui %302 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %304 = arith.xori %303, %25 : tensor<1x256xi32, #blocked8>
    %305 = arith.cmpi ne, %304, %cst_5 : tensor<1x256xi32, #blocked8>
    %306 = arith.select %305, %286, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %307 = arith.xori %273, %306 : tensor<1x256xi32, #blocked8>
    %308 = tt.reshape %307 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<16x2x8xi32, #blocked4>
    %309 = arith.muli %308, %106 : tensor<16x2x8xi32, #blocked4>
    %310 = "tt.reduce"(%309) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %311 = tt.expand_dims %310 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %312 = tt.broadcast %311 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %313 = tt.reshape %312 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %314 = arith.muli %308, %28 : tensor<16x2x8xi32, #blocked4>
    %315 = "tt.reduce"(%314) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %316 = tt.expand_dims %315 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %317 = tt.broadcast %316 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %318 = tt.reshape %317 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %319 = arith.cmpi sgt, %313, %318 : tensor<1x256xi32, #blocked8>
    %320 = arith.xori %313, %318 : tensor<1x256xi32, #blocked8>
    %321 = triton_gpu.convert_layout %arg6 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %322 = tt.reshape %321 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<16x2x8xi8, #blocked4>
    %323 = arith.muli %322, %104 : tensor<16x2x8xi8, #blocked4>
    %324 = "tt.reduce"(%323) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %325 = tt.expand_dims %324 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %326 = tt.broadcast %325 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %327 = tt.reshape %326 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %328 = arith.muli %322, %29 : tensor<16x2x8xi8, #blocked4>
    %329 = "tt.reduce"(%328) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %330 = tt.expand_dims %329 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %331 = tt.broadcast %330 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %332 = tt.reshape %331 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %333 = arith.cmpi slt, %327, %332 : tensor<1x256xi8, #blocked8>
    %334 = arith.cmpi eq, %327, %332 : tensor<1x256xi8, #blocked8>
    %335 = arith.andi %334, %319 : tensor<1x256xi1, #blocked8>
    %336 = arith.ori %333, %335 : tensor<1x256xi1, #blocked8>
    %337 = arith.extui %336 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %338 = arith.xori %337, %34 : tensor<1x256xi32, #blocked8>
    %339 = arith.cmpi ne, %338, %cst_5 : tensor<1x256xi32, #blocked8>
    %340 = arith.select %339, %320, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %341 = arith.xori %307, %340 : tensor<1x256xi32, #blocked8>
    %342 = tt.reshape %341 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<32x2x4xi32, #blocked3>
    %343 = arith.muli %342, %111 : tensor<32x2x4xi32, #blocked3>
    %344 = "tt.reduce"(%343) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %345 = tt.expand_dims %344 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %346 = tt.broadcast %345 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %347 = tt.reshape %346 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %348 = arith.muli %342, %19 : tensor<32x2x4xi32, #blocked3>
    %349 = "tt.reduce"(%348) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %350 = tt.expand_dims %349 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %351 = tt.broadcast %350 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %352 = tt.reshape %351 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %353 = arith.cmpi sgt, %347, %352 : tensor<1x256xi32, #blocked8>
    %354 = arith.xori %347, %352 : tensor<1x256xi32, #blocked8>
    %355 = triton_gpu.convert_layout %arg7 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %356 = tt.reshape %355 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<32x2x4xi8, #blocked3>
    %357 = arith.muli %356, %109 : tensor<32x2x4xi8, #blocked3>
    %358 = "tt.reduce"(%357) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %359 = tt.expand_dims %358 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %360 = tt.broadcast %359 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %361 = tt.reshape %360 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %362 = arith.muli %356, %20 : tensor<32x2x4xi8, #blocked3>
    %363 = "tt.reduce"(%362) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %364 = tt.expand_dims %363 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %365 = tt.broadcast %364 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %366 = tt.reshape %365 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %367 = arith.cmpi slt, %361, %366 : tensor<1x256xi8, #blocked8>
    %368 = arith.cmpi eq, %361, %366 : tensor<1x256xi8, #blocked8>
    %369 = arith.andi %368, %353 : tensor<1x256xi1, #blocked8>
    %370 = arith.ori %367, %369 : tensor<1x256xi1, #blocked8>
    %371 = arith.extui %370 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %372 = arith.xori %371, %34 : tensor<1x256xi32, #blocked8>
    %373 = arith.cmpi ne, %372, %cst_5 : tensor<1x256xi32, #blocked8>
    %374 = arith.select %373, %354, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %375 = arith.xori %341, %374 : tensor<1x256xi32, #blocked8>
    %376 = tt.reshape %375 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<64x2x2xi32, #blocked2>
    %377 = arith.muli %376, %116 : tensor<64x2x2xi32, #blocked2>
    %378 = "tt.reduce"(%377) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %379 = tt.expand_dims %378 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %380 = tt.broadcast %379 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %381 = tt.reshape %380 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %382 = arith.muli %376, %10 : tensor<64x2x2xi32, #blocked2>
    %383 = "tt.reduce"(%382) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %384 = tt.expand_dims %383 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %385 = tt.broadcast %384 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %386 = tt.reshape %385 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %387 = arith.cmpi sgt, %381, %386 : tensor<1x256xi32, #blocked8>
    %388 = arith.xori %381, %386 : tensor<1x256xi32, #blocked8>
    %389 = triton_gpu.convert_layout %arg8 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %390 = tt.reshape %389 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<64x2x2xi8, #blocked2>
    %391 = arith.muli %390, %114 : tensor<64x2x2xi8, #blocked2>
    %392 = "tt.reduce"(%391) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %393 = tt.expand_dims %392 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %394 = tt.broadcast %393 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %395 = tt.reshape %394 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %396 = arith.muli %390, %11 : tensor<64x2x2xi8, #blocked2>
    %397 = "tt.reduce"(%396) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %398 = tt.expand_dims %397 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %399 = tt.broadcast %398 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %400 = tt.reshape %399 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %401 = arith.cmpi slt, %395, %400 : tensor<1x256xi8, #blocked8>
    %402 = arith.cmpi eq, %395, %400 : tensor<1x256xi8, #blocked8>
    %403 = arith.andi %402, %387 : tensor<1x256xi1, #blocked8>
    %404 = arith.ori %401, %403 : tensor<1x256xi1, #blocked8>
    %405 = arith.extui %404 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %406 = arith.xori %405, %34 : tensor<1x256xi32, #blocked8>
    %407 = arith.cmpi ne, %406, %cst_5 : tensor<1x256xi32, #blocked8>
    %408 = arith.select %407, %388, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %409 = arith.xori %375, %408 : tensor<1x256xi32, #blocked8>
    %410 = tt.reshape %409 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<128x2x1xi32, #blocked9>
    %411 = arith.muli %410, %77 : tensor<128x2x1xi32, #blocked9>
    %412 = "tt.reduce"(%411) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %413 = tt.expand_dims %412 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %414 = tt.broadcast %413 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %415 = tt.reshape %414 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %416 = arith.muli %410, %66 : tensor<128x2x1xi32, #blocked9>
    %417 = "tt.reduce"(%416) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %418 = tt.expand_dims %417 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %419 = tt.broadcast %418 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %420 = tt.reshape %419 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %421 = arith.cmpi sgt, %415, %420 : tensor<1x256xi32, #blocked8>
    %422 = arith.xori %415, %420 : tensor<1x256xi32, #blocked8>
    %423 = triton_gpu.convert_layout %arg9 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %424 = tt.reshape %423 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<128x2x1xi8, #blocked9>
    %425 = arith.muli %424, %75 : tensor<128x2x1xi8, #blocked9>
    %426 = "tt.reduce"(%425) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %427 = tt.expand_dims %426 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %428 = tt.broadcast %427 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %429 = tt.reshape %428 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %430 = arith.muli %424, %72 : tensor<128x2x1xi8, #blocked9>
    %431 = "tt.reduce"(%430) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %432 = tt.expand_dims %431 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %433 = tt.broadcast %432 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %434 = tt.reshape %433 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %435 = arith.cmpi slt, %429, %434 : tensor<1x256xi8, #blocked8>
    %436 = arith.cmpi eq, %429, %434 : tensor<1x256xi8, #blocked8>
    %437 = arith.andi %436, %421 : tensor<1x256xi1, #blocked8>
    %438 = arith.ori %435, %437 : tensor<1x256xi1, #blocked8>
    %439 = arith.extui %438 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %440 = arith.xori %439, %34 : tensor<1x256xi32, #blocked8>
    %441 = arith.cmpi ne, %440, %cst_5 : tensor<1x256xi32, #blocked8>
    %442 = arith.select %441, %422, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %443 = arith.xori %409, %442 : tensor<1x256xi32, #blocked8>
    %444 = tt.reshape %443 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<8x2x16xi32, #blocked5>
    %445 = arith.muli %444, %101 : tensor<8x2x16xi32, #blocked5>
    %446 = "tt.reduce"(%445) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<8x2x16xi32, #blocked5>) -> tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %447 = tt.expand_dims %446 {axis = 1 : i32} : tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi32, #blocked5>
    %448 = tt.broadcast %447 : tensor<8x1x16xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %449 = tt.reshape %448 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %450 = arith.muli %444, %37 : tensor<8x2x16xi32, #blocked5>
    %451 = "tt.reduce"(%450) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<8x2x16xi32, #blocked5>) -> tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %452 = tt.expand_dims %451 {axis = 1 : i32} : tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi32, #blocked5>
    %453 = tt.broadcast %452 : tensor<8x1x16xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %454 = tt.reshape %453 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %455 = arith.cmpi sgt, %449, %454 : tensor<1x256xi32, #blocked8>
    %456 = arith.xori %449, %454 : tensor<1x256xi32, #blocked8>
    %457 = triton_gpu.convert_layout %arg10 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %458 = tt.reshape %457 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<8x2x16xi8, #blocked5>
    %459 = arith.muli %458, %99 : tensor<8x2x16xi8, #blocked5>
    %460 = "tt.reduce"(%459) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<8x2x16xi8, #blocked5>) -> tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %461 = tt.expand_dims %460 {axis = 1 : i32} : tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi8, #blocked5>
    %462 = tt.broadcast %461 : tensor<8x1x16xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %463 = tt.reshape %462 {allow_reorder = false} : tensor<8x2x16xi8, #blocked5> -> tensor<1x256xi8, #blocked8>
    %464 = arith.muli %458, %38 : tensor<8x2x16xi8, #blocked5>
    %465 = "tt.reduce"(%464) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<8x2x16xi8, #blocked5>) -> tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %466 = tt.expand_dims %465 {axis = 1 : i32} : tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi8, #blocked5>
    %467 = tt.broadcast %466 : tensor<8x1x16xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %468 = tt.reshape %467 {allow_reorder = false} : tensor<8x2x16xi8, #blocked5> -> tensor<1x256xi8, #blocked8>
    %469 = arith.cmpi slt, %463, %468 : tensor<1x256xi8, #blocked8>
    %470 = arith.cmpi eq, %463, %468 : tensor<1x256xi8, #blocked8>
    %471 = arith.andi %470, %455 : tensor<1x256xi1, #blocked8>
    %472 = arith.ori %469, %471 : tensor<1x256xi1, #blocked8>
    %473 = arith.extui %472 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %474 = arith.xori %473, %43 : tensor<1x256xi32, #blocked8>
    %475 = arith.cmpi ne, %474, %cst_5 : tensor<1x256xi32, #blocked8>
    %476 = arith.select %475, %456, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %477 = arith.xori %443, %476 : tensor<1x256xi32, #blocked8>
    %478 = tt.reshape %477 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<16x2x8xi32, #blocked4>
    %479 = arith.muli %478, %106 : tensor<16x2x8xi32, #blocked4>
    %480 = "tt.reduce"(%479) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %481 = tt.expand_dims %480 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %482 = tt.broadcast %481 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %483 = tt.reshape %482 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %484 = arith.muli %478, %28 : tensor<16x2x8xi32, #blocked4>
    %485 = "tt.reduce"(%484) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %486 = tt.expand_dims %485 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %487 = tt.broadcast %486 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %488 = tt.reshape %487 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %489 = arith.cmpi sgt, %483, %488 : tensor<1x256xi32, #blocked8>
    %490 = arith.xori %483, %488 : tensor<1x256xi32, #blocked8>
    %491 = triton_gpu.convert_layout %arg11 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %492 = tt.reshape %491 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<16x2x8xi8, #blocked4>
    %493 = arith.muli %492, %104 : tensor<16x2x8xi8, #blocked4>
    %494 = "tt.reduce"(%493) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %495 = tt.expand_dims %494 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %496 = tt.broadcast %495 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %497 = tt.reshape %496 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %498 = arith.muli %492, %29 : tensor<16x2x8xi8, #blocked4>
    %499 = "tt.reduce"(%498) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %500 = tt.expand_dims %499 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %501 = tt.broadcast %500 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %502 = tt.reshape %501 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %503 = arith.cmpi slt, %497, %502 : tensor<1x256xi8, #blocked8>
    %504 = arith.cmpi eq, %497, %502 : tensor<1x256xi8, #blocked8>
    %505 = arith.andi %504, %489 : tensor<1x256xi1, #blocked8>
    %506 = arith.ori %503, %505 : tensor<1x256xi1, #blocked8>
    %507 = arith.extui %506 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %508 = arith.xori %507, %43 : tensor<1x256xi32, #blocked8>
    %509 = arith.cmpi ne, %508, %cst_5 : tensor<1x256xi32, #blocked8>
    %510 = arith.select %509, %490, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %511 = arith.xori %477, %510 : tensor<1x256xi32, #blocked8>
    %512 = tt.reshape %511 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<32x2x4xi32, #blocked3>
    %513 = arith.muli %512, %111 : tensor<32x2x4xi32, #blocked3>
    %514 = "tt.reduce"(%513) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %515 = tt.expand_dims %514 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %516 = tt.broadcast %515 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %517 = tt.reshape %516 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %518 = arith.muli %512, %19 : tensor<32x2x4xi32, #blocked3>
    %519 = "tt.reduce"(%518) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %520 = tt.expand_dims %519 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %521 = tt.broadcast %520 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %522 = tt.reshape %521 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %523 = arith.cmpi sgt, %517, %522 : tensor<1x256xi32, #blocked8>
    %524 = arith.xori %517, %522 : tensor<1x256xi32, #blocked8>
    %525 = triton_gpu.convert_layout %arg12 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %526 = tt.reshape %525 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<32x2x4xi8, #blocked3>
    %527 = arith.muli %526, %109 : tensor<32x2x4xi8, #blocked3>
    %528 = "tt.reduce"(%527) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %529 = tt.expand_dims %528 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %530 = tt.broadcast %529 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %531 = tt.reshape %530 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %532 = arith.muli %526, %20 : tensor<32x2x4xi8, #blocked3>
    %533 = "tt.reduce"(%532) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %534 = tt.expand_dims %533 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %535 = tt.broadcast %534 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %536 = tt.reshape %535 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %537 = arith.cmpi slt, %531, %536 : tensor<1x256xi8, #blocked8>
    %538 = arith.cmpi eq, %531, %536 : tensor<1x256xi8, #blocked8>
    %539 = arith.andi %538, %523 : tensor<1x256xi1, #blocked8>
    %540 = arith.ori %537, %539 : tensor<1x256xi1, #blocked8>
    %541 = arith.extui %540 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %542 = arith.xori %541, %43 : tensor<1x256xi32, #blocked8>
    %543 = arith.cmpi ne, %542, %cst_5 : tensor<1x256xi32, #blocked8>
    %544 = arith.select %543, %524, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %545 = arith.xori %511, %544 : tensor<1x256xi32, #blocked8>
    %546 = tt.reshape %545 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<64x2x2xi32, #blocked2>
    %547 = arith.muli %546, %116 : tensor<64x2x2xi32, #blocked2>
    %548 = "tt.reduce"(%547) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %549 = tt.expand_dims %548 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %550 = tt.broadcast %549 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %551 = tt.reshape %550 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %552 = arith.muli %546, %10 : tensor<64x2x2xi32, #blocked2>
    %553 = "tt.reduce"(%552) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %554 = tt.expand_dims %553 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %555 = tt.broadcast %554 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %556 = tt.reshape %555 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %557 = arith.cmpi sgt, %551, %556 : tensor<1x256xi32, #blocked8>
    %558 = arith.xori %551, %556 : tensor<1x256xi32, #blocked8>
    %559 = triton_gpu.convert_layout %arg13 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %560 = tt.reshape %559 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<64x2x2xi8, #blocked2>
    %561 = arith.muli %560, %114 : tensor<64x2x2xi8, #blocked2>
    %562 = "tt.reduce"(%561) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %563 = tt.expand_dims %562 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %564 = tt.broadcast %563 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %565 = tt.reshape %564 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %566 = arith.muli %560, %11 : tensor<64x2x2xi8, #blocked2>
    %567 = "tt.reduce"(%566) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %568 = tt.expand_dims %567 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %569 = tt.broadcast %568 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %570 = tt.reshape %569 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %571 = arith.cmpi slt, %565, %570 : tensor<1x256xi8, #blocked8>
    %572 = arith.cmpi eq, %565, %570 : tensor<1x256xi8, #blocked8>
    %573 = arith.andi %572, %557 : tensor<1x256xi1, #blocked8>
    %574 = arith.ori %571, %573 : tensor<1x256xi1, #blocked8>
    %575 = arith.extui %574 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %576 = arith.xori %575, %43 : tensor<1x256xi32, #blocked8>
    %577 = arith.cmpi ne, %576, %cst_5 : tensor<1x256xi32, #blocked8>
    %578 = arith.select %577, %558, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %579 = arith.xori %545, %578 : tensor<1x256xi32, #blocked8>
    %580 = tt.reshape %579 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<128x2x1xi32, #blocked9>
    %581 = arith.muli %580, %77 : tensor<128x2x1xi32, #blocked9>
    %582 = "tt.reduce"(%581) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %583 = tt.expand_dims %582 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %584 = tt.broadcast %583 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %585 = tt.reshape %584 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %586 = arith.muli %580, %66 : tensor<128x2x1xi32, #blocked9>
    %587 = "tt.reduce"(%586) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %588 = tt.expand_dims %587 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %589 = tt.broadcast %588 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %590 = tt.reshape %589 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %591 = arith.cmpi sgt, %585, %590 : tensor<1x256xi32, #blocked8>
    %592 = arith.xori %585, %590 : tensor<1x256xi32, #blocked8>
    %593 = triton_gpu.convert_layout %arg14 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %594 = tt.reshape %593 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<128x2x1xi8, #blocked9>
    %595 = arith.muli %594, %75 : tensor<128x2x1xi8, #blocked9>
    %596 = "tt.reduce"(%595) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %597 = tt.expand_dims %596 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %598 = tt.broadcast %597 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %599 = tt.reshape %598 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %600 = arith.muli %594, %72 : tensor<128x2x1xi8, #blocked9>
    %601 = "tt.reduce"(%600) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %602 = tt.expand_dims %601 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %603 = tt.broadcast %602 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %604 = tt.reshape %603 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %605 = arith.cmpi slt, %599, %604 : tensor<1x256xi8, #blocked8>
    %606 = arith.cmpi eq, %599, %604 : tensor<1x256xi8, #blocked8>
    %607 = arith.andi %606, %591 : tensor<1x256xi1, #blocked8>
    %608 = arith.ori %605, %607 : tensor<1x256xi1, #blocked8>
    %609 = arith.extui %608 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %610 = arith.xori %609, %43 : tensor<1x256xi32, #blocked8>
    %611 = arith.cmpi ne, %610, %cst_5 : tensor<1x256xi32, #blocked8>
    %612 = arith.select %611, %592, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %613 = arith.xori %579, %612 : tensor<1x256xi32, #blocked8>
    %614 = tt.reshape %613 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<4x2x32xi32, #blocked6>
    %615 = arith.muli %614, %96 : tensor<4x2x32xi32, #blocked6>
    %616 = "tt.reduce"(%615) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<4x2x32xi32, #blocked6>) -> tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %617 = tt.expand_dims %616 {axis = 1 : i32} : tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi32, #blocked6>
    %618 = tt.broadcast %617 : tensor<4x1x32xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %619 = tt.reshape %618 {allow_reorder = false} : tensor<4x2x32xi32, #blocked6> -> tensor<1x256xi32, #blocked8>
    %620 = arith.muli %614, %46 : tensor<4x2x32xi32, #blocked6>
    %621 = "tt.reduce"(%620) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<4x2x32xi32, #blocked6>) -> tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %622 = tt.expand_dims %621 {axis = 1 : i32} : tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi32, #blocked6>
    %623 = tt.broadcast %622 : tensor<4x1x32xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %624 = tt.reshape %623 {allow_reorder = false} : tensor<4x2x32xi32, #blocked6> -> tensor<1x256xi32, #blocked8>
    %625 = arith.cmpi sgt, %619, %624 : tensor<1x256xi32, #blocked8>
    %626 = arith.xori %619, %624 : tensor<1x256xi32, #blocked8>
    %627 = triton_gpu.convert_layout %arg15 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %628 = tt.reshape %627 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<4x2x32xi8, #blocked6>
    %629 = arith.muli %628, %94 : tensor<4x2x32xi8, #blocked6>
    %630 = "tt.reduce"(%629) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<4x2x32xi8, #blocked6>) -> tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %631 = tt.expand_dims %630 {axis = 1 : i32} : tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi8, #blocked6>
    %632 = tt.broadcast %631 : tensor<4x1x32xi8, #blocked6> -> tensor<4x2x32xi8, #blocked6>
    %633 = tt.reshape %632 {allow_reorder = false} : tensor<4x2x32xi8, #blocked6> -> tensor<1x256xi8, #blocked8>
    %634 = arith.muli %628, %47 : tensor<4x2x32xi8, #blocked6>
    %635 = "tt.reduce"(%634) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<4x2x32xi8, #blocked6>) -> tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %636 = tt.expand_dims %635 {axis = 1 : i32} : tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi8, #blocked6>
    %637 = tt.broadcast %636 : tensor<4x1x32xi8, #blocked6> -> tensor<4x2x32xi8, #blocked6>
    %638 = tt.reshape %637 {allow_reorder = false} : tensor<4x2x32xi8, #blocked6> -> tensor<1x256xi8, #blocked8>
    %639 = arith.cmpi slt, %633, %638 : tensor<1x256xi8, #blocked8>
    %640 = arith.cmpi eq, %633, %638 : tensor<1x256xi8, #blocked8>
    %641 = arith.andi %640, %625 : tensor<1x256xi1, #blocked8>
    %642 = arith.ori %639, %641 : tensor<1x256xi1, #blocked8>
    %643 = arith.extui %642 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %644 = arith.xori %643, %52 : tensor<1x256xi32, #blocked8>
    %645 = arith.cmpi ne, %644, %cst_5 : tensor<1x256xi32, #blocked8>
    %646 = arith.select %645, %626, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %647 = arith.xori %613, %646 : tensor<1x256xi32, #blocked8>
    %648 = tt.reshape %647 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<8x2x16xi32, #blocked5>
    %649 = arith.muli %648, %101 : tensor<8x2x16xi32, #blocked5>
    %650 = "tt.reduce"(%649) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<8x2x16xi32, #blocked5>) -> tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %651 = tt.expand_dims %650 {axis = 1 : i32} : tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi32, #blocked5>
    %652 = tt.broadcast %651 : tensor<8x1x16xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %653 = tt.reshape %652 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %654 = arith.muli %648, %37 : tensor<8x2x16xi32, #blocked5>
    %655 = "tt.reduce"(%654) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<8x2x16xi32, #blocked5>) -> tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %656 = tt.expand_dims %655 {axis = 1 : i32} : tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi32, #blocked5>
    %657 = tt.broadcast %656 : tensor<8x1x16xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %658 = tt.reshape %657 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %659 = arith.cmpi sgt, %653, %658 : tensor<1x256xi32, #blocked8>
    %660 = arith.xori %653, %658 : tensor<1x256xi32, #blocked8>
    %661 = triton_gpu.convert_layout %arg16 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %662 = tt.reshape %661 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<8x2x16xi8, #blocked5>
    %663 = arith.muli %662, %99 : tensor<8x2x16xi8, #blocked5>
    %664 = "tt.reduce"(%663) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<8x2x16xi8, #blocked5>) -> tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %665 = tt.expand_dims %664 {axis = 1 : i32} : tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi8, #blocked5>
    %666 = tt.broadcast %665 : tensor<8x1x16xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %667 = tt.reshape %666 {allow_reorder = false} : tensor<8x2x16xi8, #blocked5> -> tensor<1x256xi8, #blocked8>
    %668 = arith.muli %662, %38 : tensor<8x2x16xi8, #blocked5>
    %669 = "tt.reduce"(%668) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<8x2x16xi8, #blocked5>) -> tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %670 = tt.expand_dims %669 {axis = 1 : i32} : tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi8, #blocked5>
    %671 = tt.broadcast %670 : tensor<8x1x16xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %672 = tt.reshape %671 {allow_reorder = false} : tensor<8x2x16xi8, #blocked5> -> tensor<1x256xi8, #blocked8>
    %673 = arith.cmpi slt, %667, %672 : tensor<1x256xi8, #blocked8>
    %674 = arith.cmpi eq, %667, %672 : tensor<1x256xi8, #blocked8>
    %675 = arith.andi %674, %659 : tensor<1x256xi1, #blocked8>
    %676 = arith.ori %673, %675 : tensor<1x256xi1, #blocked8>
    %677 = arith.extui %676 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %678 = arith.xori %677, %52 : tensor<1x256xi32, #blocked8>
    %679 = arith.cmpi ne, %678, %cst_5 : tensor<1x256xi32, #blocked8>
    %680 = arith.select %679, %660, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %681 = arith.xori %647, %680 : tensor<1x256xi32, #blocked8>
    %682 = tt.reshape %681 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<16x2x8xi32, #blocked4>
    %683 = arith.muli %682, %106 : tensor<16x2x8xi32, #blocked4>
    %684 = "tt.reduce"(%683) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %685 = tt.expand_dims %684 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %686 = tt.broadcast %685 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %687 = tt.reshape %686 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %688 = arith.muli %682, %28 : tensor<16x2x8xi32, #blocked4>
    %689 = "tt.reduce"(%688) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %690 = tt.expand_dims %689 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %691 = tt.broadcast %690 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %692 = tt.reshape %691 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %693 = arith.cmpi sgt, %687, %692 : tensor<1x256xi32, #blocked8>
    %694 = arith.xori %687, %692 : tensor<1x256xi32, #blocked8>
    %695 = triton_gpu.convert_layout %arg17 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %696 = tt.reshape %695 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<16x2x8xi8, #blocked4>
    %697 = arith.muli %696, %104 : tensor<16x2x8xi8, #blocked4>
    %698 = "tt.reduce"(%697) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %699 = tt.expand_dims %698 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %700 = tt.broadcast %699 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %701 = tt.reshape %700 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %702 = arith.muli %696, %29 : tensor<16x2x8xi8, #blocked4>
    %703 = "tt.reduce"(%702) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %704 = tt.expand_dims %703 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %705 = tt.broadcast %704 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %706 = tt.reshape %705 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %707 = arith.cmpi slt, %701, %706 : tensor<1x256xi8, #blocked8>
    %708 = arith.cmpi eq, %701, %706 : tensor<1x256xi8, #blocked8>
    %709 = arith.andi %708, %693 : tensor<1x256xi1, #blocked8>
    %710 = arith.ori %707, %709 : tensor<1x256xi1, #blocked8>
    %711 = arith.extui %710 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %712 = arith.xori %711, %52 : tensor<1x256xi32, #blocked8>
    %713 = arith.cmpi ne, %712, %cst_5 : tensor<1x256xi32, #blocked8>
    %714 = arith.select %713, %694, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %715 = arith.xori %681, %714 : tensor<1x256xi32, #blocked8>
    %716 = tt.reshape %715 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<32x2x4xi32, #blocked3>
    %717 = arith.muli %716, %111 : tensor<32x2x4xi32, #blocked3>
    %718 = "tt.reduce"(%717) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %719 = tt.expand_dims %718 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %720 = tt.broadcast %719 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %721 = tt.reshape %720 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %722 = arith.muli %716, %19 : tensor<32x2x4xi32, #blocked3>
    %723 = "tt.reduce"(%722) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %724 = tt.expand_dims %723 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %725 = tt.broadcast %724 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %726 = tt.reshape %725 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %727 = arith.cmpi sgt, %721, %726 : tensor<1x256xi32, #blocked8>
    %728 = arith.xori %721, %726 : tensor<1x256xi32, #blocked8>
    %729 = triton_gpu.convert_layout %arg18 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %730 = tt.reshape %729 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<32x2x4xi8, #blocked3>
    %731 = arith.muli %730, %109 : tensor<32x2x4xi8, #blocked3>
    %732 = "tt.reduce"(%731) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %733 = tt.expand_dims %732 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %734 = tt.broadcast %733 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %735 = tt.reshape %734 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %736 = arith.muli %730, %20 : tensor<32x2x4xi8, #blocked3>
    %737 = "tt.reduce"(%736) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %738 = tt.expand_dims %737 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %739 = tt.broadcast %738 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %740 = tt.reshape %739 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %741 = arith.cmpi slt, %735, %740 : tensor<1x256xi8, #blocked8>
    %742 = arith.cmpi eq, %735, %740 : tensor<1x256xi8, #blocked8>
    %743 = arith.andi %742, %727 : tensor<1x256xi1, #blocked8>
    %744 = arith.ori %741, %743 : tensor<1x256xi1, #blocked8>
    %745 = arith.extui %744 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %746 = arith.xori %745, %52 : tensor<1x256xi32, #blocked8>
    %747 = arith.cmpi ne, %746, %cst_5 : tensor<1x256xi32, #blocked8>
    %748 = arith.select %747, %728, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %749 = arith.xori %715, %748 : tensor<1x256xi32, #blocked8>
    %750 = tt.reshape %749 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<64x2x2xi32, #blocked2>
    %751 = arith.muli %750, %116 : tensor<64x2x2xi32, #blocked2>
    %752 = "tt.reduce"(%751) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %753 = tt.expand_dims %752 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %754 = tt.broadcast %753 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %755 = tt.reshape %754 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %756 = arith.muli %750, %10 : tensor<64x2x2xi32, #blocked2>
    %757 = "tt.reduce"(%756) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %758 = tt.expand_dims %757 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %759 = tt.broadcast %758 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %760 = tt.reshape %759 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %761 = arith.cmpi sgt, %755, %760 : tensor<1x256xi32, #blocked8>
    %762 = arith.xori %755, %760 : tensor<1x256xi32, #blocked8>
    %763 = triton_gpu.convert_layout %arg19 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %764 = tt.reshape %763 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<64x2x2xi8, #blocked2>
    %765 = arith.muli %764, %114 : tensor<64x2x2xi8, #blocked2>
    %766 = "tt.reduce"(%765) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %767 = tt.expand_dims %766 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %768 = tt.broadcast %767 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %769 = tt.reshape %768 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %770 = arith.muli %764, %11 : tensor<64x2x2xi8, #blocked2>
    %771 = "tt.reduce"(%770) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %772 = tt.expand_dims %771 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %773 = tt.broadcast %772 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %774 = tt.reshape %773 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %775 = arith.cmpi slt, %769, %774 : tensor<1x256xi8, #blocked8>
    %776 = arith.cmpi eq, %769, %774 : tensor<1x256xi8, #blocked8>
    %777 = arith.andi %776, %761 : tensor<1x256xi1, #blocked8>
    %778 = arith.ori %775, %777 : tensor<1x256xi1, #blocked8>
    %779 = arith.extui %778 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %780 = arith.xori %779, %52 : tensor<1x256xi32, #blocked8>
    %781 = arith.cmpi ne, %780, %cst_5 : tensor<1x256xi32, #blocked8>
    %782 = arith.select %781, %762, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %783 = arith.xori %749, %782 : tensor<1x256xi32, #blocked8>
    %784 = tt.reshape %783 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<128x2x1xi32, #blocked9>
    %785 = arith.muli %784, %77 : tensor<128x2x1xi32, #blocked9>
    %786 = "tt.reduce"(%785) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %787 = tt.expand_dims %786 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %788 = tt.broadcast %787 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %789 = tt.reshape %788 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %790 = arith.muli %784, %66 : tensor<128x2x1xi32, #blocked9>
    %791 = "tt.reduce"(%790) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %792 = tt.expand_dims %791 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %793 = tt.broadcast %792 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %794 = tt.reshape %793 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %795 = arith.cmpi sgt, %789, %794 : tensor<1x256xi32, #blocked8>
    %796 = arith.xori %789, %794 : tensor<1x256xi32, #blocked8>
    %797 = triton_gpu.convert_layout %arg20 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %798 = tt.reshape %797 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<128x2x1xi8, #blocked9>
    %799 = arith.muli %798, %75 : tensor<128x2x1xi8, #blocked9>
    %800 = "tt.reduce"(%799) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %801 = tt.expand_dims %800 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %802 = tt.broadcast %801 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %803 = tt.reshape %802 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %804 = arith.muli %798, %72 : tensor<128x2x1xi8, #blocked9>
    %805 = "tt.reduce"(%804) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %806 = tt.expand_dims %805 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %807 = tt.broadcast %806 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %808 = tt.reshape %807 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %809 = arith.cmpi slt, %803, %808 : tensor<1x256xi8, #blocked8>
    %810 = arith.cmpi eq, %803, %808 : tensor<1x256xi8, #blocked8>
    %811 = arith.andi %810, %795 : tensor<1x256xi1, #blocked8>
    %812 = arith.ori %809, %811 : tensor<1x256xi1, #blocked8>
    %813 = arith.extui %812 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %814 = arith.xori %813, %52 : tensor<1x256xi32, #blocked8>
    %815 = arith.cmpi ne, %814, %cst_5 : tensor<1x256xi32, #blocked8>
    %816 = arith.select %815, %796, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %817 = arith.xori %783, %816 : tensor<1x256xi32, #blocked8>
    %818 = tt.reshape %817 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<2x2x64xi32, #blocked7>
    %819 = arith.muli %818, %89 : tensor<2x2x64xi32, #blocked7>
    %820 = "tt.reduce"(%819) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<2x2x64xi32, #blocked7>) -> tensor<2x64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %821 = tt.expand_dims %820 {axis = 1 : i32} : tensor<2x64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<2x1x64xi32, #blocked7>
    %822 = tt.broadcast %821 : tensor<2x1x64xi32, #blocked7> -> tensor<2x2x64xi32, #blocked7>
    %823 = tt.reshape %822 {allow_reorder = false} : tensor<2x2x64xi32, #blocked7> -> tensor<1x256xi32, #blocked8>
    %824 = arith.muli %818, %57 : tensor<2x2x64xi32, #blocked7>
    %825 = "tt.reduce"(%824) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<2x2x64xi32, #blocked7>) -> tensor<2x64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %826 = tt.expand_dims %825 {axis = 1 : i32} : tensor<2x64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<2x1x64xi32, #blocked7>
    %827 = tt.broadcast %826 : tensor<2x1x64xi32, #blocked7> -> tensor<2x2x64xi32, #blocked7>
    %828 = tt.reshape %827 {allow_reorder = false} : tensor<2x2x64xi32, #blocked7> -> tensor<1x256xi32, #blocked8>
    %829 = arith.cmpi sgt, %823, %828 : tensor<1x256xi32, #blocked8>
    %830 = arith.xori %823, %828 : tensor<1x256xi32, #blocked8>
    %831 = triton_gpu.convert_layout %arg21 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %832 = tt.reshape %831 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<2x2x64xi8, #blocked7>
    %833 = arith.muli %832, %87 : tensor<2x2x64xi8, #blocked7>
    %834 = "tt.reduce"(%833) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<2x2x64xi8, #blocked7>) -> tensor<2x64xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %835 = tt.expand_dims %834 {axis = 1 : i32} : tensor<2x64xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<2x1x64xi8, #blocked7>
    %836 = tt.broadcast %835 : tensor<2x1x64xi8, #blocked7> -> tensor<2x2x64xi8, #blocked7>
    %837 = tt.reshape %836 {allow_reorder = false} : tensor<2x2x64xi8, #blocked7> -> tensor<1x256xi8, #blocked8>
    %838 = arith.muli %832, %59 : tensor<2x2x64xi8, #blocked7>
    %839 = "tt.reduce"(%838) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<2x2x64xi8, #blocked7>) -> tensor<2x64xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %840 = tt.expand_dims %839 {axis = 1 : i32} : tensor<2x64xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<2x1x64xi8, #blocked7>
    %841 = tt.broadcast %840 : tensor<2x1x64xi8, #blocked7> -> tensor<2x2x64xi8, #blocked7>
    %842 = tt.reshape %841 {allow_reorder = false} : tensor<2x2x64xi8, #blocked7> -> tensor<1x256xi8, #blocked8>
    %843 = arith.cmpi slt, %837, %842 : tensor<1x256xi8, #blocked8>
    %844 = arith.cmpi eq, %837, %842 : tensor<1x256xi8, #blocked8>
    %845 = arith.andi %844, %829 : tensor<1x256xi1, #blocked8>
    %846 = arith.ori %843, %845 : tensor<1x256xi1, #blocked8>
    %847 = arith.extui %846 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %848 = arith.xori %847, %54 : tensor<1x256xi32, #blocked8>
    %849 = arith.cmpi ne, %848, %cst_5 : tensor<1x256xi32, #blocked8>
    %850 = arith.select %849, %830, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %851 = arith.xori %817, %850 : tensor<1x256xi32, #blocked8>
    %852 = tt.reshape %851 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<4x2x32xi32, #blocked6>
    %853 = arith.muli %852, %96 : tensor<4x2x32xi32, #blocked6>
    %854 = "tt.reduce"(%853) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<4x2x32xi32, #blocked6>) -> tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %855 = tt.expand_dims %854 {axis = 1 : i32} : tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi32, #blocked6>
    %856 = tt.broadcast %855 : tensor<4x1x32xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %857 = tt.reshape %856 {allow_reorder = false} : tensor<4x2x32xi32, #blocked6> -> tensor<1x256xi32, #blocked8>
    %858 = arith.muli %852, %46 : tensor<4x2x32xi32, #blocked6>
    %859 = "tt.reduce"(%858) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<4x2x32xi32, #blocked6>) -> tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %860 = tt.expand_dims %859 {axis = 1 : i32} : tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi32, #blocked6>
    %861 = tt.broadcast %860 : tensor<4x1x32xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %862 = tt.reshape %861 {allow_reorder = false} : tensor<4x2x32xi32, #blocked6> -> tensor<1x256xi32, #blocked8>
    %863 = arith.cmpi sgt, %857, %862 : tensor<1x256xi32, #blocked8>
    %864 = arith.xori %857, %862 : tensor<1x256xi32, #blocked8>
    %865 = triton_gpu.convert_layout %arg22 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %866 = tt.reshape %865 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<4x2x32xi8, #blocked6>
    %867 = arith.muli %866, %94 : tensor<4x2x32xi8, #blocked6>
    %868 = "tt.reduce"(%867) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<4x2x32xi8, #blocked6>) -> tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %869 = tt.expand_dims %868 {axis = 1 : i32} : tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi8, #blocked6>
    %870 = tt.broadcast %869 : tensor<4x1x32xi8, #blocked6> -> tensor<4x2x32xi8, #blocked6>
    %871 = tt.reshape %870 {allow_reorder = false} : tensor<4x2x32xi8, #blocked6> -> tensor<1x256xi8, #blocked8>
    %872 = arith.muli %866, %47 : tensor<4x2x32xi8, #blocked6>
    %873 = "tt.reduce"(%872) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<4x2x32xi8, #blocked6>) -> tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %874 = tt.expand_dims %873 {axis = 1 : i32} : tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi8, #blocked6>
    %875 = tt.broadcast %874 : tensor<4x1x32xi8, #blocked6> -> tensor<4x2x32xi8, #blocked6>
    %876 = tt.reshape %875 {allow_reorder = false} : tensor<4x2x32xi8, #blocked6> -> tensor<1x256xi8, #blocked8>
    %877 = arith.cmpi slt, %871, %876 : tensor<1x256xi8, #blocked8>
    %878 = arith.cmpi eq, %871, %876 : tensor<1x256xi8, #blocked8>
    %879 = arith.andi %878, %863 : tensor<1x256xi1, #blocked8>
    %880 = arith.ori %877, %879 : tensor<1x256xi1, #blocked8>
    %881 = arith.extui %880 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %882 = arith.xori %881, %54 : tensor<1x256xi32, #blocked8>
    %883 = arith.cmpi ne, %882, %cst_5 : tensor<1x256xi32, #blocked8>
    %884 = arith.select %883, %864, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %885 = arith.xori %851, %884 : tensor<1x256xi32, #blocked8>
    %886 = tt.reshape %885 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<8x2x16xi32, #blocked5>
    %887 = arith.muli %886, %101 : tensor<8x2x16xi32, #blocked5>
    %888 = "tt.reduce"(%887) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<8x2x16xi32, #blocked5>) -> tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %889 = tt.expand_dims %888 {axis = 1 : i32} : tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi32, #blocked5>
    %890 = tt.broadcast %889 : tensor<8x1x16xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %891 = tt.reshape %890 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %892 = arith.muli %886, %37 : tensor<8x2x16xi32, #blocked5>
    %893 = "tt.reduce"(%892) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<8x2x16xi32, #blocked5>) -> tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %894 = tt.expand_dims %893 {axis = 1 : i32} : tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi32, #blocked5>
    %895 = tt.broadcast %894 : tensor<8x1x16xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %896 = tt.reshape %895 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %897 = arith.cmpi sgt, %891, %896 : tensor<1x256xi32, #blocked8>
    %898 = arith.xori %891, %896 : tensor<1x256xi32, #blocked8>
    %899 = triton_gpu.convert_layout %arg23 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %900 = tt.reshape %899 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<8x2x16xi8, #blocked5>
    %901 = arith.muli %900, %99 : tensor<8x2x16xi8, #blocked5>
    %902 = "tt.reduce"(%901) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<8x2x16xi8, #blocked5>) -> tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %903 = tt.expand_dims %902 {axis = 1 : i32} : tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi8, #blocked5>
    %904 = tt.broadcast %903 : tensor<8x1x16xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %905 = tt.reshape %904 {allow_reorder = false} : tensor<8x2x16xi8, #blocked5> -> tensor<1x256xi8, #blocked8>
    %906 = arith.muli %900, %38 : tensor<8x2x16xi8, #blocked5>
    %907 = "tt.reduce"(%906) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<8x2x16xi8, #blocked5>) -> tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %908 = tt.expand_dims %907 {axis = 1 : i32} : tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi8, #blocked5>
    %909 = tt.broadcast %908 : tensor<8x1x16xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %910 = tt.reshape %909 {allow_reorder = false} : tensor<8x2x16xi8, #blocked5> -> tensor<1x256xi8, #blocked8>
    %911 = arith.cmpi slt, %905, %910 : tensor<1x256xi8, #blocked8>
    %912 = arith.cmpi eq, %905, %910 : tensor<1x256xi8, #blocked8>
    %913 = arith.andi %912, %897 : tensor<1x256xi1, #blocked8>
    %914 = arith.ori %911, %913 : tensor<1x256xi1, #blocked8>
    %915 = arith.extui %914 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %916 = arith.xori %915, %54 : tensor<1x256xi32, #blocked8>
    %917 = arith.cmpi ne, %916, %cst_5 : tensor<1x256xi32, #blocked8>
    %918 = arith.select %917, %898, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %919 = arith.xori %885, %918 : tensor<1x256xi32, #blocked8>
    %920 = tt.reshape %919 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<16x2x8xi32, #blocked4>
    %921 = arith.muli %920, %106 : tensor<16x2x8xi32, #blocked4>
    %922 = "tt.reduce"(%921) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %923 = tt.expand_dims %922 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %924 = tt.broadcast %923 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %925 = tt.reshape %924 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %926 = arith.muli %920, %28 : tensor<16x2x8xi32, #blocked4>
    %927 = "tt.reduce"(%926) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %928 = tt.expand_dims %927 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %929 = tt.broadcast %928 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %930 = tt.reshape %929 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %931 = arith.cmpi sgt, %925, %930 : tensor<1x256xi32, #blocked8>
    %932 = arith.xori %925, %930 : tensor<1x256xi32, #blocked8>
    %933 = triton_gpu.convert_layout %arg24 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %934 = tt.reshape %933 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<16x2x8xi8, #blocked4>
    %935 = arith.muli %934, %104 : tensor<16x2x8xi8, #blocked4>
    %936 = "tt.reduce"(%935) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %937 = tt.expand_dims %936 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %938 = tt.broadcast %937 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %939 = tt.reshape %938 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %940 = arith.muli %934, %29 : tensor<16x2x8xi8, #blocked4>
    %941 = "tt.reduce"(%940) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %942 = tt.expand_dims %941 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %943 = tt.broadcast %942 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %944 = tt.reshape %943 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %945 = arith.cmpi slt, %939, %944 : tensor<1x256xi8, #blocked8>
    %946 = arith.cmpi eq, %939, %944 : tensor<1x256xi8, #blocked8>
    %947 = arith.andi %946, %931 : tensor<1x256xi1, #blocked8>
    %948 = arith.ori %945, %947 : tensor<1x256xi1, #blocked8>
    %949 = arith.extui %948 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %950 = arith.xori %949, %54 : tensor<1x256xi32, #blocked8>
    %951 = arith.cmpi ne, %950, %cst_5 : tensor<1x256xi32, #blocked8>
    %952 = arith.select %951, %932, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %953 = arith.xori %919, %952 : tensor<1x256xi32, #blocked8>
    %954 = tt.reshape %953 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<32x2x4xi32, #blocked3>
    %955 = arith.muli %954, %111 : tensor<32x2x4xi32, #blocked3>
    %956 = "tt.reduce"(%955) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %957 = tt.expand_dims %956 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %958 = tt.broadcast %957 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %959 = tt.reshape %958 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %960 = arith.muli %954, %19 : tensor<32x2x4xi32, #blocked3>
    %961 = "tt.reduce"(%960) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %962 = tt.expand_dims %961 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %963 = tt.broadcast %962 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %964 = tt.reshape %963 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %965 = arith.cmpi sgt, %959, %964 : tensor<1x256xi32, #blocked8>
    %966 = arith.xori %959, %964 : tensor<1x256xi32, #blocked8>
    %967 = triton_gpu.convert_layout %arg25 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %968 = tt.reshape %967 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<32x2x4xi8, #blocked3>
    %969 = arith.muli %968, %109 : tensor<32x2x4xi8, #blocked3>
    %970 = "tt.reduce"(%969) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %971 = tt.expand_dims %970 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %972 = tt.broadcast %971 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %973 = tt.reshape %972 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %974 = arith.muli %968, %20 : tensor<32x2x4xi8, #blocked3>
    %975 = "tt.reduce"(%974) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %976 = tt.expand_dims %975 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %977 = tt.broadcast %976 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %978 = tt.reshape %977 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %979 = arith.cmpi slt, %973, %978 : tensor<1x256xi8, #blocked8>
    %980 = arith.cmpi eq, %973, %978 : tensor<1x256xi8, #blocked8>
    %981 = arith.andi %980, %965 : tensor<1x256xi1, #blocked8>
    %982 = arith.ori %979, %981 : tensor<1x256xi1, #blocked8>
    %983 = arith.extui %982 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %984 = arith.xori %983, %54 : tensor<1x256xi32, #blocked8>
    %985 = arith.cmpi ne, %984, %cst_5 : tensor<1x256xi32, #blocked8>
    %986 = arith.select %985, %966, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %987 = arith.xori %953, %986 : tensor<1x256xi32, #blocked8>
    %988 = tt.reshape %987 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<64x2x2xi32, #blocked2>
    %989 = arith.muli %988, %116 : tensor<64x2x2xi32, #blocked2>
    %990 = "tt.reduce"(%989) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %991 = tt.expand_dims %990 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %992 = tt.broadcast %991 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %993 = tt.reshape %992 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %994 = arith.muli %988, %10 : tensor<64x2x2xi32, #blocked2>
    %995 = "tt.reduce"(%994) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %996 = tt.expand_dims %995 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %997 = tt.broadcast %996 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %998 = tt.reshape %997 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %999 = arith.cmpi sgt, %993, %998 : tensor<1x256xi32, #blocked8>
    %1000 = arith.xori %993, %998 : tensor<1x256xi32, #blocked8>
    %1001 = triton_gpu.convert_layout %arg26 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1002 = tt.reshape %1001 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<64x2x2xi8, #blocked2>
    %1003 = arith.muli %1002, %114 : tensor<64x2x2xi8, #blocked2>
    %1004 = "tt.reduce"(%1003) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %1005 = tt.expand_dims %1004 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %1006 = tt.broadcast %1005 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %1007 = tt.reshape %1006 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %1008 = arith.muli %1002, %11 : tensor<64x2x2xi8, #blocked2>
    %1009 = "tt.reduce"(%1008) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %1010 = tt.expand_dims %1009 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %1011 = tt.broadcast %1010 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %1012 = tt.reshape %1011 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %1013 = arith.cmpi slt, %1007, %1012 : tensor<1x256xi8, #blocked8>
    %1014 = arith.cmpi eq, %1007, %1012 : tensor<1x256xi8, #blocked8>
    %1015 = arith.andi %1014, %999 : tensor<1x256xi1, #blocked8>
    %1016 = arith.ori %1013, %1015 : tensor<1x256xi1, #blocked8>
    %1017 = arith.extui %1016 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %1018 = arith.xori %1017, %54 : tensor<1x256xi32, #blocked8>
    %1019 = arith.cmpi ne, %1018, %cst_5 : tensor<1x256xi32, #blocked8>
    %1020 = arith.select %1019, %1000, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1021 = arith.xori %987, %1020 : tensor<1x256xi32, #blocked8>
    %1022 = tt.reshape %1021 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<128x2x1xi32, #blocked9>
    %1023 = arith.muli %1022, %77 : tensor<128x2x1xi32, #blocked9>
    %1024 = "tt.reduce"(%1023) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %1025 = tt.expand_dims %1024 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %1026 = tt.broadcast %1025 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %1027 = tt.reshape %1026 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %1028 = arith.muli %1022, %66 : tensor<128x2x1xi32, #blocked9>
    %1029 = "tt.reduce"(%1028) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %1030 = tt.expand_dims %1029 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %1031 = tt.broadcast %1030 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %1032 = tt.reshape %1031 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %1033 = arith.cmpi sgt, %1027, %1032 : tensor<1x256xi32, #blocked8>
    %1034 = arith.xori %1027, %1032 : tensor<1x256xi32, #blocked8>
    %1035 = triton_gpu.convert_layout %arg27 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1036 = tt.reshape %1035 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<128x2x1xi8, #blocked9>
    %1037 = arith.muli %1036, %75 : tensor<128x2x1xi8, #blocked9>
    %1038 = "tt.reduce"(%1037) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %1039 = tt.expand_dims %1038 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %1040 = tt.broadcast %1039 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %1041 = tt.reshape %1040 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %1042 = arith.muli %1036, %72 : tensor<128x2x1xi8, #blocked9>
    %1043 = "tt.reduce"(%1042) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %1044 = tt.expand_dims %1043 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %1045 = tt.broadcast %1044 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %1046 = tt.reshape %1045 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %1047 = arith.cmpi slt, %1041, %1046 : tensor<1x256xi8, #blocked8>
    %1048 = arith.cmpi eq, %1041, %1046 : tensor<1x256xi8, #blocked8>
    %1049 = arith.andi %1048, %1033 : tensor<1x256xi1, #blocked8>
    %1050 = arith.ori %1047, %1049 : tensor<1x256xi1, #blocked8>
    %1051 = arith.extui %1050 : tensor<1x256xi1, #blocked8> to tensor<1x256xi32, #blocked8>
    %1052 = arith.xori %1051, %54 : tensor<1x256xi32, #blocked8>
    %1053 = arith.cmpi ne, %1052, %cst_5 : tensor<1x256xi32, #blocked8>
    %1054 = arith.select %1053, %1034, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1055 = arith.xori %1021, %1054 : tensor<1x256xi32, #blocked8>
    %1056 = tt.reshape %1055 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<1x2x128xi32, #blocked7>
    %1057 = arith.muli %1056, %90 : tensor<1x2x128xi32, #blocked7>
    %1058 = "tt.reduce"(%1057) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<1x2x128xi32, #blocked7>) -> tensor<1x128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %1059 = tt.expand_dims %1058 {axis = 1 : i32} : tensor<1x128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<1x1x128xi32, #blocked7>
    %1060 = tt.broadcast %1059 : tensor<1x1x128xi32, #blocked7> -> tensor<1x2x128xi32, #blocked7>
    %1061 = tt.reshape %1060 {allow_reorder = false} : tensor<1x2x128xi32, #blocked7> -> tensor<1x256xi32, #blocked8>
    %1062 = arith.muli %1056, %58 : tensor<1x2x128xi32, #blocked7>
    %1063 = "tt.reduce"(%1062) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<1x2x128xi32, #blocked7>) -> tensor<1x128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %1064 = tt.expand_dims %1063 {axis = 1 : i32} : tensor<1x128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<1x1x128xi32, #blocked7>
    %1065 = tt.broadcast %1064 : tensor<1x1x128xi32, #blocked7> -> tensor<1x2x128xi32, #blocked7>
    %1066 = tt.reshape %1065 {allow_reorder = false} : tensor<1x2x128xi32, #blocked7> -> tensor<1x256xi32, #blocked8>
    %1067 = arith.cmpi sgt, %1061, %1066 : tensor<1x256xi32, #blocked8>
    %1068 = arith.xori %1061, %1066 : tensor<1x256xi32, #blocked8>
    %1069 = triton_gpu.convert_layout %arg28 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1070 = tt.reshape %1069 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<1x2x128xi8, #blocked7>
    %1071 = arith.muli %1070, %91 : tensor<1x2x128xi8, #blocked7>
    %1072 = "tt.reduce"(%1071) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<1x2x128xi8, #blocked7>) -> tensor<1x128xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %1073 = tt.expand_dims %1072 {axis = 1 : i32} : tensor<1x128xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<1x1x128xi8, #blocked7>
    %1074 = tt.broadcast %1073 : tensor<1x1x128xi8, #blocked7> -> tensor<1x2x128xi8, #blocked7>
    %1075 = tt.reshape %1074 {allow_reorder = false} : tensor<1x2x128xi8, #blocked7> -> tensor<1x256xi8, #blocked8>
    %1076 = arith.muli %1070, %60 : tensor<1x2x128xi8, #blocked7>
    %1077 = "tt.reduce"(%1076) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<1x2x128xi8, #blocked7>) -> tensor<1x128xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %1078 = tt.expand_dims %1077 {axis = 1 : i32} : tensor<1x128xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<1x1x128xi8, #blocked7>
    %1079 = tt.broadcast %1078 : tensor<1x1x128xi8, #blocked7> -> tensor<1x2x128xi8, #blocked7>
    %1080 = tt.reshape %1079 {allow_reorder = false} : tensor<1x2x128xi8, #blocked7> -> tensor<1x256xi8, #blocked8>
    %1081 = arith.cmpi slt, %1075, %1080 : tensor<1x256xi8, #blocked8>
    %1082 = arith.cmpi eq, %1075, %1080 : tensor<1x256xi8, #blocked8>
    %1083 = arith.andi %1082, %1067 : tensor<1x256xi1, #blocked8>
    %1084 = arith.ori %1081, %1083 : tensor<1x256xi1, #blocked8>
    %1085 = arith.select %1084, %1068, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1086 = arith.xori %1055, %1085 : tensor<1x256xi32, #blocked8>
    %1087 = tt.reshape %1086 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<2x2x64xi32, #blocked7>
    %1088 = arith.muli %1087, %89 : tensor<2x2x64xi32, #blocked7>
    %1089 = "tt.reduce"(%1088) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<2x2x64xi32, #blocked7>) -> tensor<2x64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %1090 = tt.expand_dims %1089 {axis = 1 : i32} : tensor<2x64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<2x1x64xi32, #blocked7>
    %1091 = tt.broadcast %1090 : tensor<2x1x64xi32, #blocked7> -> tensor<2x2x64xi32, #blocked7>
    %1092 = tt.reshape %1091 {allow_reorder = false} : tensor<2x2x64xi32, #blocked7> -> tensor<1x256xi32, #blocked8>
    %1093 = arith.muli %1087, %57 : tensor<2x2x64xi32, #blocked7>
    %1094 = "tt.reduce"(%1093) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<2x2x64xi32, #blocked7>) -> tensor<2x64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %1095 = tt.expand_dims %1094 {axis = 1 : i32} : tensor<2x64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<2x1x64xi32, #blocked7>
    %1096 = tt.broadcast %1095 : tensor<2x1x64xi32, #blocked7> -> tensor<2x2x64xi32, #blocked7>
    %1097 = tt.reshape %1096 {allow_reorder = false} : tensor<2x2x64xi32, #blocked7> -> tensor<1x256xi32, #blocked8>
    %1098 = arith.cmpi sgt, %1092, %1097 : tensor<1x256xi32, #blocked8>
    %1099 = arith.xori %1092, %1097 : tensor<1x256xi32, #blocked8>
    %1100 = triton_gpu.convert_layout %arg29 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1101 = tt.reshape %1100 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<2x2x64xi8, #blocked7>
    %1102 = arith.muli %1101, %87 : tensor<2x2x64xi8, #blocked7>
    %1103 = "tt.reduce"(%1102) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<2x2x64xi8, #blocked7>) -> tensor<2x64xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %1104 = tt.expand_dims %1103 {axis = 1 : i32} : tensor<2x64xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<2x1x64xi8, #blocked7>
    %1105 = tt.broadcast %1104 : tensor<2x1x64xi8, #blocked7> -> tensor<2x2x64xi8, #blocked7>
    %1106 = tt.reshape %1105 {allow_reorder = false} : tensor<2x2x64xi8, #blocked7> -> tensor<1x256xi8, #blocked8>
    %1107 = arith.muli %1101, %59 : tensor<2x2x64xi8, #blocked7>
    %1108 = "tt.reduce"(%1107) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<2x2x64xi8, #blocked7>) -> tensor<2x64xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
    %1109 = tt.expand_dims %1108 {axis = 1 : i32} : tensor<2x64xi8, #triton_gpu.slice<{dim = 1, parent = #blocked7}>> -> tensor<2x1x64xi8, #blocked7>
    %1110 = tt.broadcast %1109 : tensor<2x1x64xi8, #blocked7> -> tensor<2x2x64xi8, #blocked7>
    %1111 = tt.reshape %1110 {allow_reorder = false} : tensor<2x2x64xi8, #blocked7> -> tensor<1x256xi8, #blocked8>
    %1112 = arith.cmpi slt, %1106, %1111 : tensor<1x256xi8, #blocked8>
    %1113 = arith.cmpi eq, %1106, %1111 : tensor<1x256xi8, #blocked8>
    %1114 = arith.andi %1113, %1098 : tensor<1x256xi1, #blocked8>
    %1115 = arith.ori %1112, %1114 : tensor<1x256xi1, #blocked8>
    %1116 = arith.select %1115, %1099, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1117 = arith.xori %1086, %1116 : tensor<1x256xi32, #blocked8>
    %1118 = tt.reshape %1117 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<4x2x32xi32, #blocked6>
    %1119 = arith.muli %1118, %96 : tensor<4x2x32xi32, #blocked6>
    %1120 = "tt.reduce"(%1119) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<4x2x32xi32, #blocked6>) -> tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %1121 = tt.expand_dims %1120 {axis = 1 : i32} : tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi32, #blocked6>
    %1122 = tt.broadcast %1121 : tensor<4x1x32xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %1123 = tt.reshape %1122 {allow_reorder = false} : tensor<4x2x32xi32, #blocked6> -> tensor<1x256xi32, #blocked8>
    %1124 = arith.muli %1118, %46 : tensor<4x2x32xi32, #blocked6>
    %1125 = "tt.reduce"(%1124) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<4x2x32xi32, #blocked6>) -> tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %1126 = tt.expand_dims %1125 {axis = 1 : i32} : tensor<4x32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi32, #blocked6>
    %1127 = tt.broadcast %1126 : tensor<4x1x32xi32, #blocked6> -> tensor<4x2x32xi32, #blocked6>
    %1128 = tt.reshape %1127 {allow_reorder = false} : tensor<4x2x32xi32, #blocked6> -> tensor<1x256xi32, #blocked8>
    %1129 = arith.cmpi sgt, %1123, %1128 : tensor<1x256xi32, #blocked8>
    %1130 = arith.xori %1123, %1128 : tensor<1x256xi32, #blocked8>
    %1131 = triton_gpu.convert_layout %arg30 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1132 = tt.reshape %1131 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<4x2x32xi8, #blocked6>
    %1133 = arith.muli %1132, %94 : tensor<4x2x32xi8, #blocked6>
    %1134 = "tt.reduce"(%1133) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<4x2x32xi8, #blocked6>) -> tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %1135 = tt.expand_dims %1134 {axis = 1 : i32} : tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi8, #blocked6>
    %1136 = tt.broadcast %1135 : tensor<4x1x32xi8, #blocked6> -> tensor<4x2x32xi8, #blocked6>
    %1137 = tt.reshape %1136 {allow_reorder = false} : tensor<4x2x32xi8, #blocked6> -> tensor<1x256xi8, #blocked8>
    %1138 = arith.muli %1132, %47 : tensor<4x2x32xi8, #blocked6>
    %1139 = "tt.reduce"(%1138) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<4x2x32xi8, #blocked6>) -> tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %1140 = tt.expand_dims %1139 {axis = 1 : i32} : tensor<4x32xi8, #triton_gpu.slice<{dim = 1, parent = #blocked6}>> -> tensor<4x1x32xi8, #blocked6>
    %1141 = tt.broadcast %1140 : tensor<4x1x32xi8, #blocked6> -> tensor<4x2x32xi8, #blocked6>
    %1142 = tt.reshape %1141 {allow_reorder = false} : tensor<4x2x32xi8, #blocked6> -> tensor<1x256xi8, #blocked8>
    %1143 = arith.cmpi slt, %1137, %1142 : tensor<1x256xi8, #blocked8>
    %1144 = arith.cmpi eq, %1137, %1142 : tensor<1x256xi8, #blocked8>
    %1145 = arith.andi %1144, %1129 : tensor<1x256xi1, #blocked8>
    %1146 = arith.ori %1143, %1145 : tensor<1x256xi1, #blocked8>
    %1147 = arith.select %1146, %1130, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1148 = arith.xori %1117, %1147 : tensor<1x256xi32, #blocked8>
    %1149 = tt.reshape %1148 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<8x2x16xi32, #blocked5>
    %1150 = arith.muli %1149, %101 : tensor<8x2x16xi32, #blocked5>
    %1151 = "tt.reduce"(%1150) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<8x2x16xi32, #blocked5>) -> tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %1152 = tt.expand_dims %1151 {axis = 1 : i32} : tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi32, #blocked5>
    %1153 = tt.broadcast %1152 : tensor<8x1x16xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %1154 = tt.reshape %1153 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %1155 = arith.muli %1149, %37 : tensor<8x2x16xi32, #blocked5>
    %1156 = "tt.reduce"(%1155) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<8x2x16xi32, #blocked5>) -> tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %1157 = tt.expand_dims %1156 {axis = 1 : i32} : tensor<8x16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi32, #blocked5>
    %1158 = tt.broadcast %1157 : tensor<8x1x16xi32, #blocked5> -> tensor<8x2x16xi32, #blocked5>
    %1159 = tt.reshape %1158 {allow_reorder = false} : tensor<8x2x16xi32, #blocked5> -> tensor<1x256xi32, #blocked8>
    %1160 = arith.cmpi sgt, %1154, %1159 : tensor<1x256xi32, #blocked8>
    %1161 = arith.xori %1154, %1159 : tensor<1x256xi32, #blocked8>
    %1162 = triton_gpu.convert_layout %arg31 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1163 = tt.reshape %1162 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<8x2x16xi8, #blocked5>
    %1164 = arith.muli %1163, %99 : tensor<8x2x16xi8, #blocked5>
    %1165 = "tt.reduce"(%1164) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<8x2x16xi8, #blocked5>) -> tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %1166 = tt.expand_dims %1165 {axis = 1 : i32} : tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi8, #blocked5>
    %1167 = tt.broadcast %1166 : tensor<8x1x16xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %1168 = tt.reshape %1167 {allow_reorder = false} : tensor<8x2x16xi8, #blocked5> -> tensor<1x256xi8, #blocked8>
    %1169 = arith.muli %1163, %38 : tensor<8x2x16xi8, #blocked5>
    %1170 = "tt.reduce"(%1169) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<8x2x16xi8, #blocked5>) -> tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %1171 = tt.expand_dims %1170 {axis = 1 : i32} : tensor<8x16xi8, #triton_gpu.slice<{dim = 1, parent = #blocked5}>> -> tensor<8x1x16xi8, #blocked5>
    %1172 = tt.broadcast %1171 : tensor<8x1x16xi8, #blocked5> -> tensor<8x2x16xi8, #blocked5>
    %1173 = tt.reshape %1172 {allow_reorder = false} : tensor<8x2x16xi8, #blocked5> -> tensor<1x256xi8, #blocked8>
    %1174 = arith.cmpi slt, %1168, %1173 : tensor<1x256xi8, #blocked8>
    %1175 = arith.cmpi eq, %1168, %1173 : tensor<1x256xi8, #blocked8>
    %1176 = arith.andi %1175, %1160 : tensor<1x256xi1, #blocked8>
    %1177 = arith.ori %1174, %1176 : tensor<1x256xi1, #blocked8>
    %1178 = arith.select %1177, %1161, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1179 = arith.xori %1148, %1178 : tensor<1x256xi32, #blocked8>
    %1180 = tt.reshape %1179 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<16x2x8xi32, #blocked4>
    %1181 = arith.muli %1180, %106 : tensor<16x2x8xi32, #blocked4>
    %1182 = "tt.reduce"(%1181) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %1183 = tt.expand_dims %1182 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %1184 = tt.broadcast %1183 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %1185 = tt.reshape %1184 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %1186 = arith.muli %1180, %28 : tensor<16x2x8xi32, #blocked4>
    %1187 = "tt.reduce"(%1186) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<16x2x8xi32, #blocked4>) -> tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %1188 = tt.expand_dims %1187 {axis = 1 : i32} : tensor<16x8xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi32, #blocked4>
    %1189 = tt.broadcast %1188 : tensor<16x1x8xi32, #blocked4> -> tensor<16x2x8xi32, #blocked4>
    %1190 = tt.reshape %1189 {allow_reorder = false} : tensor<16x2x8xi32, #blocked4> -> tensor<1x256xi32, #blocked8>
    %1191 = arith.cmpi sgt, %1185, %1190 : tensor<1x256xi32, #blocked8>
    %1192 = arith.xori %1185, %1190 : tensor<1x256xi32, #blocked8>
    %1193 = triton_gpu.convert_layout %arg32 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1194 = tt.reshape %1193 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<16x2x8xi8, #blocked4>
    %1195 = arith.muli %1194, %104 : tensor<16x2x8xi8, #blocked4>
    %1196 = "tt.reduce"(%1195) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %1197 = tt.expand_dims %1196 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %1198 = tt.broadcast %1197 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %1199 = tt.reshape %1198 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %1200 = arith.muli %1194, %29 : tensor<16x2x8xi8, #blocked4>
    %1201 = "tt.reduce"(%1200) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<16x2x8xi8, #blocked4>) -> tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %1202 = tt.expand_dims %1201 {axis = 1 : i32} : tensor<16x8xi8, #triton_gpu.slice<{dim = 1, parent = #blocked4}>> -> tensor<16x1x8xi8, #blocked4>
    %1203 = tt.broadcast %1202 : tensor<16x1x8xi8, #blocked4> -> tensor<16x2x8xi8, #blocked4>
    %1204 = tt.reshape %1203 {allow_reorder = false} : tensor<16x2x8xi8, #blocked4> -> tensor<1x256xi8, #blocked8>
    %1205 = arith.cmpi slt, %1199, %1204 : tensor<1x256xi8, #blocked8>
    %1206 = arith.cmpi eq, %1199, %1204 : tensor<1x256xi8, #blocked8>
    %1207 = arith.andi %1206, %1191 : tensor<1x256xi1, #blocked8>
    %1208 = arith.ori %1205, %1207 : tensor<1x256xi1, #blocked8>
    %1209 = arith.select %1208, %1192, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1210 = arith.xori %1179, %1209 : tensor<1x256xi32, #blocked8>
    %1211 = tt.reshape %1210 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<32x2x4xi32, #blocked3>
    %1212 = arith.muli %1211, %111 : tensor<32x2x4xi32, #blocked3>
    %1213 = "tt.reduce"(%1212) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %1214 = tt.expand_dims %1213 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %1215 = tt.broadcast %1214 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %1216 = tt.reshape %1215 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %1217 = arith.muli %1211, %19 : tensor<32x2x4xi32, #blocked3>
    %1218 = "tt.reduce"(%1217) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<32x2x4xi32, #blocked3>) -> tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %1219 = tt.expand_dims %1218 {axis = 1 : i32} : tensor<32x4xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi32, #blocked3>
    %1220 = tt.broadcast %1219 : tensor<32x1x4xi32, #blocked3> -> tensor<32x2x4xi32, #blocked3>
    %1221 = tt.reshape %1220 {allow_reorder = false} : tensor<32x2x4xi32, #blocked3> -> tensor<1x256xi32, #blocked8>
    %1222 = arith.cmpi sgt, %1216, %1221 : tensor<1x256xi32, #blocked8>
    %1223 = arith.xori %1216, %1221 : tensor<1x256xi32, #blocked8>
    %1224 = triton_gpu.convert_layout %arg33 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1225 = tt.reshape %1224 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<32x2x4xi8, #blocked3>
    %1226 = arith.muli %1225, %109 : tensor<32x2x4xi8, #blocked3>
    %1227 = "tt.reduce"(%1226) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %1228 = tt.expand_dims %1227 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %1229 = tt.broadcast %1228 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %1230 = tt.reshape %1229 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %1231 = arith.muli %1225, %20 : tensor<32x2x4xi8, #blocked3>
    %1232 = "tt.reduce"(%1231) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<32x2x4xi8, #blocked3>) -> tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %1233 = tt.expand_dims %1232 {axis = 1 : i32} : tensor<32x4xi8, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1x4xi8, #blocked3>
    %1234 = tt.broadcast %1233 : tensor<32x1x4xi8, #blocked3> -> tensor<32x2x4xi8, #blocked3>
    %1235 = tt.reshape %1234 {allow_reorder = false} : tensor<32x2x4xi8, #blocked3> -> tensor<1x256xi8, #blocked8>
    %1236 = arith.cmpi slt, %1230, %1235 : tensor<1x256xi8, #blocked8>
    %1237 = arith.cmpi eq, %1230, %1235 : tensor<1x256xi8, #blocked8>
    %1238 = arith.andi %1237, %1222 : tensor<1x256xi1, #blocked8>
    %1239 = arith.ori %1236, %1238 : tensor<1x256xi1, #blocked8>
    %1240 = arith.select %1239, %1223, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1241 = arith.xori %1210, %1240 : tensor<1x256xi32, #blocked8>
    %1242 = tt.reshape %1241 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<64x2x2xi32, #blocked2>
    %1243 = arith.muli %1242, %116 : tensor<64x2x2xi32, #blocked2>
    %1244 = "tt.reduce"(%1243) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %1245 = tt.expand_dims %1244 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %1246 = tt.broadcast %1245 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %1247 = tt.reshape %1246 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %1248 = arith.muli %1242, %10 : tensor<64x2x2xi32, #blocked2>
    %1249 = "tt.reduce"(%1248) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<64x2x2xi32, #blocked2>) -> tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %1250 = tt.expand_dims %1249 {axis = 1 : i32} : tensor<64x2xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi32, #blocked2>
    %1251 = tt.broadcast %1250 : tensor<64x1x2xi32, #blocked2> -> tensor<64x2x2xi32, #blocked2>
    %1252 = tt.reshape %1251 {allow_reorder = false} : tensor<64x2x2xi32, #blocked2> -> tensor<1x256xi32, #blocked8>
    %1253 = arith.cmpi sgt, %1247, %1252 : tensor<1x256xi32, #blocked8>
    %1254 = arith.xori %1247, %1252 : tensor<1x256xi32, #blocked8>
    %1255 = triton_gpu.convert_layout %arg34 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1256 = tt.reshape %1255 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<64x2x2xi8, #blocked2>
    %1257 = arith.muli %1256, %114 : tensor<64x2x2xi8, #blocked2>
    %1258 = "tt.reduce"(%1257) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %1259 = tt.expand_dims %1258 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %1260 = tt.broadcast %1259 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %1261 = tt.reshape %1260 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %1262 = arith.muli %1256, %11 : tensor<64x2x2xi8, #blocked2>
    %1263 = "tt.reduce"(%1262) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<64x2x2xi8, #blocked2>) -> tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %1264 = tt.expand_dims %1263 {axis = 1 : i32} : tensor<64x2xi8, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1x2xi8, #blocked2>
    %1265 = tt.broadcast %1264 : tensor<64x1x2xi8, #blocked2> -> tensor<64x2x2xi8, #blocked2>
    %1266 = tt.reshape %1265 {allow_reorder = false} : tensor<64x2x2xi8, #blocked2> -> tensor<1x256xi8, #blocked8>
    %1267 = arith.cmpi slt, %1261, %1266 : tensor<1x256xi8, #blocked8>
    %1268 = arith.cmpi eq, %1261, %1266 : tensor<1x256xi8, #blocked8>
    %1269 = arith.andi %1268, %1253 : tensor<1x256xi1, #blocked8>
    %1270 = arith.ori %1267, %1269 : tensor<1x256xi1, #blocked8>
    %1271 = arith.select %1270, %1254, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1272 = arith.xori %1241, %1271 : tensor<1x256xi32, #blocked8>
    %1273 = tt.reshape %1272 {allow_reorder = false} : tensor<1x256xi32, #blocked8> -> tensor<128x2x1xi32, #blocked9>
    %1274 = arith.muli %1273, %77 : tensor<128x2x1xi32, #blocked9>
    %1275 = "tt.reduce"(%1274) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %1276 = tt.expand_dims %1275 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %1277 = tt.broadcast %1276 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %1278 = tt.reshape %1277 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %1279 = arith.muli %1273, %66 : tensor<128x2x1xi32, #blocked9>
    %1280 = "tt.reduce"(%1279) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i32, %arg37: i32):
      %1305 = arith.addi %arg36, %arg37 : i32
      tt.reduce.return %1305 : i32
    }) : (tensor<128x2x1xi32, #blocked9>) -> tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %1281 = tt.expand_dims %1280 {axis = 1 : i32} : tensor<128x1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi32, #blocked9>
    %1282 = tt.broadcast %1281 : tensor<128x1x1xi32, #blocked9> -> tensor<128x2x1xi32, #blocked9>
    %1283 = tt.reshape %1282 {allow_reorder = false} : tensor<128x2x1xi32, #blocked9> -> tensor<1x256xi32, #blocked8>
    %1284 = arith.cmpi sgt, %1278, %1283 : tensor<1x256xi32, #blocked8>
    %1285 = arith.xori %1278, %1283 : tensor<1x256xi32, #blocked8>
    %1286 = triton_gpu.convert_layout %arg35 : tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked8>
    %1287 = tt.reshape %1286 {allow_reorder = false} : tensor<1x256xi8, #blocked8> -> tensor<128x2x1xi8, #blocked9>
    %1288 = arith.muli %1287, %75 : tensor<128x2x1xi8, #blocked9>
    %1289 = "tt.reduce"(%1288) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %1290 = tt.expand_dims %1289 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %1291 = tt.broadcast %1290 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %1292 = tt.reshape %1291 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %1293 = arith.muli %1287, %72 : tensor<128x2x1xi8, #blocked9>
    %1294 = "tt.reduce"(%1293) <{axis = 1 : i32}> ({
    ^bb0(%arg36: i8, %arg37: i8):
      %1305 = arith.addi %arg36, %arg37 : i8
      tt.reduce.return %1305 : i8
    }) : (tensor<128x2x1xi8, #blocked9>) -> tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %1295 = tt.expand_dims %1294 {axis = 1 : i32} : tensor<128x1xi8, #triton_gpu.slice<{dim = 1, parent = #blocked9}>> -> tensor<128x1x1xi8, #blocked9>
    %1296 = tt.broadcast %1295 : tensor<128x1x1xi8, #blocked9> -> tensor<128x2x1xi8, #blocked9>
    %1297 = tt.reshape %1296 {allow_reorder = false} : tensor<128x2x1xi8, #blocked9> -> tensor<1x256xi8, #blocked8>
    %1298 = arith.cmpi slt, %1292, %1297 : tensor<1x256xi8, #blocked8>
    %1299 = arith.cmpi eq, %1292, %1297 : tensor<1x256xi8, #blocked8>
    %1300 = arith.andi %1299, %1284 : tensor<1x256xi1, #blocked8>
    %1301 = arith.ori %1298, %1300 : tensor<1x256xi1, #blocked8>
    %1302 = arith.select %1301, %1285, %cst_5 : tensor<1x256xi1, #blocked8>, tensor<1x256xi32, #blocked8>
    %1303 = arith.xori %1272, %1302 : tensor<1x256xi32, #blocked8>
    %1304 = triton_gpu.convert_layout %1303 : tensor<1x256xi32, #blocked8> -> tensor<1x256xi32, #blocked1>
    tt.return %1304 : tensor<1x256xi32, #blocked1>
  }
}
