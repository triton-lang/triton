// RUN: triton-opt %s -split-input-file --tritongpu-remove-layout-conversions -canonicalize | FileCheck %s
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: remove_layout_multiple_outputs
  tt.func public @remove_layout_multiple_outputs(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %second_reduce_input = arith.constant dense<9223372036854775807> : tensor<256x256xi64, #blocked>
    %load_mask = arith.constant dense<1>: tensor<1x256xi1, #blocked>
    %store_mask = arith.constant dense<1>: tensor<256x1xi1, #blocked1>
    %default_load_val = arith.constant dense<0.000000e+00> : tensor<256x256xf16, #blocked>
    %70 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<256x256x!tt.ptr<f16, 1>, #blocked>
    %76 = tt.broadcast %load_mask : (tensor<1x256xi1, #blocked>) -> tensor<256x256xi1, #blocked>
    %87 = tt.load %70, %76, %default_load_val {cache = 1 : i32, evict = 2 : i32, isVolatile = false} : tensor<256x256xf16, #blocked>
    %88 = triton_gpu.convert_layout %87 : (tensor<256x256xf16, #blocked>) -> tensor<256x256xf16, #blocked1>
    %89 = arith.extf %87 : tensor<256x256xf16, #blocked> to tensor<256x256xf32, #blocked>
    %108:2 = "tt.reduce"(%89, %second_reduce_input) <{axis = 1 : i32}> ({
    ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: i64):
      tt.reduce.return %arg7, %arg6 : f32, i64
    }) : (tensor<256x256xf32, #blocked>, tensor<256x256xi64, #blocked>) -> (tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>)
    %111 = tt.splat %arg1 : (!tt.ptr<i64, 1>) -> tensor<256x1x!tt.ptr<i64, 1>, #blocked1>
    %109 = tt.expand_dims %108#1 {axis = 1 : i32} : (tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<256x1xi64, #blocked>
    %110 = triton_gpu.convert_layout %109 : (tensor<256x1xi64, #blocked>) -> tensor<256x1xi64, #blocked1>
    tt.store %111, %110, %store_mask {cache = 1 : i32, evict = 1 : i32} : tensor<256x1xi64, #blocked1>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: make_range_layout_inference
  tt.func public @make_range_layout_inference(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst_5 = arith.constant dense<1> : tensor<128x1xi1, #blocked>
    %cst_7 = arith.constant dense<1> : tensor<128x4xi1, #blocked>
    %cst_8 = arith.constant dense<128> : tensor<128x1xi32, #blocked>
    %cst_9 = arith.constant dense<1.1> : tensor<128x1xf32, #blocked>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    %5 = triton_gpu.convert_layout %4 : (tensor<128x1xi32, #blocked2>) -> tensor<128x1xi32, #blocked>
    %18 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<128x4x!tt.ptr<f32, 1>, #blocked>
    %24 = triton_gpu.convert_layout %18 : (tensor<128x4x!tt.ptr<f32, 1>, #blocked>) -> tensor<128x4x!tt.ptr<f32, 1>, #blocked2>
    %25 = triton_gpu.convert_layout %cst_7 : (tensor<128x4xi1, #blocked>) -> tensor<128x4xi1, #blocked2>
    %27 = tt.load %24, %25 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x4xf32, #blocked2>
    %28 = triton_gpu.convert_layout %27 : (tensor<128x4xf32, #blocked2>) -> tensor<128x4xf32, #blocked>
    %48 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<128x1x!tt.ptr<f32, 1>, #blocked>
    %49 = tt.addptr %48, %5 : tensor<128x1x!tt.ptr<f32, 1>, #blocked>, tensor<128x1xi32, #blocked>
    %52 = tt.load %49, %cst_5 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<128x1xf32, #blocked>
    %56:1 = "tt.reduce"(%28) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      tt.reduce.return %arg3 : f32
    }) : (tensor<128x4xf32, #blocked>) -> (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>)
    %60 = tt.expand_dims %56#0 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xf32, #blocked>
    %74 = arith.addf %60, %52 : tensor<128x1xf32, #blocked>
    tt.store %48, %74 {cache = 1 : i32, evict = 1 : i32} : tensor<128x1xf32, #blocked>
    tt.return
  }
}
