// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -cse | FileCheck %s

// Regression test for issue #10987. The input is the real IR that enters
// remove-layout-conversions for a chained `x / s / s` feeding a reduction
// (both divisions in the distributed #blocked layout). The pass must NOT push
// the second division into the replicated #linear layout (lane/warp bases all
// zero), which materializes the whole tensor per thread and blows up codegen
// (4128 div.full.f32 instead of ~64).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_div_reduce
  // CHECK-NOT: arith.divf{{.*}}#linear
  tt.func public @chained_div_reduce(%a: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %cst = arith.constant dense<256> : tensor<16x1xi32, #blocked>
    %i0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %i0_0 = tt.expand_dims %i0 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %i1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %i1_1 = tt.expand_dims %i1 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %x = arith.muli %i0_0, %cst : tensor<16x1xi32, #blocked>
    %x_2 = tt.splat %a : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %x_3 = tt.addptr %x_2, %x : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %x_4 = tt.broadcast %x_3 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x256x!tt.ptr<f32>, #blocked>
    %x_5 = tt.broadcast %i1_1 : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    %x_6 = tt.addptr %x_4, %x_5 : tensor<16x256x!tt.ptr<f32>, #blocked>, tensor<16x256xi32, #blocked>
    %x_7 = tt.load %x_6 : tensor<16x256x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x_7) <{axis = 1 : i32}> ({
    ^bb0(%s_15: f32, %s_16: f32):
      %s_17 = arith.addf %s_15, %s_16 : f32
      tt.reduce.return %s_17 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %s_8 = tt.reshape %s : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #linear>
    %y = ttg.convert_layout %s_8 : tensor<16x1xf32, #linear> -> tensor<16x1xf32, #blocked>
    %y_9 = tt.broadcast %y : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    %y_10 = arith.divf %x_7, %y_9 : tensor<16x256xf32, #blocked>
    %w = arith.divf %y_10, %y_9 : tensor<16x256xf32, #blocked>
    %z = "tt.reduce"(%w) <{axis = 1 : i32}> ({
    ^bb0(%z_15: f32, %z_16: f32):
      %z_17 = arith.addf %z_15, %z_16 : f32
      tt.reduce.return %z_17 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %z_11 = tt.reshape %z : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #linear>
    %z_12 = ttg.convert_layout %z_11 : tensor<16x1xf32, #linear> -> tensor<16x1xf32, #blocked>
    %z_13 = tt.broadcast %z_12 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    %z_14 = arith.addf %z_13, %w : tensor<16x256xf32, #blocked>
    %0 = tt.splat %out : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %1 = tt.addptr %0, %x : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x256x!tt.ptr<f32>, #blocked>
    %3 = tt.addptr %2, %x_5 : tensor<16x256x!tt.ptr<f32>, #blocked>, tensor<16x256xi32, #blocked>
    tt.store %3, %z_14 : tensor<16x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
