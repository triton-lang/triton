// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline="num_stages=4 use_async_copy=1" -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @direct_chained_dots

  // We have no ops between the dots so we just check that dot and memory ops are in the correct order and check if basic pipelining (prologue, epilogue) is working correctly.
  // CHECK-COUNT-2: ttg.local_load
  // CHECK: scf.for
  // CHECK: tt.dot
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: tt.dot
  // CHECK: ttg.async_wait
  // CHECK: ttg.local_load
  // CHECK: scf.yield
  // CHECK: ttg.async_wait
  // CHECK: ttg.local_load

  tt.func @direct_chained_dots(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg3: i32, %arg4: i32) -> tensor<128x16xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %0 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %4 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %5 = tt.addptr %3, %4 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %6 = scf.for %arg6 = %c0_i32 to %arg3 step %arg4 iter_args(%arg5 = %cst) -> (tensor<128x16xf32, #mma>)  : i32 {
      %7 = tt.load %5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %8 = ttg.convert_layout %7 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %9 = tt.dot %arg2, %8, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %10 = tt.dot %arg2, %8, %9 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      scf.yield %10 : tensor<128x16xf32, #mma>
    }
    tt.return %6 : tensor<128x16xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @chained_dots_with_ops_in_between

  // Ops between dots
  // dot1 -> reduce -> addf %dot1, %reduce1 -> add -> exp2 -> add -> dot2
  // We expect to split after the reduce because the result is used twice

  // CHECK: scf.for

  // CHECK: tt.dot
  // CHECK: arith.addf
  // CHECK: math.exp2
  // CHECK: arith.addf

  // CHECK: ttg.async_wait
  // CHECK: ttg.local_load
  // CHECK: ttg.async_copy_global_to_local

  // CHECK: tt.dot
  // CHECK: tt.reduce

  // CHECK: ttg.async_wait
  // CHECK: ttg.local_load
  // CHECK: ttg.async_copy_global_to_local

  // CHECK: scf.yield

  tt.func @chained_dots_with_ops_in_between(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg2: i32, %arg3: i32) -> tensor<128x16xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %0 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %4 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %5 = tt.addptr %3, %4 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %6 = scf.for %arg5 = %c0_i32 to %arg2 step %arg3 iter_args(%arg6 = %cst) -> (tensor<128x16xf32, #mma>)  : i32 {
      %7 = tt.load %5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %8 = ttg.convert_layout %7 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %9 = tt.load %5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %10 = ttg.convert_layout %9 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %11 = tt.dot %arg1, %8, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %12 = "tt.reduce"(%11) <{axis = 1 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %20 = arith.maxnumf %arg8, %arg9 : f32
        tt.reduce.return %20 : f32
      }) : (tensor<128x16xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %14 = tt.expand_dims %12 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %15 = tt.broadcast %14 : tensor<128x1xf32, #mma> -> tensor<128x16xf32, #mma>
      // Split here since %15 is used twice
      %16 = arith.addf %11, %15 : tensor<128x16xf32, #mma>
      %17 = math.exp2 %15 : tensor<128x16xf32, #mma>
      %18 = arith.addf %16, %17 : tensor<128x16xf32, #mma>
      %19 = tt.dot %arg1, %10, %18 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      scf.yield %19 : tensor<128x16xf32, #mma>
    }
    tt.return %6#0 : tensor<128x16xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @chained_dots_with_loop_carried_partial_result

  // Similar to the previous test but we take the max of the reduce over all iterations (loop carried) so expect a split after the maximum

  // CHECK: scf.for

  // CHECK: tt.dot
  // CHECK: arith.mulf

  // CHECK: ttg.async_wait
  // CHECK: ttg.local_load
  // CHECK: ttg.async_copy_global_to_local

  // CHECK: tt.dot
  // CHECK: tt.reduce
  // CHECK: arith.maxnumf

  // CHECK: ttg.async_wait
  // CHECK: ttg.local_load
  // CHECK: ttg.async_copy_global_to_local

  // CHECK: scf.yield

  tt.func @chained_dots_with_loop_carried_partial_result(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg2: i32, %arg3: i32, %arg101: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>) -> tensor<128x16xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %0 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %4 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %5 = tt.addptr %3, %4 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %6:2 = scf.for %arg4 = %c0_i32 to %arg2 step %arg3 iter_args(%arg5 = %cst, %arg100 = %arg101) -> (tensor<128x16xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>)  : i32 {
      %7 = tt.load %5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %8 = ttg.convert_layout %7 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %9 = tt.load %5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %10 = ttg.convert_layout %9 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %11 = tt.dot %arg1, %8, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %12 = "tt.reduce"(%11) <{axis = 1 : i32}> ({
      ^bb0(%arg6: f32, %arg7: f32):
        %21 = arith.maxnumf %arg6, %arg7 : f32
        tt.reduce.return %21 : f32
      }) : (tensor<128x16xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %24 = arith.maxnumf %12, %arg100 :tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      // Split here since %24 is used twice
      %13 = tt.expand_dims %24 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %14 = tt.broadcast %13 : tensor<128x1xf32, #mma> -> tensor<128x16xf32, #mma>
      %15 = arith.mulf %14, %11 : tensor<128x16xf32, #mma>
      %18 = tt.dot %arg1, %10, %15 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      scf.yield %18, %24 : tensor<128x16xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    }
    tt.return %6 : tensor<128x16xf32, #mma>
  }
}
