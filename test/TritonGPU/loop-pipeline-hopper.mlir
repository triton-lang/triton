// RUN: triton-opt %s -split-input-file -tritongpu-pipeline -canonicalize | FileCheck --dump-input-context=50 %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: dot_chained_single_load
  tt.func @dot_chained_single_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x64xf32, #mma> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: scf.for
    // CHECK:   triton_gpu.async_wait {{.*}} {num = 1 : i32}
    // CHECK:   triton_nvidia_gpu.warp_group_dot
    // CHECK-NEXT: triton_nvidia_gpu.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
    // CHECK:   triton_nvidia_gpu.warp_group_dot
    // CHECK:   triton_gpu.async_copy_global_to_local
    // CHECK:   triton_gpu.async_commit_group
    // CHECK:   scf.yield
    %17:2 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_3, %arg5 = %16) -> (tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>)  : i32 {
      %18 = tt.load %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = triton_gpu.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %20 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>
      %21 = triton_nvidia_gpu.warp_group_dot %19, %20, %cst_2 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1>
      %23 = tt.trans %20 {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> !tt.memdesc<16x64xf16, #shared, #triton_gpu.shared_memory>
      %24 = triton_gpu.convert_layout %22 : tensor<128x16xf16, #mma1> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
      %25 = triton_nvidia_gpu.warp_group_dot %24, %23, %arg4 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<16x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf32, #mma>
      %26 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %25, %26 : tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>
    }
    tt.return %17#0 : tensor<128x64xf32, #mma>
  }
}
