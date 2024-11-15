// RUN: triton-opt %s -split-input-file -tritongpu-loop-scheduling=num-stages=3 -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
// CHECK-LABEL: tt.func @load_two_users
  tt.func @load_two_users(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: triton_gpu.async_wait {{.*}} {num = 1 : i32}
    // CHECK: scf.for
    // CHECK:   tt.dot
    // CHECK:   tt.dot
    // CHECK:   triton_gpu.async_copy_global_to_local
    // CHECK:   triton_gpu.async_wait {{.*}} {num = 1 : i32}
    // CHECK:   scf.yield
    // CHECK: triton_gpu.async_wait {num = 0 : i32}

    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = triton_gpu.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = triton_gpu.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = triton_gpu.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory>
      %25 = tt.trans %24 {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory>
      %26 = triton_gpu.local_load %25 : !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----

// CHECK-NOT:  triton_gpu.convert_layout {{.*}} : tensor<32x64xf32, #shared> -> tensor<32x64xf32, #shared1>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = tt.load %arg3 : !tt.ptr<i64>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.splat %1 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.addi %5, %4 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %8 = tt.splat %3 : i64 -> tensor<64x1xi64, #blocked>
    %9 = arith.extsi %7 : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
    %10 = arith.addi %8, %9 : tensor<64x1xi64, #blocked>
    %11 = arith.extsi %arg5 : i32 to i64
    %12 = tt.splat %11 : i64 -> tensor<64x1xi64, #blocked>
    %13 = arith.muli %10, %12 : tensor<64x1xi64, #blocked>
    %14 = arith.muli %2, %arg5 : i32
    %15 = arith.extsi %14 : i32 to i64
    %16 = tt.splat %15 : i64 -> tensor<64x1xi64, #blocked>
    %17 = arith.addi %13, %16 : tensor<64x1xi64, #blocked>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %21 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %22 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked>
    %23 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %24 = arith.muli %20, %22 : tensor<1x64xi32, #blocked>
    %25 = arith.muli %21, %23 : tensor<1x64xi32, #blocked1>
    %26 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %27 = arith.extsi %24 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %28 = arith.extsi %25 : tensor<1x64xi32, #blocked1> to tensor<1x64xi64, #blocked1>
    %29 = tt.broadcast %27 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %30 = arith.addi %26, %29 : tensor<64x64xi64, #blocked>
    %31 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %33 = tt.splat %3 : i64 -> tensor<32x1xi64, #blocked1>
    %34 = arith.extsi %32 : tensor<32x1xi32, #blocked1> to tensor<32x1xi64, #blocked1>
    %35 = arith.addi %33, %34 : tensor<32x1xi64, #blocked1>
    %36 = tt.splat %11 : i64 -> tensor<32x1xi64, #blocked1>
    %37 = arith.muli %35, %36 : tensor<32x1xi64, #blocked1>
    %38 = tt.splat %15 : i64 -> tensor<32x1xi64, #blocked1>
    %39 = arith.addi %37, %38 : tensor<32x1xi64, #blocked1>
    %40 = tt.broadcast %39 : tensor<32x1xi64, #blocked1> -> tensor<32x64xi64, #blocked1>
    %41 = tt.broadcast %28 : tensor<1x64xi64, #blocked1> -> tensor<32x64xi64, #blocked1>
    %42 = arith.addi %40, %41 : tensor<32x64xi64, #blocked1>
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %45 = tt.expand_dims %43 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %46 = tt.expand_dims %44 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %47 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked1>
    %48 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked>
    %49 = arith.muli %45, %47 : tensor<1x32xi32, #blocked1>
    %50 = arith.muli %46, %48 : tensor<1x32xi32, #blocked>
    %51 = tt.broadcast %39 : tensor<32x1xi64, #blocked1> -> tensor<32x32xi64, #blocked1>
    %52 = arith.extsi %49 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
    %53 = arith.extsi %50 : tensor<1x32xi32, #blocked> to tensor<1x32xi64, #blocked>
    %54 = tt.broadcast %52 : tensor<1x32xi64, #blocked1> -> tensor<32x32xi64, #blocked1>
    %55 = arith.addi %51, %54 : tensor<32x32xi64, #blocked1>
    %56 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %57 = tt.addptr %56, %30 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi64, #blocked>
    %58 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked1>
    %59 = tt.addptr %58, %42 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi64, #blocked1>
    %60 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %61 = tt.addptr %60, %55 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi64, #blocked1>
    %62 = tt.load %57 : tensor<64x64x!tt.ptr<f32>, #blocked>
    %63 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c32_i32 iter_args(%arg7 = %cst) -> (tensor<64x32xf32, #mma>)  : i32 {
      %70 = tt.load %59 : tensor<32x64x!tt.ptr<f32>, #blocked1>
      %71 = triton_gpu.convert_layout %62 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = triton_gpu.local_alloc %70 : (tensor<32x64xf32, #blocked1>) -> !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory>
      %73 = tt.trans %72 {order=array<i32: 1,0>} : !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<64x32xf32, #shared1, #triton_gpu.shared_memory>
      %74 = triton_gpu.local_load %73 : !tt.memdesc<64x32xf32, #shared1, #triton_gpu.shared_memory> -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %75 = tt.dot %71, %74, %cst : tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %76 = tt.load %61 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %77 = triton_gpu.convert_layout %75 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %78 = triton_gpu.convert_layout %76 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %79 = tt.dot %77, %78, %arg7 : tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      scf.yield %79 : tensor<64x32xf32, #mma>
    }
    %64 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %65 = tt.broadcast %53 : tensor<1x32xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %66 = arith.addi %64, %65 : tensor<64x32xi64, #blocked>
    %67 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked>
    %68 = tt.addptr %67, %66 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi64, #blocked>
    %69 = triton_gpu.convert_layout %63 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #blocked>
    tt.store %68, %69 : tensor<64x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
} // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
//   CHECK-LABEL: @matmul_tma
//     CHECK-DAG:   triton_gpu.local_alloc  : () -> !tt.memdesc<3x128x64xf16, #{{.+}}, #triton_gpu.shared_memory, mutable>
//     CHECK-DAG:   triton_gpu.local_alloc  : () -> !tt.memdesc<3x64x256xf16, #{{.+}}, #triton_gpu.shared_memory, mutable>
//     CHECK-DAG:   triton_gpu.local_alloc  : () -> !tt.memdesc<3xi64, #{{.+}}, #triton_gpu.shared_memory, mutable>
// CHECK-COUNT-3:   triton_nvidia_gpu.init_barrier
// CHECK-COUNT-4:   triton_nvidia_gpu.async_tma_copy_global_to_local
//         CHECK:   scf.for
//         CHECK:     triton_nvidia_gpu.wait_barrier
//     CHECK-NOT:     triton_nvidia_gpu.wait_barrier
// CHECK-COUNT-2:     triton_nvidia_gpu.async_tma_copy_global_to_local
//         CHECK:     scf.yield
  tt.func public @matmul_tma(%arg0: !tt.tensordesc<tensor<128x64xf16>>, %arg1: !tt.tensordesc<tensor<64x256xf16>>) -> tensor<128x256xf32, #mma> {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0:2 = scf.for %arg3 = %c0_i32 to %c256_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
      %1 = tt.experimental_descriptor_load %arg0[%c0_i32, %arg5] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
      %2 = triton_gpu.local_alloc %1 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %3 = tt.experimental_descriptor_load %arg1[%arg5, %c0_i32] : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
      %4 = triton_gpu.local_alloc %3 : (tensor<64x256xf16, #blocked1>) -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory>
      %5 = triton_nvidia_gpu.warp_group_dot %2, %4, %arg4 { inputPrecision = 0 : i32 } : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
      %6 = arith.addi %arg5, %c64_i32 : i32
      scf.yield %5, %6 : tensor<128x256xf32, #mma>, i32
    }
    tt.return %0#0 : tensor<128x256xf32, #mma>
  }
}
