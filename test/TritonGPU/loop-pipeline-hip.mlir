// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline=num_stages=2 -canonicalize | FileCheck %s --check-prefixes=COMMON,SYNC
// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline="num_stages=2 use_async_copy=1" -canonicalize | FileCheck %s --check-prefixes=COMMON,ASYNC

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func @load_two_users
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
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // SYNC: ttg.local_store
    // SYNC: scf.for
    // SYNC:   tt.load
    // SYNC:   tt.dot
    // SYNC:   tt.dot
    // SYNC:   ttg.local_store
    // SYNC:   scf.yield

    // ASYNC: ttg.async_copy_global_to_local
    // ASYNC: scf.for
    // ASYNC:  ttg.async_wait
    // ASYNC:  ttg.async_copy_global_to_local
    // ASYNC:  tt.dot
    // ASYNC:  tt.dot
    // ASYNC:  scf.yield
    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = ttg.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = ttg.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = ttg.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>
      %25 = ttg.memdesc_trans %24 {order=array<i32: 1,0>} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared1, #smem, mutable>
      %26 = ttg.local_load %25 : !ttg.memdesc<16x64xf16, #shared1, #smem, mutable> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----

// COMMON-LABEL: tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de
// COMMON-NOT:  ttg.convert_layout {{.*}} : tensor<32x64xf32, #shared> -> tensor<32x64xf32, #shared1>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = tt.load %arg3 : !tt.ptr<i64>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.splat %1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.addi %5, %4 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
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
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %21 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %22 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked>
    %23 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %24 = arith.muli %20, %22 : tensor<1x64xi32, #blocked>
    %25 = arith.muli %21, %23 : tensor<1x64xi32, #blocked1>
    %26 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %27 = arith.extsi %24 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %28 = arith.extsi %25 : tensor<1x64xi32, #blocked1> to tensor<1x64xi64, #blocked1>
    %29 = tt.broadcast %27 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %30 = arith.addi %26, %29 : tensor<64x64xi64, #blocked>
    %31 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
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
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %45 = tt.expand_dims %43 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %46 = tt.expand_dims %44 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
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
      %71 = ttg.convert_layout %62 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = ttg.local_alloc %70 : (tensor<32x64xf32, #blocked1>) -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
      %73 = ttg.memdesc_trans %72 {order=array<i32: 1,0>} : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable>
      %74 = ttg.local_load %73 : !ttg.memdesc<64x32xf32, #shared1, #smem, mutable> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %75 = tt.dot %71, %74, %cst : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %76 = tt.load %61 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %77 = ttg.convert_layout %75 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %78 = ttg.convert_layout %76 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %79 = tt.dot %77, %78, %arg7 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      scf.yield %79 : tensor<64x32xf32, #mma>
    }
    %64 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %65 = tt.broadcast %53 : tensor<1x32xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %66 = arith.addi %64, %65 : tensor<64x32xi64, #blocked>
    %67 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked>
    %68 = tt.addptr %67, %66 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi64, #blocked>
    %69 = ttg.convert_layout %63 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #blocked>
    tt.store %68, %69 : tensor<64x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
} // end module

// -----

// Disable pipelining for loops that contain barrier.
//   Barriers are problematic since they are not chained to any other operation.
// COMMON-LABEL: tt.func public @add_barrier_kernel
// COMMON:  scf.for
// COMMON:    tt.load
// COMMON:    gpu.barrier
// COMMON:    tt.store
// COMMON-NOT:  gpu.barrier
// COMMON:  tt.return

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @add_barrier_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %cval_f32 = arith.constant dense<0.3> : tensor<1024xf32, #blocked>
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %arg3 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
      gpu.barrier
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %15 = arith.addf %12, %cval_f32 : tensor<1024xf32, #blocked>
      tt.store %16, %15 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
} // end module

// -----

// COMMON-NOT: #ttg.swizzled_shared<{{.*}} order = [2, 0, 1]
// COMMON: #ttg.swizzled_shared<{{.*}} order = [2, 1, 0]
// COMMON-NOT: #ttg.swizzled_shared<{{.*}} order = [2, 0, 1]

// COMMON-LABEL: tt.func public @slowest_dim_is_batch
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [4, 1, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [16, 1, 4], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [16, 1, 4], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @slowest_dim_is_batch(%arg0: tensor<1x512x!tt.ptr<f32>, #blocked2>, %arg1: tensor<64x8x32x!tt.ptr<f32>, #blocked1>, %arg2: tensor<64x1x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x1x32xf32, #blocked>
    %cst_0 = arith.constant dense<512> : tensor<1x512xi32, #blocked2>
    %cst_1 = arith.constant dense<128> : tensor<64x8x32xi32, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %33:3 = scf.for %arg7 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %arg0, %arg10 = %arg1) -> (tensor<64x1x32xf32, #blocked>, tensor<1x512x!tt.ptr<f32>, #blocked2>, tensor<64x8x32x!tt.ptr<f32>, #blocked1>)  : i32 {
      %39 = tt.load %arg9 : tensor<1x512x!tt.ptr<f32>, #blocked2>
      %40 = tt.load %arg10 : tensor<64x8x32x!tt.ptr<f32>, #blocked1>
      %41 = tt.reshape %39 allow_reorder : tensor<1x512xf32, #blocked2> -> tensor<64x1x8xf32, #blocked5>
      %43 = ttg.convert_layout %41 : tensor<64x1x8xf32, #blocked5> -> tensor<64x1x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %44 = ttg.convert_layout %40 : tensor<64x8x32xf32, #blocked1> -> tensor<64x8x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %45 = tt.dot %43, %44, %arg8 : tensor<64x1x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x8x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x1x32xf32, #blocked>
      %46 = tt.addptr %arg9, %cst_0 : tensor<1x512x!tt.ptr<f32>, #blocked2>, tensor<1x512xi32, #blocked2>
      %47 = tt.addptr %arg10, %cst_1 : tensor<64x8x32x!tt.ptr<f32>, #blocked1>, tensor<64x8x32xi32, #blocked1>
      scf.yield %45, %46, %47 : tensor<64x1x32xf32, #blocked>, tensor<1x512x!tt.ptr<f32>, #blocked2>, tensor<64x8x32x!tt.ptr<f32>, #blocked1>
    }
    tt.store %arg2, %33#0 : tensor<64x1x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Check that the stream pipeliner updates the resulting memory layout of transpose ops to mutable if immutable local buffers are replaced
// COMMON-LABEL: loop_with_dot_and_transpose
// COMMON: ttg.local_alloc {{.*}}, mutable>
// COMMON: ttg.memdesc_trans {{.*}}, mutable, {{.*}} -> {{.*}}, mutable

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @loop_with_dot_and_transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg4: tensor<32x32x!tt.ptr<f32>, #blocked1>, %arg5: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg3 = %cst) -> (tensor<32x32xf32, #blocked>)  : i32 {
      %2 = tt.load %arg4 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %3 = ttg.local_alloc %2 : (tensor<32x32xf32, #blocked1>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
      %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x32xf32, #shared, #smem> -> !ttg.memdesc<32x32xf32, #shared1, #smem>
      %5 = ttg.local_load %4 : !ttg.memdesc<32x32xf32, #shared1, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %6 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %7 = tt.dot %6, %5, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
      scf.yield %7 : tensor<32x32xf32, #blocked>
    }
    tt.store %arg5, %0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Check that the stream pipeliner updates atomic op in the k-loop correctly
// COMMON-LABEL: _triton_gemm_kernel_atomic_rmw
// COMMON:  scf.for
// COMMON: tt.atomic_rmw fadd, acq_rel, gpu
// COMMON:  tt.dot
// COMMON: scf.yield

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_triton_gemm_kernel_atomic_rmw(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<32> : tensor<32x32xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %2 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<32x1xi32, #blocked>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<32x32xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %10 = tt.addptr %9, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %11 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %12 = tt.addptr %11, %8 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %13 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %3 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %15 = tt.broadcast %14 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %16 = tt.addptr %15, %7 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %17 = tt.splat %arg3 : i32 -> tensor<32x1xi32, #blocked>
    %18 = arith.cmpi slt, %1, %17 : tensor<32x1xi32, #blocked>
    %19 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #blocked>
    %20 = arith.cmpi slt, %5, %19 : tensor<1x32xi32, #blocked>
    %21 = tt.broadcast %18 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %22 = tt.broadcast %20 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %23 = arith.andi %21, %22 : tensor<32x32xi1, #blocked>
    %24 = arith.addi %arg3, %c31_i32 : i32
    %25 = arith.divsi %24, %c32_i32 : i32
    %26 = arith.muli %arg4, %c32_i32 : i32
    %27 = tt.splat %26 : i32 -> tensor<32x32xi32, #blocked>
    %28:3 = scf.for %arg5 = %c0_i32 to %25 step %c1_i32 iter_args(%arg6 = %cst_0, %arg7 = %10, %arg8 = %12) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>)  : i32 {
      %32 = tt.load %arg7 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %33 = tt.load %arg8 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %34 = ttg.convert_layout %32 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %35 = ttg.convert_layout %33 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %36 = tt.dot %34, %35, %arg6 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf32, #mma>
      %37 = tt.addptr %arg7, %cst : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %38 = tt.addptr %arg8, %27 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %39 = arith.truncf %36 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
      %40 = ttg.convert_layout %39 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #blocked>
      %41 = tt.atomic_rmw fadd, acq_rel, gpu, %16, %40, %23 : (tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xf16, #blocked>, tensor<32x32xi1, #blocked>) -> tensor<32x32xf16, #blocked>
      scf.yield %36, %37, %38 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked>
    }
    %29 = arith.truncf %28#0 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
    %30 = ttg.convert_layout %16 : tensor<32x32x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #mma>
    %31 = ttg.convert_layout %23 : tensor<32x32xi1, #blocked> -> tensor<32x32xi1, #mma>
    tt.store %30, %29, %31 : tensor<32x32x!tt.ptr<f16>, #mma>
    tt.return
  }
}

// -----

// Check that we can pipeline scaled dot with linear layout
// COMMON-LABEL: mxfp8_mxfp4_matmul

// Prologue
// SYNC-3: ttg.local_alloc
// SYNC-3: tt.load
// SYNC-3: ttg.local_store
//
// ASYNC-3: ttg.async_copy_global_to_local

// Main loop
//         COMMON: scf.for
//          ASYNC: ttg.async_wait
// COMMON-COUNT-3:   ttg.local_load
//         COMMON:   tt.dot_scaled
//         COMMON:   scf.yield

// Epilogue
//          ASYNC: ttg.async_wait
// COMMON-COUNT-3: ttg.local_load
//         COMMON: scf.if
//         COMMON:   tt.dot_scaled
// COMMON-COUNT-2:   scf.yield
// COMMON-COUNT-3: ttg.local_dealloc

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 2], [0, 4], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 2], [0, 4], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mxfp8_mxfp4_matmul(
      %arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32},
      %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32},
      %71: tensor<128x256x!tt.ptr<f32>, #blocked3>) {
    %cst = arith.constant dense<256> : tensor<128x256xi32, #blocked>
    %cst_0 = arith.constant dense<8> : tensor<256x8xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked2>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c255_i32 = arith.constant 255 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %8 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %10 = arith.addi %8, %6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = arith.addi %9, %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %12 = tt.splat %arg4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.remsi %10, %12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = arith.muli %4, %c256_i32 : i32
    %15 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %18 = tt.splat %14 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.splat %14 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.splat %14 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %21 = arith.addi %18, %15 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %22 = arith.addi %19, %16 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %23 = arith.addi %20, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %24 = tt.splat %arg5 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %25 = tt.splat %arg5 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %26 = arith.remsi %21, %24 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.remsi %22, %25 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %28 = tt.expand_dims %26 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %29 = tt.splat %arg7 : i32 -> tensor<256x1xi32, #blocked1>
    %30 = arith.muli %28, %29 : tensor<256x1xi32, #blocked1>
    %31 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<256x1x!tt.ptr<i8>, #blocked1>
    %32 = tt.addptr %31, %30 : tensor<256x1x!tt.ptr<i8>, #blocked1>, tensor<256x1xi32, #blocked1>
    %33 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x8xi32, #blocked1>
    %35 = tt.broadcast %32 : tensor<256x1x!tt.ptr<i8>, #blocked1> -> tensor<256x8x!tt.ptr<i8>, #blocked1>
    %36 = tt.broadcast %34 : tensor<1x8xi32, #blocked1> -> tensor<256x8xi32, #blocked1>
    %37 = tt.addptr %35, %36 : tensor<256x8x!tt.ptr<i8>, #blocked1>, tensor<256x8xi32, #blocked1>
    %38 = tt.expand_dims %13 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %39 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
    %40 = arith.muli %38, %39 : tensor<128x1xi32, #blocked>
    %41 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %42 = tt.expand_dims %41 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %43 = tt.broadcast %40 : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %44 = tt.broadcast %42 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %45 = arith.addi %43, %44 : tensor<128x256xi32, #blocked>
    %46 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
    %47 = tt.addptr %46, %45 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>
    %48 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %49 = tt.expand_dims %48 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %50 = tt.splat %arg9 : i32 -> tensor<128x1xi32, #blocked>
    %51 = arith.muli %49, %50 : tensor<128x1xi32, #blocked>
    %52 = tt.expand_dims %27 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %53 = tt.broadcast %51 : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %54 = tt.broadcast %52 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %55 = arith.addi %53, %54 : tensor<128x256xi32, #blocked>
    %56 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x256x!tt.ptr<i8>, #blocked>
    %57 = tt.addptr %56, %55 : tensor<128x256x!tt.ptr<i8>, #blocked>, tensor<128x256xi32, #blocked>
    %58 = arith.addi %arg6, %c255_i32 : i32
    %59 = arith.divsi %58, %c256_i32 : i32
    %60 = arith.muli %arg9, %c128_i32 : i32
    %61 = tt.splat %60 : i32 -> tensor<128x256xi32, #blocked>
    %62:5 = scf.for %arg11 = %c0_i32 to %59 step %c1_i32 iter_args(%arg12 = %cst_2, %arg13 = %47, %arg14 = %57, %arg15 = %37, %arg16 = %cst_3)
      -> (tensor<128x256xf32, #blocked2>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256x!tt.ptr<i8>, #blocked>, tensor<256x8x!tt.ptr<i8>, #blocked1>, tensor<128x256xf32, #mma>)  : i32 {
      %80 = tt.load %arg13 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
      %81 = tt.load %arg14 : tensor<128x256x!tt.ptr<i8>, #blocked>
      %82 = tt.load %arg15 : tensor<256x8x!tt.ptr<i8>, #blocked1>
      %83 = ttg.convert_layout %80 : tensor<128x256xf8E5M2, #blocked> -> tensor<128x256xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %84 = ttg.convert_layout %81 : tensor<128x256xi8, #blocked> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %85 = ttg.convert_layout %82 : tensor<256x8xi8, #blocked1> -> tensor<256x8xi8, #linear1>
      %86 = tt.dot_scaled %83 scale %cst_1, %84 scale %85, %arg16 lhs = e5m2 rhs = e2m1 {fastMath = false} : tensor<128x256xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<128x256xf32, #mma>
      %87 = ttg.convert_layout %86 : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked2>
      %88 = tt.addptr %arg13, %cst : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>
      %89 = tt.addptr %arg14, %61 : tensor<128x256x!tt.ptr<i8>, #blocked>, tensor<128x256xi32, #blocked>
      %90 = tt.addptr %arg15, %cst_0 : tensor<256x8x!tt.ptr<i8>, #blocked1>, tensor<256x8xi32, #blocked1>
      scf.yield %87, %88, %89, %90, %86 : tensor<128x256xf32, #blocked2>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256x!tt.ptr<i8>, #blocked>, tensor<256x8x!tt.ptr<i8>, #blocked1>, tensor<128x256xf32, #mma>
    } {tt.num_stages = 2 : i32}
    %79 = ttg.convert_layout %62#0 : tensor<128x256xf32, #blocked2> -> tensor<128x256xf32, #blocked3>
    tt.store %71, %79 : tensor<128x256x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

// Check that we can pipeline a simple matmul kernel

// COMMON-LABEL: simple_matmul_kernel

// Prologue
// COMMON-COUNT-2: ttg.local_alloc
  // SYNC-COUNT-2: tt.load
  // SYNC-COUNT-2: ttg.local_store
  //
  // ASYNC-COUNT-2: ttg.async_copy_global_to_local

// Main loop
//         COMMON:   scf.for
//
  // SYNC-COUNT-2:   ttg.local_load
  //         SYNC:   tt.dot
  //         SYNC:   scf.yield
  //
  //         ASYNC:    ttg.async_wait
  //         ASYNC:    ttg.async_copy_global_to_local
  //         ASYNC:    ttg.local_load {{.*}} token
  //         ASYNC:    ttg.async_copy_global_to_local
  //         ASYNC:    ttg.local_load {{.*}} token
  //         ASYNC:    ttg.dot

// Epilogue
//          ASYNC: ttg.async_wait
// COMMON-COUNT-2: ttg.local_load
//         COMMON: scf.if
//         COMMON:   tt.dot
// COMMON-COUNT-2:   scf.yield
// COMMON-COUNT-2: ttg.local_dealloc

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @simple_matmul_kernel(%test: tensor<1x64xi32, #blocked1>, %arg0: tensor<64x64x!tt.ptr<f16>, #mma>, %arg1: i32, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<32x64xi32, #blocked1>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %1 = arith.muli %arg1, %c64_i32 : i32
    %2 = tt.splat %1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %3 = arith.addi %2, %0 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %4 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = arith.remsi %3, %4 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %9 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %10 = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %11 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %13 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
    %14 = tt.addptr %13, %12 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
    %15:3 = scf.for %arg11 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg12 = %cst_1, %arg13 = %10, %arg14 = %14) -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %17 = tt.load %arg13 : tensor<64x32x!tt.ptr<f16>, #blocked>
      %18 = tt.load %arg14 : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %19 = ttg.convert_layout %17 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %20 = ttg.convert_layout %18 : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %21 = tt.dot %19, %20, %arg12, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      %22 = tt.addptr %arg13, %cst : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %23 = tt.addptr %arg14, %cst_0 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      scf.yield %21, %22, %23 : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>
    }
    %16 = arith.truncf %15#0 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
    tt.store %arg0, %16 : tensor<64x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}

// -----

// Check that we can pipeline small width vectors (like scale factor)
// COMMON-LABEL: pipeline_small_vector

// Prologue
// COMMON-COUNT-4: tt.load

// Main loop
//         COMMON: scf.for
// COMMON-COUNT-4:   tt.load
//         COMMON:   tt.dot_scaled
//         COMMON:   scf.yield

// Epilogue
//         COMMON: scf.if
//         COMMON:   tt.dot_scaled
//         COMMON:   scf.yield

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pipeline_small_vector(%arg0: !tt.ptr<f8E5M2>, %arg1: !tt.ptr<f8E5M2>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<i8>, %arg4: !tt.ptr<i8>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) -> tensor<128x256xf32, #blocked3> {
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<4> : tensor<128x4xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf8E5M2, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf8E5M2, #blocked2>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked3>
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant dense<4> : tensor<256x4xi32, #blocked4>
    %cst_4 = arith.constant dense<128> : tensor<128x128xi32, #blocked2>
    %cst_5 = arith.constant dense<8> : tensor<256x1xi32, #blocked4>
    %cst_6 = arith.constant dense<8> : tensor<128x1xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg5, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %9 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %11 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %13 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %14 = arith.addi %11, %6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %15 = arith.addi %12, %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %16 = arith.addi %13, %9 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
    %17 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %19 = arith.remsi %14, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.remsi %15, %18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %21 = arith.muli %4, %c256_i32 : i32
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %23 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %25 = tt.splat %21 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %26 = tt.splat %21 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %21 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %28 = arith.addi %25, %22 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %29 = arith.addi %26, %23 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %30 = arith.addi %27, %24 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %31 = tt.splat %arg6 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %32 = tt.splat %arg6 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %33 = arith.remsi %28, %31 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %34 = arith.remsi %29, %32 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %35 = tt.expand_dims %19 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %36 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %37 = arith.muli %35, %cst_6 : tensor<128x1xi32, #blocked>
    %38 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked>
    %39 = tt.addptr %38, %37 : tensor<128x1x!tt.ptr<i8>, #blocked>, tensor<128x1xi32, #blocked>
    %40 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %41 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %42 = tt.expand_dims %40 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x4xi32, #blocked4>
    %43 = tt.expand_dims %41 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi32, #blocked>
    %44 = tt.broadcast %39 : tensor<128x1x!tt.ptr<i8>, #blocked> -> tensor<128x4x!tt.ptr<i8>, #blocked>
    %45 = tt.broadcast %43 : tensor<1x4xi32, #blocked> -> tensor<128x4xi32, #blocked>
    %46 = tt.addptr %44, %45 : tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<128x4xi32, #blocked>
    %47 = tt.expand_dims %33 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<256x1xi32, #blocked4>
    %48 = arith.muli %47, %cst_5 : tensor<256x1xi32, #blocked4>
    %49 = tt.splat %arg4 : !tt.ptr<i8> -> tensor<256x1x!tt.ptr<i8>, #blocked4>
    %50 = tt.addptr %49, %48 : tensor<256x1x!tt.ptr<i8>, #blocked4>, tensor<256x1xi32, #blocked4>
    %51 = tt.broadcast %50 : tensor<256x1x!tt.ptr<i8>, #blocked4> -> tensor<256x4x!tt.ptr<i8>, #blocked4>
    %52 = tt.broadcast %42 : tensor<1x4xi32, #blocked4> -> tensor<256x4xi32, #blocked4>
    %53 = tt.addptr %51, %52 : tensor<256x4x!tt.ptr<i8>, #blocked4>, tensor<256x4xi32, #blocked4>
    %54 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
    %55 = arith.muli %36, %54 : tensor<128x1xi32, #blocked2>
    %56 = tt.expand_dims %10 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %57 = tt.broadcast %55 : tensor<128x1xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
    %58 = tt.broadcast %56 : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
    %59 = arith.addi %57, %58 : tensor<128x128xi32, #blocked2>
    %60 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<128x128x!tt.ptr<f8E5M2>, #blocked2>
    %61 = tt.addptr %60, %59 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<128x128xi32, #blocked2>
    %62 = tt.expand_dims %8 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %63 = tt.splat %arg9 : i32 -> tensor<128x1xi32, #blocked1>
    %64 = arith.muli %62, %63 : tensor<128x1xi32, #blocked1>
    %65 = tt.expand_dims %34 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %66 = tt.broadcast %64 : tensor<128x1xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %67 = tt.broadcast %65 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %68 = arith.addi %66, %67 : tensor<128x256xi32, #blocked1>
    %69 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>
    %70 = tt.addptr %69, %68 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<128x256xi32, #blocked1>
    %71 = arith.addi %arg7, %c127_i32 : i32
    %72 = arith.divsi %71, %c128_i32 : i32
    %73 = arith.muli %arg9, %c128_i32 : i32
    %74 = tt.splat %73 : i32 -> tensor<128x256xi32, #blocked1>
    %75:5 = scf.for %arg11 = %c0_i32 to %72 step %c1_i32 iter_args(%arg12 = %cst_2, %arg13 = %46, %arg14 = %61, %arg15 = %70, %arg16 = %53) -> (tensor<128x256xf32, #blocked3>, tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<128x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x4x!tt.ptr<i8>, #blocked4>)  : i32 {
      %93 = arith.muli %arg11, %c128_i32 : i32
      %94 = arith.subi %arg7, %93 : i32
      %95 = tt.splat %94 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %96 = tt.splat %94 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %97 = arith.cmpi slt, %10, %95 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %98 = arith.cmpi slt, %8, %96 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %99 = tt.expand_dims %97 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi1, #blocked2>
      %100 = tt.broadcast %99 : tensor<1x128xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
      %101 = tt.load %arg14, %100, %cst_1 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked2>
      %102 = ttg.convert_layout %101 : tensor<128x128xf8E5M2, #blocked2> -> tensor<128x128xf8E5M2, #blocked6>
      %103 = tt.expand_dims %98 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi1, #blocked1>
      %104 = tt.broadcast %103 : tensor<128x1xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
      %105 = tt.load %arg15, %104, %cst_0 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>
      %106 = ttg.convert_layout %105 : tensor<128x256xf8E5M2, #blocked1> -> tensor<128x256xf8E5M2, #blocked3>
      %107 = tt.load %arg13 : tensor<128x4x!tt.ptr<i8>, #blocked>
      %108 = tt.load %arg16 : tensor<256x4x!tt.ptr<i8>, #blocked4>
      %109 = ttg.convert_layout %108 : tensor<256x4xi8, #blocked4> -> tensor<256x4xi8, #blocked>
      %110 = tt.dot_scaled %102 scale %107, %106 scale %109, %arg12 lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<128x128xf8E5M2, #blocked6>, tensor<128x4xi8, #blocked> * tensor<128x256xf8E5M2, #blocked3>, tensor<256x4xi8, #blocked> -> tensor<128x256xf32, #blocked3>
      %111 = tt.addptr %arg14, %cst_4 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<128x128xi32, #blocked2>
      %112 = tt.addptr %arg15, %74 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<128x256xi32, #blocked1>
      %113 = tt.addptr %arg13, %cst : tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<128x4xi32, #blocked>
      %114 = tt.addptr %arg16, %cst_3 : tensor<256x4x!tt.ptr<i8>, #blocked4>, tensor<256x4xi32, #blocked4>
      scf.yield %110, %113, %111, %112, %114 : tensor<128x256xf32, #blocked3>, tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<128x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x4x!tt.ptr<i8>, #blocked4>
    } {tt.num_stages = 2 : i32}
    tt.return %75#0 : tensor<128x256xf32, #blocked3>
  }
}

// -----

// Check we do not get AsyncCopyGlobalToLocal because the vec width will be < 32bit.
// The order of the shared memory will be getMemoryOrder(#linear1) == [0, 1]
// which differs from the order [1, 0] of the blocked layout. Since we have to
// gather into lds with AsyncCopyGlobalToLocal we have to fallback to registers

// COMMON-LABEL: pipeline_scale_memory_order
// COMMON-NOT: ttg.async_copy_global_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [64, 1], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 4], [16, 0], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [128, 0], [256, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 8], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pipeline_scale_memory_order(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i64 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg3: tensor<128x512xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg4: tensor<128x512x!tt.ptr<f32>, #mma>, %arg5: tensor<512x8x!tt.ptr<i8>, #blocked>) {
    %cst = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %cst_0 = arith.constant dense<8> : tensor<512x8xi32, #blocked>
    %c256_i64 = arith.constant 256 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x512xf32, #mma>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = arith.extsi %0 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<8xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi64, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x8x!tt.ptr<i8>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<1x8x!tt.ptr<i8>, #blocked>, tensor<1x8xi64, #blocked>
    %5 = tt.broadcast %4 : tensor<1x8x!tt.ptr<i8>, #blocked> -> tensor<512x8x!tt.ptr<i8>, #blocked>
    %6:2 = scf.for %arg6 = %c0_i64 to %arg1 step %c256_i64 iter_args(%arg7 = %cst_1, %arg8 = %5) -> (tensor<128x512xf32, #mma>, tensor<512x8x!tt.ptr<i8>, #blocked>)  : i64 {
      %7 = tt.load %arg8 : tensor<512x8x!tt.ptr<i8>, #blocked>
      %8 = ttg.convert_layout %7 : tensor<512x8xi8, #blocked> -> tensor<512x8xi8, #linear1>
      %9 = tt.dot_scaled %arg2 scale %cst, %arg3 scale %8, %arg7 lhs = e4m3 rhs = e2m1 {fastMath = true} : tensor<128x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x8xi8, #linear> * tensor<128x512xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<512x8xi8, #linear1> -> tensor<128x512xf32, #mma>
      %10 = tt.addptr %arg8, %cst_0 : tensor<512x8x!tt.ptr<i8>, #blocked>, tensor<512x8xi32, #blocked>
      scf.yield %9, %10 : tensor<128x512xf32, #mma>, tensor<512x8x!tt.ptr<i8>, #blocked>
    }
    tt.store %arg4, %6#0 : tensor<128x512x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
// Verify that we do not get AsyncCopies because we cannot lower it on gfx942 since we only have 32bit wide loads to lds
// COMMON-LABEL: @reject_fp64_pipelining_with_async_copy_gfx942
// ASYNC-NOT: ttg.async_copy_global_to_local
tt.func @reject_fp64_pipelining_with_async_copy_gfx942(
                  %a_ptr : tensor<128x32x!tt.ptr<f64>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B : tensor<32x128xf64, #B>, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf64, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf64, #C>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%prev_c = %c_init) -> (tensor<128x128xf64, #C>) : i32 {
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f64>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf64, #AL> -> tensor<128x32xf64, #A>
    %c = tt.dot %a, %B, %prev_c : tensor<128x32xf64, #A> * tensor<32x128xf64, #B> -> tensor<128x128xf64, #C>
    scf.yield %c : tensor<128x128xf64, #C>
  }
  tt.return %loop: tensor<128x128xf64, #C>
}
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
// On GFX950 we can use AsyncCopy if sizePerThread >= 2 and it's contiguous because we can load 2 fp64 with one direct to lds instruction
// COMMON-LABEL: @pipeline_fp64_with_async_copy_gfx950
// ASYNC: ttg.async_copy_global_to_local
// ASYNC: tt.load
// ASYNC: ttg.async_copy_global_to_local
// ASYNC: tt.load
tt.func @pipeline_fp64_with_async_copy_gfx950(
                  %a_ptr : tensor<128x32x!tt.ptr<f64>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %b_ptr : tensor<32x128x!tt.ptr<f64>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 2 : i32},
                  %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf64, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf64, #C>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%prev_c = %c_init) -> (tensor<128x128xf64, #C>) : i32 {
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f64>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf64, #AL> -> tensor<128x32xf64, #A>
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f64>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf64, #BL> -> tensor<32x128xf64, #B>
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf64, #A> * tensor<32x128xf64, #B> -> tensor<128x128xf64, #C>
    scf.yield %c : tensor<128x128xf64, #C>
  }
  tt.return %loop: tensor<128x128xf64, #C>
}
}

// -----

// COMMON-LABEL: pipelining_local_load_packed_transposed

// Prologue
// COMMON: ttg.local_alloc
// COMMON: ttg.local_alloc
// ASYNC: ttg.async_copy_global_to_local
// SYNC: tt.load
// COMMON: tt.load
// SYNC: ttg.local_store
// COMMON: ttg.local_store

// Main loop
//         COMMON: scf.for
//         COMMON:   ttg.local_load
//         COMMON:   amdgpu.local_load_packed_tranposed
//         COMMON:   tt.dot_scaled
//         COMMON:   scf.yield

// Epilogue
//         COMMON:   ttg.local_load
//         COMMON: amdgpu.local_load_packed_tranposed
//         COMMON: scf.if
//         COMMON:   tt.dot_scaled
// COMMON-COUNT-2:   scf.yield
// COMMON-COUNT-2: ttg.local_dealloc

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pipelining_local_load_packed_transposed(%a_ptr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %b_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_scale: i32 {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bn: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<128> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %M, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %9 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %11 = arith.addi %9, %6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = arith.addi %10, %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %13 = arith.muli %4, %c128_i32 : i32
    %14 = arith.divsi %13, %c2_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %16 = tt.splat %14 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %17 = arith.addi %16, %15 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %18 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %19 = tt.expand_dims %12 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %20 = tt.splat %stride_am : i32 -> tensor<128x1xi32, #blocked>
    %21 = arith.muli %18, %20 : tensor<128x1xi32, #blocked>
    %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %24 = tt.broadcast %21 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %25 = tt.broadcast %23 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %26 = arith.addi %24, %25 : tensor<128x128xi32, #blocked>
    %27 = tt.splat %a_ptr : !tt.ptr<f8E5M2> -> tensor<128x128x!tt.ptr<f8E5M2>, #blocked>
    %28 = tt.addptr %27, %26 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked>, tensor<128x128xi32, #blocked>
    %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %30 = tt.expand_dims %29 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %31 = tt.expand_dims %17 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %32 = tt.splat %stride_bn : i32 -> tensor<1x64xi32, #blocked1>
    %33 = arith.muli %31, %32 : tensor<1x64xi32, #blocked1>
    %34 = tt.broadcast %30 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %35 = tt.broadcast %33 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %36 = arith.addi %34, %35 : tensor<128x64xi32, #blocked1>
    %37 = tt.splat %b_ptr : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked1>
    %38 = tt.addptr %37, %36 : tensor<128x64x!tt.ptr<i8>, #blocked1>, tensor<128x64xi32, #blocked1>
    %39 = arith.addi %K, %c127_i32 : i32
    %40 = arith.divsi %39, %c128_i32 : i32
    %accumulator:3 = scf.for %accumulator_2 = %c0_i32 to %40 step %c1_i32 iter_args(%arg11 = %cst_1, %arg12 = %28, %arg13 = %38) -> (tensor<128x128xf32, #mma>, tensor<128x128x!tt.ptr<f8E5M2>, #blocked>, tensor<128x64x!tt.ptr<i8>, #blocked1>)  : i32 {
      %60 = tt.load %arg12 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked>
      %61 = tt.load %arg13 : tensor<128x64x!tt.ptr<i8>, #blocked1>
      %62 = ttg.convert_layout %60 : tensor<128x128xf8E5M2, #blocked> -> tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %63 = ttg.local_alloc %61 : (tensor<128x64xi8, #blocked1>) -> !ttg.memdesc<128x64xi8, #shared, #smem>
      %64 = amdgpu.local_load_packed_tranposed %63 : !ttg.memdesc<128x64xi8, #shared, #smem> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %65 = tt.dot_scaled %62, %64, %arg11 lhs = e5m2 rhs = e2m1 {fastMath = false} : tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x128xf32, #mma>
      %66 = tt.addptr %arg12, %cst : tensor<128x128x!tt.ptr<f8E5M2>, #blocked>, tensor<128x128xi32, #blocked>
      %67 = tt.addptr %arg13, %cst_0 : tensor<128x64x!tt.ptr<i8>, #blocked1>, tensor<128x64xi32, #blocked1>
      scf.yield %65, %66, %67 : tensor<128x128xf32, #mma>, tensor<128x128x!tt.ptr<f8E5M2>, #blocked>, tensor<128x64x!tt.ptr<i8>, #blocked1>
    } {tt.num_stages = 2 : i32}
    %41 = tt.splat %13 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %42 = arith.addi %41, %8 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %43 = tt.splat %stride_cm : i32 -> tensor<128x1xi32, #blocked2>
    %44 = arith.muli %43, %19 : tensor<128x1xi32, #blocked2>
    %45 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked2>
    %46 = tt.addptr %45, %44 : tensor<128x1x!tt.ptr<f32>, #blocked2>, tensor<128x1xi32, #blocked2>
    %47 = tt.expand_dims %42 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %48 = tt.broadcast %46 : tensor<128x1x!tt.ptr<f32>, #blocked2> -> tensor<128x128x!tt.ptr<f32>, #blocked2>
    %49 = tt.broadcast %47 : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
    %50 = tt.addptr %48, %49 : tensor<128x128x!tt.ptr<f32>, #blocked2>, tensor<128x128xi32, #blocked2>
    %51 = tt.splat %M : i32 -> tensor<128x1xi32, #blocked2>
    %52 = arith.cmpi slt, %19, %51 : tensor<128x1xi32, #blocked2>
    %53 = tt.splat %N : i32 -> tensor<1x128xi32, #blocked2>
    %54 = arith.cmpi slt, %47, %53 : tensor<1x128xi32, #blocked2>
    %55 = tt.broadcast %52 : tensor<128x1xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
    %56 = tt.broadcast %54 : tensor<1x128xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
    %57 = arith.andi %55, %56 : tensor<128x128xi1, #blocked2>
    %58 = ttg.convert_layout %50 : tensor<128x128x!tt.ptr<f32>, #blocked2> -> tensor<128x128x!tt.ptr<f32>, #mma>
    %59 = ttg.convert_layout %57 : tensor<128x128xi1, #blocked2> -> tensor<128x128xi1, #mma>
    tt.store %58, %accumulator#0, %59 : tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}
