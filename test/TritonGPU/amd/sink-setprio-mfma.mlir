// RUN: triton-opt %s -split-input-file --tritonamdgpu-block-pingpong | FileCheck %s

//CHECK: triton_gpu.local_load
//CHECK: rocdl.sched.barrier 6
//CHECK: tt.load
//CHECK: rocdl.sched.barrier 6
//CHECK: triton_gpu.local_load
//CHECK: rocdl.sched.barrier 6
//CHECK: tt.load
//CHECK: rocdl.sched.barrier 0
//CHECK: rocdl.s.setprio 1
//CHECK: tt.dot
//CHECK: rocdl.s.setprio 0
//CHECK: rocdl.sched.barrier 0

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %7 = arith.muli %5, %6 : tensor<128x1xi32, #blocked1>
    %8 = tt.addptr %0, %7 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %9 = tt.broadcast %8 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %13 = tt.addptr %9, %12 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %17 = tt.addptr %14, %16 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %18 = tt.broadcast %17 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %19 = tt.splat %arg7 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    %22 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %23 = triton_gpu.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    %24 = triton_gpu.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x64x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %25:6 = scf.for %arg10 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg11 = %cst, %arg12 = %13, %arg13 = %20, %arg14 = %c0_i32, %arg15 = %23, %arg16 = %24) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg12, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg13, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %30 = triton_gpu.local_load %arg15 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %31 = triton_gpu.local_load %arg16 : !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %32 = tt.dot %30, %31, %arg11 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %33 = arith.addi %arg14, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = triton_gpu.memdesc_subview %21[%35, %c0_i32, %c0_i32] : !tt.memdesc<1x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %27, %36 : tensor<128x64xf16, #blocked1> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
      %37 = triton_gpu.memdesc_subview %22[%35, %c0_i32, %c0_i32] : !tt.memdesc<1x64x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %29, %37 : tensor<64x128xf16, #blocked> -> !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %32, %26, %28, %35, %36, %37 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %21 : !tt.memdesc<1x128x64xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %22 : !tt.memdesc<1x64x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}
