// RUN: triton-opt %s -split-input-file --tritonamdgpu-block-pingpong | FileCheck %s

//CHECK-LABEL: pingpong_small
//CHECK: ttg.local_load
//CHECK: rocdl.s.setprio 1
//CHECK: tt.load
//CHECK: rocdl.sched.barrier
//CHECK: ttg.local_load
//CHECK: rocdl.s.setprio 0
//CHECK: tt.load
//CHECK: rocdl.sched.barrier
//CHECK: rocdl.s.setprio 1
//CHECK: tt.dot
//CHECK: rocdl.s.setprio 0

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_small(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %7 = arith.muli %5, %6 : tensor<128x1xi32, #blocked1>
    %8 = tt.addptr %0, %7 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %9 = tt.broadcast %8 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %13 = tt.addptr %9, %12 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %17 = tt.addptr %14, %16 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %18 = tt.broadcast %17 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %19 = tt.splat %arg7 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg10 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg11 = %cst, %arg12 = %13, %arg13 = %20, %arg14 = %c0_i32, %arg15 = %23, %arg16 = %24) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg12, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg13, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg15 : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %31 = ttg.local_load %arg16 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %32 = tt.dot %30, %31, %arg11 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %33 = arith.addi %arg14, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_subview %21[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_subview %22[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %37 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %32, %26, %28, %35, %36, %37 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// CHECK: gpu.barrier
// CHECK: %[[IDX:.+]] = rocdl.workitem.id.x
// CHECK: %[[XDIV:.+]] = arith.divsi %[[IDX]]
// CHECK: %[[WARPLOW:.+]] = arith.cmpi eq, %[[XDIV]]
// CHECK: %[[WARPHIGH:.+]] = arith.cmpi ne, %[[XDIV]]
// CHECK: amdgpu.cond_barrier %[[WARPHIGH]]
// CHECK: scf.for
// CHECK: tt.load
// CHECK: %[[SLICEA0:.+]] = ttg.local_load
// CHECK: %[[SLICEB0:.+]] = ttg.local_load
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT0:.+]] = tt.dot %[[SLICEA0]], %[[SLICEB0]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: tt.load
// CHECK: %[[SLICEA1:.+]] = ttg.local_load
// CHECK: %[[SLICEB1:.+]] = ttg.local_load
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT1:.+]] = tt.dot %[[SLICEA1]], %[[SLICEB1]], %[[DOT0]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: %[[SLICEA2:.+]] = ttg.local_load
// CHECK: %[[SLICEB2:.+]] = ttg.local_load
// CHECK: %[[SLICEA3:.+]] = ttg.local_load
// CHECK: %[[SLICEB3:.+]] = ttg.local_load
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT2:.+]] = tt.dot %[[SLICEA2]], %[[SLICEB2]], %[[DOT1]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: ttg.local_store
// CHECK: ttg.local_store
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: tt.dot %[[SLICEA3]], %[[SLICEB3]], %[[DOT2]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: scf.yield
// CHECK: amdgpu.cond_barrier %[[WARPLOW]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_large(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x256xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #blocked1>
    %7 = arith.muli %5, %6 : tensor<256x1xi32, #blocked1>
    %8 = tt.addptr %0, %7 : tensor<256x1x!tt.ptr<f16>, #blocked1>, tensor<256x1xi32, #blocked1>
    %9 = tt.broadcast %8 : tensor<256x1x!tt.ptr<f16>, #blocked1> -> tensor<256x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x64xi32, #blocked1> -> tensor<256x64xi32, #blocked1>
    %13 = tt.addptr %9, %12 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
    %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %17 = tt.addptr %14, %16 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %18 = tt.broadcast %17 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %19 = tt.splat %arg7 : i32 -> tensor<64x256xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg10 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg11 = %cst, %arg12 = %13, %arg13 = %20, %arg14 = %c0_i32, %arg15 = %23, %arg16 = %24) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg12, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg13, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %29 = tt.load %28 : tensor<64x256x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg15 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg16 : !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %31, %arg11 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
      %33 = arith.addi %arg14, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_subview %21[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_subview %22[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %37 : tensor<64x256xf16, #blocked> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %32, %26, %28, %35, %36, %37 : tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// CHECK: gpu.barrier
// CHECK: %[[IDX:.+]] = rocdl.workitem.id.x
// CHECK: %[[XDIV:.+]] = arith.divsi %[[IDX]]
// CHECK: %[[WARPLOW:.+]] = arith.cmpi eq, %[[XDIV]]
// CHECK: %[[WARPHIGH:.+]] = arith.cmpi ne, %[[XDIV]]
// CHECK: amdgpu.cond_barrier %[[WARPHIGH]]
// CHECK: scf.for

// CHECK: %[[SLICEA0:.+]] = ttg.local_load
// CHECK: %[[SLICEB0:.+]] = ttg.local_load
// CHECK: rocdl.sched.barrier 0
// CHECK: tt.load
// CHECK: rocdl.sched.barrier 0
// CHECK: %[[SLICEA1:.+]] = ttg.local_load
// CHECK: %[[SLICEB1:.+]] = ttg.local_load
// CHECK: rocdl.sched.barrier 0
// CHECK: tt.load
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT0:.+]] = tt.dot %[[SLICEA0]], %[[SLICEB0]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: ttg.local_store
// CHECK: ttg.local_store
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT1:.+]] = tt.dot %[[SLICEA1]], %[[SLICEB1]], %[[DOT0]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: scf.yield
// CHECK: amdgpu.cond_barrier %[[WARPLOW]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #blocked1>
    %7 = arith.muli %5, %6 : tensor<256x1xi32, #blocked1>
    %8 = tt.addptr %0, %7 : tensor<256x1x!tt.ptr<f16>, #blocked1>, tensor<256x1xi32, #blocked1>
    %9 = tt.broadcast %8 : tensor<256x1x!tt.ptr<f16>, #blocked1> -> tensor<256x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x64xi32, #blocked1> -> tensor<256x64xi32, #blocked1>
    %13 = tt.addptr %9, %12 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
    %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %17 = tt.addptr %14, %16 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %18 = tt.broadcast %17 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %19 = tt.splat %arg7 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg10 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg11 = %cst, %arg12 = %13, %arg13 = %20, %arg14 = %c0_i32, %arg15 = %23, %arg16 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg12, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg13, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg15 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg16 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %31, %arg11 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
      %33 = arith.addi %arg14, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_subview %21[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_subview %22[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %37 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %32, %26, %28, %35, %36, %37 : tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: pingpong_medium_cast
// CHECK-COUNT-2: local_load
// CHECK-NOT: setprio
// CHECK-NOT: barrier

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium_cast(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #blocked1>
    %7 = arith.muli %5, %6 : tensor<256x1xi32, #blocked1>
    %8 = tt.addptr %0, %7 : tensor<256x1x!tt.ptr<f16>, #blocked1>, tensor<256x1xi32, #blocked1>
    %9 = tt.broadcast %8 : tensor<256x1x!tt.ptr<f16>, #blocked1> -> tensor<256x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x64xi32, #blocked1> -> tensor<256x64xi32, #blocked1>
    %13 = tt.addptr %9, %12 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
    %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %17 = tt.addptr %14, %16 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %18 = tt.broadcast %17 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %19 = tt.splat %arg7 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x128xi16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x128xi16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg10 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg11 = %cst, %arg12 = %13, %arg13 = %20, %arg14 = %c0_i32, %arg15 = %23, %arg16 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg12, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg13, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %cast2 = tt.bitcast %29 : tensor<64x128xf16, #blocked> -> tensor<64x128xi16, #blocked>
      %30 = ttg.local_load %arg15 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg16 : !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xi16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %cast = tt.bitcast %31 : tensor<64x128xi16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> ->  tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %cast, %arg11 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
      %33 = arith.addi %arg14, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_subview %21[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_subview %22[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x128xi16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %cast2, %37 : tensor<64x128xi16, #blocked> -> !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %32, %26, %28, %35, %36, %37 : tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xi16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}


// -----


// CHECK-LABEL: pingpong_reject
// CHECK-COUNT-2: local_load
// CHECK-NOT: local_load
// CHECK-NOT: setprio
// CHECK-NOT: barrier

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_reject(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<16x256xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x16xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #blocked1>
    %7 = arith.muli %5, %6 : tensor<256x1xi32, #blocked1>
    %8 = tt.addptr %0, %7 : tensor<256x1x!tt.ptr<f16>, #blocked1>, tensor<256x1xi32, #blocked1>
    %9 = tt.broadcast %8 : tensor<256x1x!tt.ptr<f16>, #blocked1> -> tensor<256x16x!tt.ptr<f16>, #blocked1>
    %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x16xi32, #blocked1> -> tensor<256x16xi32, #blocked1>
    %13 = tt.addptr %9, %12 : tensor<256x16x!tt.ptr<f16>, #blocked1>, tensor<256x16xi32, #blocked1>
    %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #blocked>
    %15 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %17 = tt.addptr %14, %16 : tensor<16x1x!tt.ptr<f16>, #blocked>, tensor<16x1xi32, #blocked>
    %18 = tt.broadcast %17 : tensor<16x1x!tt.ptr<f16>, #blocked> -> tensor<16x256x!tt.ptr<f16>, #blocked>
    %19 = tt.splat %arg7 : i32 -> tensor<16x256xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<16x256x!tt.ptr<f16>, #blocked>, tensor<16x256xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x16xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x16x256xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x16xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg10 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg11 = %cst, %arg12 = %13, %arg13 = %20, %arg14 = %c0_i32, %arg15 = %23, %arg16 = %24) -> (tensor<256x256xf32, #mma>, tensor<256x16x!tt.ptr<f16>, #blocked1>, tensor<16x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg12, %cst_1 : tensor<256x16x!tt.ptr<f16>, #blocked1>, tensor<256x16xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x16x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg13, %cst_0 : tensor<16x256x!tt.ptr<f16>, #blocked>, tensor<16x256xi32, #blocked>
      %29 = tt.load %28 : tensor<16x256x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg15 : !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %31 = ttg.local_load %arg16 : !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %32 = tt.dot %30, %31, %arg11 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
      %33 = arith.addi %arg14, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_subview %21[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x16xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x16xf16, #blocked1> -> !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_subview %22[%35, %c0_i32, %c0_i32] : !ttg.memdesc<1x16x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %37 : tensor<16x256xf16, #blocked> -> !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %32, %26, %28, %35, %36, %37 : tensor<256x256xf32, #mma>, tensor<256x16x!tt.ptr<f16>, #blocked1>, tensor<16x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x16xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x16x256xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
// CHECK-LABEL: pingpong_medium_epilogue


// CHECK: gpu.barrier
// CHECK: %[[IDX:.+]] = rocdl.workitem.id.x
// CHECK: %[[XDIV:.+]] = arith.divsi %[[IDX]]
// CHECK: %[[WARPLOW:.+]] = arith.cmpi eq, %[[XDIV]]
// CHECK: %[[WARPHIGH:.+]] = arith.cmpi ne, %[[XDIV]]
// CHECK: amdgpu.cond_barrier %[[WARPHIGH]]
// CHECK: scf.for

// CHECK: %[[SLICEA0:.+]] = ttg.local_load
// CHECK: %[[SLICEB0:.+]] = ttg.local_load
// CHECK: rocdl.sched.barrier 0
// CHECK: tt.load
// CHECK: rocdl.sched.barrier 0
// CHECK: %[[SLICEA1:.+]] = ttg.local_load
// CHECK: %[[SLICEB1:.+]] = ttg.local_load
// CHECK: rocdl.sched.barrier 0
// CHECK: tt.load
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT0:.+]] = tt.dot %[[SLICEA0]], %[[SLICEB0]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: ttg.local_store
// CHECK: ttg.local_store
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT1:.+]] = tt.dot %[[SLICEA1]], %[[SLICEB1]], %[[DOT0]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: scf.yield
// CHECK: amdgpu.cond_barrier %[[WARPLOW]]

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium_epilogue(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %2 = tt.get_program_id x : i32
    %c255_i32 = arith.constant 255 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %c256_i32 = arith.constant 256 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %c8_i32 = arith.constant 8 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %2, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %c127_i32 = arith.constant 127 : i32
    %8 = arith.addi %arg3, %c127_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %9 = arith.divsi %8, %c128_i32 : i32
    %10 = arith.subi %9, %7 : i32
    %11 = arith.minsi %10, %c8_i32 : i32
    %12 = arith.remsi %2, %11 : i32
    %13 = arith.addi %7, %12 : i32
    %14 = arith.muli %13, %c128_i32 : i32
    %15 = arith.subi %arg3, %14 : i32
    %16 = tt.splat %15 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %17 = arith.cmpi slt, %1, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %cst = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %18 = arith.select %17, %1, %cst {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %20 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %21 = arith.muli %19, %20 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %22 = tt.broadcast %21 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %25 = tt.broadcast %24 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %26 = arith.addi %22, %25 : tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %27 = tt.addptr %0, %26 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %c63_i32 = arith.constant 63 : i32
    %28 = arith.addi %arg5, %c63_i32 : i32
    %c64_i32 = arith.constant 64 : i32
    %29 = arith.divsi %28, %c64_i32 : i32
    %30 = arith.muli %9, %4 : i32
    %c304_i32 = arith.constant 304 : i32
    %31 = arith.divsi %30, %c304_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %35 = arith.muli %29, %31 : i32
    %c0_i32 = arith.constant 0 : i32
    %36 = arith.cmpi sgt, %35, %c0_i32 : i32
    %37 = tt.splat %36 : i1 -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %38 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %39 = tt.expand_dims %38 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %40 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %41 = arith.cmpi slt, %39, %40 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %42 = tt.broadcast %41 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %43 = arith.andi %37, %42 : tensor<128x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %44 = tt.load %27, %43, %cst_0 {OpIdx = #amdgpu.OpIdx<0>} : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %45 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %46 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %47 = tt.expand_dims %46 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %48 = tt.broadcast %47 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %49 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %50 = arith.remsi %2, %5 : i32
    %51 = arith.divsi %50, %11 : i32
    %52 = arith.muli %51, %c256_i32 : i32
    %53 = arith.subi %arg4, %52 : i32
    %54 = tt.splat %53 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %55 = arith.cmpi slt, %49, %54 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %cst_1 = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %56 = arith.select %55, %49, %cst_1 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %57 = tt.expand_dims %56 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %58 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %59 = arith.muli %57, %58 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %60 = tt.broadcast %59 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %61 = arith.addi %48, %60 : tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %62 = tt.addptr %45, %61 : tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>, tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %63 = tt.splat %36 : i1 -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %66 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %67 = arith.cmpi slt, %65, %66 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %68 = tt.broadcast %67 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %69 = arith.andi %63, %68 : tensor<64x256xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %70 = tt.load %62, %69, %cst_2 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
    %71 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
    %72 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
    %73 = arith.subi %29, %c1_i32 : i32
    %74 = ttg.local_alloc  : () -> !ttg.memdesc<1x128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %75 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %76 = ttg.memdesc_subview %74[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
    ttg.local_store %44, %76 {OpIdx = #amdgpu.OpIdx<0>} : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %77 = ttg.memdesc_subview %75[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    ttg.local_store %70, %77 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %78 = arith.subi %35, %c1_i32 : i32
    %79:13 = scf.for %arg9 = %c0_i32 to %78 step %c1_i32 iter_args(%arg10 = %c0_i32, %arg11 = %2, %arg12 = %13, %arg13 = %51, %arg14 = %cst_3, %arg15 = %18, %arg16 = %56, %arg17 = %c0_i32, %arg18 = %76, %arg19 = %77, %arg20 = %c0_i32, %arg21 = %13, %arg22 = %51) -> (i32, i32, i32, i32, tensor<128x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>, i32, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>, i32, i32, i32)  : i32 {
      %86 = arith.cmpi eq, %arg10, %73 : i32
      %87 = arith.addi %arg10, %c1_i32 : i32
      %88 = arith.select %86, %c0_i32, %87 : i32
      %89 = arith.cmpi eq, %88, %c0_i32 : i32
      %91 = tt.expand_dims %arg15 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %92 = arith.muli %91, %20 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %93 = tt.broadcast %92 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %94 = arith.muli %88, %c64_i32 : i32
      %95 = tt.splat %94 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
      %96 = arith.addi %95, %23 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
      %97 = tt.expand_dims %96 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %98 = tt.broadcast %97 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %99 = arith.addi %93, %98 : tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %100 = tt.addptr %0, %99 : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>, tensor<128x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %101 = arith.subi %arg5, %94 : i32
      %102 = tt.splat %101 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %103 = arith.cmpi slt, %39, %102 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %104 = tt.broadcast %103 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<128x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %105 = tt.load %100, %104, %cst_0 {OpIdx = #amdgpu.OpIdx<0>} : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %106 = tt.splat %94 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
      %107 = arith.addi %106, %46 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
      %108 = tt.expand_dims %107 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %109 = tt.broadcast %108 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %110 = tt.expand_dims %arg16 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %111 = arith.muli %110, %58 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %112 = tt.broadcast %111 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %113 = arith.addi %109, %112 : tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %114 = tt.addptr %45, %113 : tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>, tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %115 = tt.splat %101 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %116 = arith.cmpi slt, %65, %115 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %117 = tt.broadcast %116 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %118 = ttg.local_load %arg18 : !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
      %119 = ttg.local_load %arg19 : !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
      %120 = tt.load %114, %117, %cst_2 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %121 = tt.dot %118, %119, %arg14, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<128x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
      %122 = arith.cmpi eq, %arg20, %73 : i32
      scf.if %122 {
        %129 = arith.muli %arg21, %c128_i32 : i32
        %130 = tt.splat %129 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
        %132 = arith.muli %arg22, %c256_i32 : i32
        %133 = tt.splat %132 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
        %134 = arith.addi %133, %72 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
        %135 = tt.expand_dims %130 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %136 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %137 = arith.muli %136, %135 : tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %138 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %139 = tt.addptr %138, %137 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>, tensor<128x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %140 = tt.expand_dims %134 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<1x256xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %141 = tt.broadcast %139 : tensor<128x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x256x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %142 = tt.broadcast %140 : tensor<1x256xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>> -> tensor<128x256xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %143 = tt.addptr %141, %142 : tensor<128x256x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>, tensor<128x256xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %151 = arith.truncf %121 : tensor<128x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>> to tensor<128x256xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        tt.store %143, %151 : tensor<128x256x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
      }
      %124 = arith.addi %arg17, %c1_i32 : i32
      %125 = arith.cmpi slt, %124, %c1_i32 : i32
      %126 = arith.select %125, %124, %c0_i32 : i32
      %127 = ttg.memdesc_subview %74[%126, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
      ttg.local_store %105, %127 {OpIdx = #amdgpu.OpIdx<0>} : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %128 = ttg.memdesc_subview %75[%126, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
      ttg.local_store %120, %128 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
      scf.yield %88, %arg11, %arg12, %arg13, %121, %arg15, %arg16, %126, %127, %128, %88, %arg12, %arg13 : i32, i32, i32, i32, tensor<128x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>, i32, !ttg.memdesc<128x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>, i32, i32, i32
    }
    tt.return
  }
}

// -----
// CHECK-LABEL: pingpong_large_epilogue

// CHECK: gpu.barrier
// CHECK: %[[IDX:.+]] = rocdl.workitem.id.x
// CHECK: %[[XDIV:.+]] = arith.divsi %[[IDX]]
// CHECK: %[[WARPLOW:.+]] = arith.cmpi eq, %[[XDIV]]
// CHECK: %[[WARPHIGH:.+]] = arith.cmpi ne, %[[XDIV]]
// CHECK: amdgpu.cond_barrier %[[WARPHIGH]]
// CHECK: scf.for
// CHECK: tt.load
// CHECK: %[[SLICEA0:.+]] = ttg.local_load
// CHECK: %[[SLICEB0:.+]] = ttg.local_load
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT0:.+]] = tt.dot %[[SLICEA0]], %[[SLICEB0]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: tt.load
// CHECK: %[[SLICEA1:.+]] = ttg.local_load
// CHECK: %[[SLICEB1:.+]] = ttg.local_load
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT1:.+]] = tt.dot %[[SLICEA1]], %[[SLICEB1]], %[[DOT0]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: %[[SLICEA2:.+]] = ttg.local_load
// CHECK: %[[SLICEB2:.+]] = ttg.local_load
// CHECK: %[[SLICEA3:.+]] = ttg.local_load
// CHECK: %[[SLICEB3:.+]] = ttg.local_load
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: %[[DOT2:.+]] = tt.dot %[[SLICEA2]], %[[SLICEB2]], %[[DOT1]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: ttg.local_store
// CHECK: ttg.local_store
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.setprio 1
// CHECK: tt.dot %[[SLICEA3]], %[[SLICEB3]], %[[DOT2]]
// CHECK: rocdl.s.setprio 0
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: scf.yield
// CHECK: amdgpu.cond_barrier %[[WARPLOW]]

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_large_epilogue(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %2 = tt.get_program_id x : i32
    %c255_i32 = arith.constant 255 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %c256_i32 = arith.constant 256 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %c8_i32 = arith.constant 8 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %2, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.addi %arg3, %c255_i32 : i32
    %9 = arith.divsi %8, %c256_i32 : i32
    %10 = arith.subi %9, %7 : i32
    %11 = arith.minsi %10, %c8_i32 : i32
    %12 = arith.remsi %2, %11 : i32
    %13 = arith.addi %7, %12 : i32
    %14 = arith.muli %13, %c256_i32 : i32
    %15 = arith.subi %arg3, %14 : i32
    %16 = tt.splat %15 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %17 = arith.cmpi slt, %1, %16 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %18 = arith.select %17, %1, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>, tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<256x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %20 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %21 = arith.muli %19, %20 : tensor<256x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %22 = tt.broadcast %21 : tensor<256x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<256x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %25 = tt.broadcast %24 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<256x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %26 = arith.addi %22, %25 : tensor<256x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %27 = tt.addptr %0, %26 : tensor<256x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>, tensor<256x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %c63_i32 = arith.constant 63 : i32
    %28 = arith.addi %arg5, %c63_i32 : i32
    %c64_i32 = arith.constant 64 : i32
    %29 = arith.divsi %28, %c64_i32 : i32
    %30 = arith.muli %9, %4 : i32
    %c304_i32 = arith.constant 304 : i32
    %31 = arith.divsi %30, %c304_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %35 = arith.muli %29, %31 : i32
    %c0_i32 = arith.constant 0 : i32
    %36 = arith.cmpi sgt, %35, %c0_i32 : i32
    %37 = tt.splat %36 : i1 -> tensor<256x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %38 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %39 = tt.expand_dims %38 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %40 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %41 = arith.cmpi slt, %39, %40 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %42 = tt.broadcast %41 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<256x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %43 = arith.andi %37, %42 : tensor<256x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %44 = tt.load %27, %43, %cst_0 {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %45 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %46 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %47 = tt.expand_dims %46 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %48 = tt.broadcast %47 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %49 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %50 = arith.remsi %2, %5 : i32
    %51 = arith.divsi %50, %11 : i32
    %52 = arith.muli %51, %c256_i32 : i32
    %53 = arith.subi %arg4, %52 : i32
    %54 = tt.splat %53 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %55 = arith.cmpi slt, %49, %54 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %cst_1 = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %56 = arith.select %55, %49, %cst_1 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %57 = tt.expand_dims %56 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %58 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %59 = arith.muli %57, %58 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %60 = tt.broadcast %59 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %61 = arith.addi %48, %60 : tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %62 = tt.addptr %45, %61 : tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>, tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %63 = tt.splat %36 : i1 -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %66 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %67 = arith.cmpi slt, %65, %66 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %68 = tt.broadcast %67 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %69 = arith.andi %63, %68 : tensor<64x256xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %70 = tt.load %62, %69, %cst_2 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
    %71 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
    %72 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
    %73 = arith.subi %29, %c1_i32 : i32
    %74 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %75 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %76 = ttg.memdesc_subview %74[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
    ttg.local_store %44, %76 {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> !ttg.memdesc<256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %77 = ttg.memdesc_subview %75[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    ttg.local_store %70, %77 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %78 = arith.subi %35, %c1_i32 : i32
    %79:13 = scf.for %arg9 = %c0_i32 to %78 step %c1_i32 iter_args(%arg10 = %c0_i32, %arg11 = %2, %arg12 = %13, %arg13 = %51, %arg14 = %cst_3, %arg15 = %18, %arg16 = %56, %arg17 = %c0_i32, %arg18 = %76, %arg19 = %77, %arg20 = %c0_i32, %arg21 = %13, %arg22 = %51) -> (i32, i32, i32, i32, tensor<256x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>, tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>, i32, !ttg.memdesc<256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>, i32, i32, i32)  : i32 {
      %86 = arith.cmpi eq, %arg10, %73 : i32
      %87 = arith.addi %arg10, %c1_i32 : i32
      %88 = arith.select %86, %c0_i32, %87 : i32
      %89 = arith.cmpi eq, %88, %c0_i32 : i32
      %91 = tt.expand_dims %arg15 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<256x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %92 = arith.muli %91, %20 : tensor<256x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %93 = tt.broadcast %92 : tensor<256x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<256x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %94 = arith.muli %88, %c64_i32 : i32
      %95 = tt.splat %94 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
      %96 = arith.addi %95, %23 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
      %97 = tt.expand_dims %96 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %98 = tt.broadcast %97 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<256x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %99 = arith.addi %93, %98 : tensor<256x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %100 = tt.addptr %0, %99 : tensor<256x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>, tensor<256x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %105 = tt.load %100 {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
      %106 = tt.splat %94 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
      %107 = arith.addi %106, %46 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>
      %108 = tt.expand_dims %107 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %109 = tt.broadcast %108 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %110 = tt.expand_dims %arg16 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %111 = arith.muli %110, %58 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %112 = tt.broadcast %111 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %113 = arith.addi %109, %112 : tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %114 = tt.addptr %45, %113 : tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>, tensor<64x256xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %118 = ttg.local_load %arg18 : !ttg.memdesc<256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
      %119 = ttg.local_load %arg19 : !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
      %120 = tt.load %114 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>>
      %121 = tt.dot %118, %119, %arg14, inputPrecision = tf32 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<256x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
      %122 = arith.cmpi eq, %arg20, %73 : i32
      scf.if %122 {
        %129 = arith.muli %arg21, %c256_i32 : i32
        %130 = tt.splat %129 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
        %132 = arith.muli %arg22, %c256_i32 : i32
        %133 = tt.splat %132 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
        %134 = arith.addi %133, %72 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>>
        %135 = tt.expand_dims %130 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<256x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %136 = tt.splat %arg8 : i32 -> tensor<256x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %137 = arith.muli %136, %135 : tensor<256x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %138 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %139 = tt.addptr %138, %137 : tensor<256x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>, tensor<256x1xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %140 = tt.expand_dims %134 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>}>> -> tensor<1x256xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %141 = tt.broadcast %139 : tensor<256x1x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>> -> tensor<256x256x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %142 = tt.broadcast %140 : tensor<1x256xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>> -> tensor<256x256xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %143 = tt.addptr %141, %142 : tensor<256x256x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>, tensor<256x256xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        %151 = arith.truncf %121 : tensor<256x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>> to tensor<256x256xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
        tt.store %143, %151 : tensor<256x256x!tt.ptr<f16>, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>
      }
      %124 = arith.addi %arg17, %c1_i32 : i32
      %125 = arith.cmpi slt, %124, %c1_i32 : i32
      %126 = arith.select %125, %124, %c0_i32 : i32
      %127 = ttg.memdesc_subview %74[%126, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
      ttg.local_store %105, %127 {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> !ttg.memdesc<256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %128 = ttg.memdesc_subview %75[%126, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
      ttg.local_store %120, %128 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256xf16, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>> -> !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>
      scf.yield %88, %arg11, %arg12, %arg13, %121, %arg15, %arg16, %126, %127, %128, %88, %arg12, %arg13 : i32, i32, i32, i32, tensor<256x256xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>>, tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>}>>, i32, !ttg.memdesc<256x64xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>, #ttg.shared_memory, mutable>, i32, i32, i32
    }
    tt.return
  }
}
