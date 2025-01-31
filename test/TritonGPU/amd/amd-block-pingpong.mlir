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
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
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
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
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
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
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
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
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
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
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
