// RUN: triton-opt %s -split-input-file --tritonamdgpu-block-pingpong="num-stages=2" | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonamdgpu-block-pingpong="num-stages=3" | FileCheck %s --check-prefixes CHECK-NS3

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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_small(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
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
    %6 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared1, #smem, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>)  : i32 {
      %26 = tt.addptr %arg7, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg10 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %31 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %32 = arith.negf %31 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %33 = tt.dot %30, %32, %arg6 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %34 = arith.addi %arg9, %c1_i32 : i32
      %35 = arith.cmpi slt, %34, %c1_i32 : i32
      %36 = arith.select %35, %34, %c0_i32 : i32
      %37 = ttg.memdesc_index %21[%36] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %27, %37 : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %38 = ttg.memdesc_index %22[%36] : !ttg.memdesc<1x64x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
      ttg.local_store %29, %38 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
      scf.yield %33, %26, %28, %36, %37, %38 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #smem, mutable>
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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_large(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
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
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x256xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg8, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %29 = tt.load %28 : tensor<64x256x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg11 : !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %31, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
      %33 = arith.addi %arg9, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_index %21[%35] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_index %22[%35] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
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
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %31, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
      %33 = arith.addi %arg9, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_index %21[%35] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_index %22[%35] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium_cast(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
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
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xi16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xi16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %cast2 = tt.bitcast %29 : tensor<64x128xf16, #blocked> -> tensor<64x128xi16, #blocked>
      %30 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg11 : !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xi16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %cast = tt.bitcast %31 : tensor<64x128xi16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> ->  tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %cast, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
      %33 = arith.addi %arg9, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_index %21[%35] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_index %22[%35] : !ttg.memdesc<1x64x128xi16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xi16, #shared1, #ttg.shared_memory, mutable>
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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_reject(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
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
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<16x256xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<16x256x!tt.ptr<f16>, #blocked>, tensor<16x256xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256x16xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x16x256xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x16xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x16x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x256xf32, #mma>, tensor<256x16x!tt.ptr<f16>, #blocked1>, tensor<16x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg7, %cst_1 : tensor<256x16x!tt.ptr<f16>, #blocked1>, tensor<256x16xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x16x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg8, %cst_0 : tensor<16x256x!tt.ptr<f16>, #blocked>, tensor<16x256xi32, #blocked>
      %29 = tt.load %28 : tensor<16x256x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg10 : !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %31 = ttg.local_load %arg11 : !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %32 = tt.dot %30, %31, %arg6 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
      %33 = arith.addi %arg9, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_index %21[%35] : !ttg.memdesc<1x256x16xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x16xf16, #blocked1> -> !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_index %22[%35] : !ttg.memdesc<1x16x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %37 : tensor<16x256xf16, #blocked> -> !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %32, %26, %28, %35, %36, %37 : tensor<256x256xf32, #mma>, tensor<256x16x!tt.ptr<f16>, #blocked1>, tensor<16x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x16xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<16x256xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x16xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x16x256xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: pingpong_small_prologue_load
// CHECK-NOT: setprio

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_small_prologue_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = arith.cmpi eq, %arg5, %c0_i32: i32
      %27 = scf.if %26 -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> {
        %28 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
        %29 = tt.broadcast %28 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
        %30 = tt.load %29 : tensor<128x64x!tt.ptr<f16>, #blocked1>
        %31 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
        %32 = ttg.memdesc_index %31[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
        ttg.local_store %30, %32 : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
        %33 = ttg.local_load %32 : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
        scf.yield %33 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      } else {
        scf.yield %cst_2 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      }
      %34 = tt.addptr %arg7, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %35 = tt.load %34 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %36 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %37 = tt.load %36 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %38 = ttg.local_load %arg10 : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %39 = arith.addf %38, %27: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %40 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %41 = tt.dot %39, %40, %arg6 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %42 = arith.addi %arg9, %c1_i32 : i32
      %43 = arith.cmpi slt, %42, %c1_i32 : i32
      %44 = arith.select %43, %42, %c0_i32 : i32
      %45 = ttg.memdesc_index %21[%44] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %35, %45 : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      %46 = ttg.memdesc_index %22[%44] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %37, %46 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %41, %34, %36, %44, %45, %46 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}


// -----
// CHECK-LABEL: pingpong_medium_dependency

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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium_dependency(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %31, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
      %33 = arith.addf %32, %cst_2 : tensor<256x128xf32, #mma>
      %34 = arith.addi %arg9, %c1_i32 : i32
      %35 = arith.cmpi slt, %34, %c1_i32 : i32
      %36 = arith.select %35, %34, %c0_i32 : i32
      %37 = ttg.memdesc_index %21[%36] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %37 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %38 = ttg.memdesc_index %22[%36] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %38 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %33, %26, %28, %36, %37, %38 : tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
// CHECK-LABEL: pingpong_large_dependency

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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_large_dependency(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x256xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x256xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63: i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x256xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg8, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %29 = tt.load %28 : tensor<64x256x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg11 : !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %31, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
      %33 = arith.addf %32, %cst_2 : tensor<256x256xf32, #mma>
      %34 = arith.addi %arg9, %c1_i32 : i32
      %35 = arith.cmpi slt, %34, %c1_i32 : i32
      %36 = arith.select %35, %34, %c0_i32 : i32
      %37 = ttg.memdesc_index %21[%36] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %37 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %38 = ttg.memdesc_index %22[%36] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %38 : tensor<64x256xf16, #blocked> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %33, %26, %28, %36, %37, %38 : tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}
// -----
//CHECK-LABEL: pingpong_small_load_reorder
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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_small_load_reorder(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
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
    %6 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      // This swaps the assumption on the ordering of the local load and
      // global load from the base test to ensure the one ping pong cluster
      // is robust to different patterns.
      %26 = ttg.local_load %arg10 : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %27 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %28 = tt.addptr %arg7, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %29 = tt.load %28 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %30 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %31 = tt.load %30 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %32 = tt.dot %26, %27, %arg6 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %33 = arith.addi %arg9, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_index %21[%35] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %29, %36 : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_index %22[%35] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %31, %37 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %32, %28, %30, %35, %36, %37 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}


// -----
//CHECK-LABEL: pingpong_small_local_load_dep
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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_small_local_load_dep(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg7, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg10 : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %31 = arith.addf %30, %cst_2 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %32 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %33 = tt.dot %31, %32, %arg6 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %34 = arith.addi %arg9, %c1_i32 : i32
      %35 = arith.cmpi slt, %34, %c1_i32 : i32
      %36 = arith.select %35, %34, %c0_i32 : i32
      %37 = ttg.memdesc_index %21[%36] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %37 : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      %38 = ttg.memdesc_index %22[%36] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %38 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %33, %26, %28, %36, %37, %38 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
//CHECK-LABEL: pingpong_medium_load_iter
//CHECK: ttg.local_load
//CHECK: ttg.local_load
//CHECK: rocdl.sched.barrier
//CHECK: tt.load
//CHECK: rocdl.sched.barrier
//CHECK: ttg.local_load
//CHECK: ttg.local_load
//CHECK: rocdl.sched.barrier
//CHECK: tt.load
//CHECK: rocdl.s.barrier
//CHECK: rocdl.sched.barrier
//CHECK: rocdl.s.setprio 1
//CHECK: tt.dot
//CHECK: rocdl.s.setprio 0
//CHECK: gpu.barrier
//CHECK: rocdl.sched.barrier
//CHECK: ttg.local_store
//CHECK: ttg.local_store
//CHECK: gpu.barrier
//CHECK: rocdl.sched.barrier
//CHECK: rocdl.s.setprio 1
//CHECK: tt.dot
//CHECK: rocdl.s.setprio 0
//CHECK: gpu.barrier
//CHECK: rocdl.sched.barrier

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @pingpong_medium_load_iter(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c192_i32 = arith.constant 192 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<1024> : tensor<64x1xi64, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked1>
    %3 = tt.splat %1 : i64 -> tensor<256x64xi64, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.extsi %4 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %7 = arith.extsi %5 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %8 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %9 = tt.splat %1 : i64 -> tensor<64x128xi64, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared1, #smem, mutable>
    %12 = tt.load %2 : tensor<256x64x!tt.ptr<f16>, #blocked1>
    %13 = tt.load %8 : tensor<64x128x!tt.ptr<f16>, #blocked>
    %14 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    ttg.local_store %12, %14 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %15 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
    ttg.local_store %13, %15 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
    %16:6 = scf.for %arg3 = %c0_i32 to %c192_i32 step %c64_i32 iter_args(%arg4 = %c0_i64, %arg5 = %c0_i64, %arg6 = %cst, %arg7 = %c0_i32, %arg8 = %14, %arg9 = %15) -> (i64, i64, tensor<256x128xf32, #mma>, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>)  : i32 {
      %22 = arith.addi %arg4, %c64_i64 : i64
      %23 = arith.addi %arg5, %c64_i64 : i64
      %24 = tt.splat %22 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %25 = arith.addi %24, %6 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi64, #blocked1>
      %27 = tt.broadcast %26 : tensor<1x64xi64, #blocked1> -> tensor<256x64xi64, #blocked1>
      %28 = arith.addi %3, %27 : tensor<256x64xi64, #blocked1>
      %29 = tt.addptr %2, %28 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi64, #blocked1>
      %30 = tt.load %29 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %31 = ttg.local_load %arg8 : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %32 = tt.splat %23 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %33 = arith.addi %32, %7 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %34 = tt.expand_dims %33 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
      %35 = arith.muli %34, %cst_0 : tensor<64x1xi64, #blocked>
      %36 = tt.broadcast %35 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %37 = arith.addi %36, %9 : tensor<64x128xi64, #blocked>
      %38 = tt.addptr %8, %37 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %39 = tt.load %38 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %40 = ttg.local_load %arg9 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %41 = tt.dot %31, %40, %arg6, inputPrecision = tf32 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
      %42 = arith.addi %arg7, %c1_i32 : i32
      %43 = arith.cmpi slt, %42, %c1_i32 : i32
      %44 = arith.select %43, %42, %c0_i32 : i32
      %45 = ttg.memdesc_index %10[%44] : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      ttg.local_store %30, %45 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %46 = ttg.memdesc_index %11[%44] : !ttg.memdesc<1x64x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
      ttg.local_store %39, %46 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
      scf.yield %22, %23, %41, %44, %45, %46 : i64, i64, tensor<256x128xf32, #mma>, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
    }
    %17 = ttg.local_load %16#4 : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %18 = ttg.local_load %16#5 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %19 = tt.dot %17, %18, %16#2, inputPrecision = tf32 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    ttg.local_dealloc %10 : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %11 : !ttg.memdesc<1x64x128xf16, #shared1, #smem, mutable>
    %20 = arith.truncf %19 : tensor<256x128xf32, #mma> to tensor<256x128xf16, #mma>
    %21 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #mma>
    tt.store %21, %20 : tensor<256x128x!tt.ptr<f16>, #mma>
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
// CHECK: scf.if
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: scf.yield
// CHECK: amdgpu.cond_barrier %[[WARPLOW]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium_epilogue(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg2 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg3 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg4 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg5 = %cst, %arg6 = %13, %arg7 = %20, %arg8 = %c0_i32, %arg9 = %23, %arg10 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg6, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg7, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %29 = tt.load %28 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg9 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg10 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %31, %arg5 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
      %33 = arith.addi %arg8, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_index %21[%35] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_index %22[%35] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %37 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      %38 = arith.cmpi eq, %arg4, %c63_i32: i32
      %39 = scf.if %38 -> tensor<256x128xf32, #mma> {
        %40 = arith.addf %32, %cst_2: tensor<256x128xf32, #mma>
        scf.yield %40: tensor<256x128xf32, #mma>
      } else {
        scf.yield %32: tensor<256x128xf32, #mma>
      }
      scf.yield %39, %26, %28, %35, %36, %37 : tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
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
// CHECK: scf.if
// CHECK: gpu.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: scf.yield
// CHECK: amdgpu.cond_barrier %[[WARPLOW]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_large_epilogue(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x256xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x256xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg2 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg3 : i32 -> tensor<64x256xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %25:6 = scf.for %arg4 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg5 = %cst, %arg6 = %13, %arg7 = %20, %arg8 = %c0_i32, %arg9 = %23, %arg10 = %24) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %26 = tt.addptr %arg6, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %27 = tt.load %26 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %28 = tt.addptr %arg7, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %29 = tt.load %28 : tensor<64x256x!tt.ptr<f16>, #blocked>
      %30 = ttg.local_load %arg9 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %31 = ttg.local_load %arg10 : !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %32 = tt.dot %30, %31, %arg5 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
      %33 = arith.addi %arg8, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = ttg.memdesc_index %21[%35] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %36 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_index %22[%35] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %29, %37 : tensor<64x256xf16, #blocked> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
      %38 = arith.cmpi eq, %arg4, %c63_i32: i32
      %39 = scf.if %38 -> tensor<256x256xf32, #mma> {
        %40 = arith.addf %32, %cst_2: tensor<256x256xf32, #mma>
        scf.yield %40: tensor<256x256xf32, #mma>
      } else {
        scf.yield %32: tensor<256x256xf32, #mma>
      }
      scf.yield %39, %26, %28, %35, %36, %37 : tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
// CHECK-LABEL: pingpong_reject_small_three_load
// CHECK-COUNT-2: local_load
// CHECK-NOT: setprio
// CHECK-NOT: barrier


#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_reject_small_three_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #mma>
    %26 = tt.broadcast %25 : tensor<128x1x!tt.ptr<f32>, #mma> -> tensor<128x128x!tt.ptr<f32>, #mma>
    %27 = tt.load %26: tensor<128x128x!tt.ptr<f32>, #mma>
    %28:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %29 = tt.addptr %arg7, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %30 = tt.load %29 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %31 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %32 = tt.load %31 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %33 = ttg.local_load %arg10 : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %34 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %35 = tt.dot %33, %34, %arg6 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %36 = ttg.local_alloc  : () -> !ttg.memdesc<1x128x128xf32, #shared, #ttg.shared_memory, mutable>
      %37 = ttg.memdesc_index %36[%c0_i32] : !ttg.memdesc<1x128x128xf32, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %27, %37 : tensor<128x128xf32, #mma> -> !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory, mutable>
      %38 = ttg.local_load %37 : !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128x128xf32, #mma>
      %39 = arith.addf %35, %38: tensor<128x128xf32, #mma>
      %40 = arith.addi %arg9, %c1_i32 : i32
      %41 = arith.cmpi slt, %40, %c1_i32 : i32
      %42 = arith.select %41, %40, %c0_i32 : i32
      %43 = ttg.memdesc_index %21[%42] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %30, %43 : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      %44 = ttg.memdesc_index %22[%42] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %32, %44 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %39, %29, %31, %42, %43, %44: tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}


// -----
// CHECK-LABEL: pingpong_small_persistent_epilogue_load
// CHECK: ttg.local_load
// CHECK: rocdl.s.setprio 1
// CHECK: tt.load
// CHECK: rocdl.sched.barrier
// CHECK: ttg.local_load
// CHECK: rocdl.s.setprio 0
// CHECK: tt.load
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.setprio 1
// CHECK: tt.dot
// CHECK: rocdl.s.setprio 0

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_small_persistent_epilogue_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #mma>
    %26 = tt.broadcast %25 : tensor<128x1x!tt.ptr<f32>, #mma> -> tensor<128x128x!tt.ptr<f32>, #mma>
    %27 = tt.load %26: tensor<128x128x!tt.ptr<f32>, #mma>
    %28:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %29 = arith.cmpi eq, %arg5, %c0_i32: i32
      %30 = scf.if %29 -> i32 {
        scf.yield %c0_i32 : i32
      } else {
        scf.yield %arg5 : i32
      }
      %31 = tt.addptr %arg7, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %32 = tt.load %31 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %33 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %34 = tt.load %33 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %35 = ttg.local_load %arg10 : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %36 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %37 = tt.dot %35, %36, %arg6 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %38 = arith.cmpi eq, %30, %c63_i32: i32
      %39 = scf.if %38 -> tensor<128x128xf32, #mma> {
        %40 = ttg.local_alloc  : () -> !ttg.memdesc<1x128x128xf32, #shared, #ttg.shared_memory, mutable>
        %41 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x128x128xf32, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory, mutable>
        ttg.local_store %27, %41 : tensor<128x128xf32, #mma> -> !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory, mutable>
        %42 = ttg.local_load %41 : !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128x128xf32, #mma>
        %43 = arith.addf %37, %42: tensor<128x128xf32, #mma>
        scf.yield %43 : tensor<128x128xf32, #mma>
      } else {
        scf.yield %37 : tensor<128x128xf32, #mma>
      }
      %44 = arith.addi %arg9, %c1_i32 : i32
      %45 = arith.cmpi slt, %44, %c1_i32 : i32
      %46 = arith.select %45, %44, %c0_i32 : i32
      %47 = ttg.memdesc_index %21[%46] : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %32, %47 : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>
      %48 = ttg.memdesc_index %22[%46] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %34, %48 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %39, %31, %33, %46, %47, %48: tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x128x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
// CHECK-LABEL: pingpong_medium_persistent_epilogue_load
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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium_persistent_epilogue_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x1x!tt.ptr<f32>, #mma>
    %26 = tt.broadcast %25 : tensor<256x1x!tt.ptr<f32>, #mma> -> tensor<256x128x!tt.ptr<f32>, #mma>
    %27 = tt.load %26: tensor<256x128x!tt.ptr<f32>, #mma>
    %28:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %29 = arith.cmpi eq, %arg5, %c0_i32: i32
      %30 = scf.if %29 -> i32 {
        scf.yield %c0_i32 : i32
      } else {
        scf.yield %arg5 : i32
      }
      %31 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %32 = tt.load %31 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %33 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %34 = tt.load %33 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %35 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %36 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %37 = tt.dot %35, %36, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x128xf32, #mma>
      %38 = arith.cmpi eq, %30, %c63_i32: i32
      %39 = scf.if %38 -> tensor<256x128xf32, #mma> {
        %40 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x128xf32, #shared, #ttg.shared_memory, mutable>
        %41 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x256x128xf32, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable>
        ttg.local_store %27, %41 : tensor<256x128xf32, #mma> -> !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable>
        %42 = ttg.local_load %41 : !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<256x128xf32, #mma>
        %43 = arith.addf %37, %42: tensor<256x128xf32, #mma>
        scf.yield %43 : tensor<256x128xf32, #mma>
      } else {
        scf.yield %37 : tensor<256x128xf32, #mma>
      }
      %44 = arith.addi %arg9, %c1_i32 : i32
      %45 = arith.cmpi slt, %44, %c1_i32 : i32
      %46 = arith.select %45, %44, %c0_i32 : i32
      %47 = ttg.memdesc_index %21[%46] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %32, %47 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %48 = ttg.memdesc_index %22[%46] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %34, %48 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %39, %31, %33, %46, %47, %48: tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}


// -----
// CHECK-LABEL: pingpong_large_persistent_epilogue_load
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
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_large_persistent_epilogue_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x256xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x256xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x1x!tt.ptr<f32>, #mma>
    %26 = tt.broadcast %25 : tensor<256x1x!tt.ptr<f32>, #mma> -> tensor<256x256x!tt.ptr<f32>, #mma>
    %27 = tt.load %26: tensor<256x256x!tt.ptr<f32>, #mma>
    %28:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %29 = arith.cmpi eq, %arg5, %c0_i32: i32
      %30 = scf.if %29 -> i32 {
        scf.yield %c0_i32 : i32
      } else {
        scf.yield %arg5 : i32
      }
      %31 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %32 = tt.load %31 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %33 = tt.addptr %arg8, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %34 = tt.load %33 : tensor<64x256x!tt.ptr<f16>, #blocked>
      %35 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %36 = ttg.local_load %arg11 : !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %37 = tt.dot %35, %36, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
      %38 = arith.cmpi eq, %30, %c63_i32: i32
      %39 = scf.if %38 -> tensor<256x256xf32, #mma> {
        %40 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x256xf32, #shared, #ttg.shared_memory, mutable>
        %41 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x256x256xf32, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x256xf32, #shared, #ttg.shared_memory, mutable>
        ttg.local_store %27, %41 : tensor<256x256xf32, #mma> -> !ttg.memdesc<256x256xf32, #shared, #ttg.shared_memory, mutable>
        %42 = ttg.local_load %41 : !ttg.memdesc<256x256xf32, #shared, #ttg.shared_memory, mutable> -> tensor<256x256xf32, #mma>
        %43 = arith.addf %37, %42: tensor<256x256xf32, #mma>
        scf.yield %43 : tensor<256x256xf32, #mma>
      } else {
        scf.yield %37 : tensor<256x256xf32, #mma>
      }
      %44 = arith.addi %arg9, %c1_i32 : i32
      %45 = arith.cmpi slt, %44, %c1_i32 : i32
      %46 = arith.select %45, %44, %c0_i32 : i32
      %47 = ttg.memdesc_index %21[%46] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %32, %47 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %48 = ttg.memdesc_index %22[%46] : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %34, %48 : tensor<64x256xf16, #blocked> -> !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %39, %31, %33, %46, %47, %48: tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x256xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x256xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
// CHECK-LABEL: pingpong_medium_else_reject
// CHECK-COUNT-2: local_load
// CHECK-NOT: setprio
// CHECK-NOT: barrier

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium_else_reject(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x1x!tt.ptr<f32>, #mma>
    %26 = tt.broadcast %25 : tensor<256x1x!tt.ptr<f32>, #mma> -> tensor<256x128x!tt.ptr<f32>, #mma>
    %27 = tt.load %26: tensor<256x128x!tt.ptr<f32>, #mma>
    %28:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %29 = arith.cmpi eq, %arg5, %c0_i32: i32
      %30 = scf.if %29 -> i32 {
        scf.yield %c0_i32 : i32
      } else {
        scf.yield %arg5 : i32
      }
      %31 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %32 = tt.load %31 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %33 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %34 = tt.load %33 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %35 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %36 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %37 = tt.dot %35, %36, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x128xf32, #mma>
      %38 = arith.cmpi eq, %30, %c63_i32: i32
      %39 = scf.if %38 -> tensor<256x128xf32, #mma> {
        scf.yield %37 : tensor<256x128xf32, #mma>
      } else {
        %40 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x128xf32, #shared, #ttg.shared_memory, mutable>
        %41 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x256x128xf32, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable>
        ttg.local_store %27, %41 : tensor<256x128xf32, #mma> -> !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable>
        %42 = ttg.local_load %41 : !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<256x128xf32, #mma>
        %43 = arith.addf %37, %42: tensor<256x128xf32, #mma>
        scf.yield %43 : tensor<256x128xf32, #mma>
      }
      %44 = arith.addi %arg9, %c1_i32 : i32
      %45 = arith.cmpi slt, %44, %c1_i32 : i32
      %46 = arith.select %45, %44, %c0_i32 : i32
      %47 = ttg.memdesc_index %21[%46] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %32, %47 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %48 = ttg.memdesc_index %22[%46] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %34, %48 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %39, %31, %33, %46, %47, %48: tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
// CHECK-LABEL: pingpong_medium_if_else_reject
// CHECK-COUNT-2: local_load
// CHECK-NOT: setprio
// CHECK-NOT: barrier

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pingpong_medium_if_else_reject(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %1 = tt.get_program_id x : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = arith.addi %2, %3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %6 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked1>
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
    %19 = tt.splat %arg4 : i32 -> tensor<64x128xi32, #blocked>
    %20 = tt.addptr %18, %19 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %23 = ttg.memdesc_index %21[%c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
    %24 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x1x!tt.ptr<f32>, #mma>
    %26 = tt.broadcast %25 : tensor<256x1x!tt.ptr<f32>, #mma> -> tensor<256x128x!tt.ptr<f32>, #mma>
    %27 = tt.load %26: tensor<256x128x!tt.ptr<f32>, #mma>
    %28:6 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %13, %arg8 = %20, %arg9 = %c0_i32, %arg10 = %23, %arg11 = %24) -> (tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>)  : i32 {
      %29 = arith.cmpi eq, %arg5, %c0_i32: i32
      %30 = scf.if %29 -> i32 {
        scf.yield %c0_i32 : i32
      } else {
        scf.yield %arg5 : i32
      }
      %31 = tt.addptr %arg7, %cst_1 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %32 = tt.load %31 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %33 = tt.addptr %arg8, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %34 = tt.load %33 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %35 = ttg.local_load %arg10 : !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %36 = ttg.local_load %arg11 : !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %37 = tt.dot %35, %36, %arg6 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x128xf32, #mma>
      %38 = arith.cmpi eq, %30, %c63_i32: i32
      %39 = scf.if %38 -> tensor<256x128xf32, #mma> {
        %40 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x128xf32, #shared, #ttg.shared_memory, mutable>
        %41 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x256x128xf32, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable>
        ttg.local_store %27, %41 : tensor<256x128xf32, #mma> -> !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable>
        %42 = ttg.local_load %41 : !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<256x128xf32, #mma>
        %43 = arith.subf %37, %42: tensor<256x128xf32, #mma>
        scf.yield %43 : tensor<256x128xf32, #mma>
      } else {
        %44 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x128xf32, #shared, #ttg.shared_memory, mutable>
        %45 = ttg.memdesc_index %44[%c0_i32] : !ttg.memdesc<1x256x128xf32, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable>
        ttg.local_store %27, %45 : tensor<256x128xf32, #mma> -> !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable>
        %46 = ttg.local_load %45 : !ttg.memdesc<256x128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<256x128xf32, #mma>
        %47 = arith.addf %37, %46: tensor<256x128xf32, #mma>
        scf.yield %47 : tensor<256x128xf32, #mma>
      }
      %48 = arith.addi %arg9, %c1_i32 : i32
      %49 = arith.cmpi slt, %48, %c1_i32 : i32
      %50 = arith.select %49, %48, %c0_i32 : i32
      %51 = ttg.memdesc_index %21[%50] : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      ttg.local_store %32, %51 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>
      %52 = ttg.memdesc_index %22[%50] : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      ttg.local_store %34, %52 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
      scf.yield %39, %31, %33, %50, %51, %52: tensor<256x128xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf16, #shared1, #ttg.shared_memory, mutable>
    }
    ttg.local_dealloc %21 : !ttg.memdesc<1x256x64xf16, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x64x128xf16, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
// CHECK-LABEL: async_ns3_gemm
// CHECK-NOT: rocdl
// CHECK-NS3-LABEL: async_ns3_gemm
// CHECK-NS3: amdgpu.cond_barrier
// CHECK-NS3: %[[LL0:.+]] = ttg.local_load
// CHECK-NS3: %[[LL1:.+]] = ttg.local_load
// CHECK-NS3: ttg.async_wait
// CHECK-NS3: tt.dot %[[LL0]], %[[LL1]]
// CHECK-NS3: amdgpu.cond_barrier

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_ns3_gemm(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: tensor<256x32x!tt.ptr<bf16>, #blocked>, %arg11: tensor<32x256x!tt.ptr<bf16>, #blocked1>, %arg12: !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, %arg13: !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, %arg14: !ttg.async.token, %arg15: !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, %arg17: !ttg.async.token, %arg18: !ttg.async.token, %arg19: !ttg.async.token, %arg20: tensor<256x32xi32, #blocked>, %arg21: tensor<32x256xi32, #blocked1>, %arg22: !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable>, %arg23: !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable>, %arg24: tensor<256x256x!tt.ptr<bf16>, #mma>, %arg25: tensor<256x256xi1, #mma>) {
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:12 = scf.for %arg26 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg27 = %cst, %arg28 = %arg10, %arg29 = %arg11, %arg30 = %c1_i32, %arg31 = %arg12, %arg32 = %arg13, %arg33 = %arg14, %arg34 = %arg15, %arg35 = %arg16, %arg36 = %arg17, %arg37 = %arg18, %arg38 = %arg19) -> (tensor<256x256xf32, #mma>, tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<32x256x!tt.ptr<bf16>, #blocked1>, i32, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %4 = tt.addptr %arg28, %arg20 : tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<256x32xi32, #blocked>
      %5 = tt.addptr %arg29, %arg21 : tensor<32x256x!tt.ptr<bf16>, #blocked1>, tensor<32x256xi32, #blocked1>
      %6 = arith.addi %arg30, %c1_i32 : i32
      %7 = arith.cmpi slt, %6, %c3_i32 : i32
      %8 = arith.select %7, %6, %c0_i32 : i32
      %9 = ttg.memdesc_index %arg22[%8] : !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>
      %10 = ttg.async_copy_global_to_local %4, %9 : tensor<256x32x!tt.ptr<bf16>, #blocked> -> <256x32xbf16, #shared, #smem, mutable>
      %11 = ttg.async_commit_group tokens %10
      %12 = ttg.local_load %arg31 token %arg33 : !ttg.memdesc<256x32xbf16, #shared, #smem, mutable> -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %13 = ttg.memdesc_index %arg23[%8] : !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>
      %14 = ttg.async_copy_global_to_local %5, %13 : tensor<32x256x!tt.ptr<bf16>, #blocked1> -> <32x256xbf16, #shared1, #smem, mutable>
      %15 = ttg.async_commit_group tokens %14
      %16 = ttg.local_load %arg34 token %arg36 : !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %17 = tt.dot %12, %16, %arg27 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
      %18 = ttg.async_wait %arg37 {num = 0 : i32}
      %19 = ttg.async_wait %arg38 {num = 0 : i32}
      scf.yield %17, %4, %5, %8, %arg32, %9, %18, %arg35, %13, %19, %11, %15 : tensor<256x256xf32, #mma>, tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<32x256x!tt.ptr<bf16>, #blocked1>, i32, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %1 = ttg.async_wait %0#10 {num = 0 : i32}
    %2 = ttg.async_wait %0#11 {num = 0 : i32}
    ttg.local_dealloc %arg22 : !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable>
    ttg.local_dealloc %arg23 : !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable>
    %3 = arith.truncf %0#0 : tensor<256x256xf32, #mma> to tensor<256x256xbf16, #mma>
    tt.store %arg24, %3, %arg25 : tensor<256x256x!tt.ptr<bf16>, #mma>
    tt.return
  }
}


// -----
// CHECK-LABEL: gemm_mxfp4
// CHECK: amdgpu.cond_barrier
// CHECK: %[[WAIT:.+]] = ttg.async_wait
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_copy_global_to_local
// CHECK: rocdl.sched.barrier 0
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier 0
// CHECK: %[[LL0:.+]] = ttg.local_load
// CHECK-SAME: %[[WAIT]]
// CHECK: %[[LL1:.+]] = ttg.local_load
// CHECK-SAME: %[[WAIT]]
// CHECK: %[[LL2:.+]] = ttg.local_load
// CHECK-SAME: %[[WAIT]]
// CHECK: %[[LL3:.+]] = ttg.local_load
// CHECK-SAME: %[[WAIT]]
// CHECK: tt.dot_scaled %[[LL2]] scale %[[LL0]], %[[LL3]] scale %[[LL1]]
// CHECK: amdgpu.cond_barrier

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @gemm_mxfp4(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: tensor<256x8x!tt.ptr<i8>, #blocked>, %arg15: tensor<256x8x!tt.ptr<i8>, #blocked>, %arg16: tensor<256x128x!tt.ptr<i8>, #blocked1>, %arg17: tensor<128x256x!tt.ptr<i8>, #blocked2>, %arg18: !ttg.async.token, %arg19: !ttg.async.token, %arg20: !ttg.async.token, %arg21: !ttg.async.token, %arg22: !ttg.memdesc<256x8xi8, #shared, #smem, mutable>, %arg23: !ttg.memdesc<256x8xi8, #shared, #smem, mutable>, %arg24: !ttg.memdesc<256x128xi8, #shared1, #smem, mutable>, %arg25: !ttg.memdesc<128x256xi8, #shared2, #smem, mutable>, %arg26: tensor<256x8xi32, #blocked>, %arg27: tensor<256x8xi32, #blocked>, %arg28: tensor<256x256x!tt.ptr<bf16>, #mma>, %arg29: tensor<256x256xi1, #mma>) {
    %c63_i32 = arith.constant 63 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<128> : tensor<256x128xi32, #blocked1>
    %cst_0 = arith.constant dense<128> : tensor<128x256xi32, #blocked2>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x256x128xi8, #shared1, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2x128x256xi8, #shared2, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<2x256x8xi8, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<2x256x8xi8, #shared, #smem, mutable>
    %4:14 = scf.for %arg30 = %c0_i32 to %c63_i32 step %c1_i32 iter_args(%arg31 = %cst_1, %arg32 = %arg14, %arg33 = %arg15, %arg34 = %arg16, %arg35 = %arg17, %arg36 = %c0_i32, %arg37 = %arg18, %arg38 = %arg19, %arg39 = %arg20, %arg40 = %arg21, %arg41 = %arg22, %arg42 = %arg23, %arg43 = %arg24, %arg44 = %arg25) -> (tensor<256x256xf32, #mma>, tensor<256x8x!tt.ptr<i8>, #blocked>, tensor<256x8x!tt.ptr<i8>, #blocked>, tensor<256x128x!tt.ptr<i8>, #blocked1>, tensor<128x256x!tt.ptr<i8>, #blocked2>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.memdesc<256x8xi8, #shared, #smem, mutable>, !ttg.memdesc<256x8xi8, #shared, #smem, mutable>, !ttg.memdesc<256x128xi8, #shared1, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared2, #smem, mutable>)  : i32 {
      %7 = ttg.async_wait %arg37, %arg38, %arg39, %arg40 {num = 0 : i32}
      %8 = tt.addptr %arg34, %cst : tensor<256x128x!tt.ptr<i8>, #blocked1>, tensor<256x128xi32, #blocked1>
      %9 = tt.addptr %arg35, %cst_0 : tensor<128x256x!tt.ptr<i8>, #blocked2>, tensor<128x256xi32, #blocked2>
      %10 = tt.addptr %arg32, %arg26 : tensor<256x8x!tt.ptr<i8>, #blocked>, tensor<256x8xi32, #blocked>
      %11 = tt.addptr %arg33, %arg27 : tensor<256x8x!tt.ptr<i8>, #blocked>, tensor<256x8xi32, #blocked>
      %12 = arith.addi %arg36, %c1_i32 : i32
      %13 = arith.cmpi slt, %12, %c2_i32 : i32
      %14 = arith.select %13, %12, %c0_i32 : i32
      %15 = ttg.memdesc_index %2[%14] : !ttg.memdesc<2x256x8xi8, #shared, #smem, mutable> -> !ttg.memdesc<256x8xi8, #shared, #smem, mutable>
      %16 = ttg.async_copy_global_to_local %10, %15 : tensor<256x8x!tt.ptr<i8>, #blocked> -> <256x8xi8, #shared, #smem, mutable>
      %17 = ttg.async_commit_group tokens %16
      %18 = ttg.local_load %arg41 token %7 : !ttg.memdesc<256x8xi8, #shared, #smem, mutable> -> tensor<256x8xi8, #linear>
      %19 = ttg.memdesc_index %3[%14] : !ttg.memdesc<2x256x8xi8, #shared, #smem, mutable> -> !ttg.memdesc<256x8xi8, #shared, #smem, mutable>
      %20 = ttg.async_copy_global_to_local %11, %19 : tensor<256x8x!tt.ptr<i8>, #blocked> -> <256x8xi8, #shared, #smem, mutable>
      %21 = ttg.async_commit_group tokens %20
      %22 = ttg.local_load %arg42 token %7 : !ttg.memdesc<256x8xi8, #shared, #smem, mutable> -> tensor<256x8xi8, #linear1>
      %23 = ttg.memdesc_index %0[%14] : !ttg.memdesc<2x256x128xi8, #shared1, #smem, mutable> -> !ttg.memdesc<256x128xi8, #shared1, #smem, mutable>
      %24 = ttg.async_copy_global_to_local %8, %23 : tensor<256x128x!tt.ptr<i8>, #blocked1> -> <256x128xi8, #shared1, #smem, mutable>
      %25 = ttg.async_commit_group tokens %24
      %26 = ttg.local_load %arg43 token %7 : !ttg.memdesc<256x128xi8, #shared1, #smem, mutable> -> tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
      %27 = ttg.memdesc_index %1[%14] : !ttg.memdesc<2x128x256xi8, #shared2, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared2, #smem, mutable>
      %28 = ttg.async_copy_global_to_local %9, %27 : tensor<128x256x!tt.ptr<i8>, #blocked2> -> <128x256xi8, #shared2, #smem, mutable>
      %29 = ttg.async_commit_group tokens %28
      %30 = ttg.local_load %arg44 token %7 : !ttg.memdesc<128x256xi8, #shared2, #smem, mutable> -> tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %31 = tt.dot_scaled %26 scale %18, %30 scale %22, %arg31 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
      scf.yield %31, %10, %11, %8, %9, %14, %17, %21, %25, %29, %15, %19, %23, %27 : tensor<256x256xf32, #mma>, tensor<256x8x!tt.ptr<i8>, #blocked>, tensor<256x8x!tt.ptr<i8>, #blocked>, tensor<256x128x!tt.ptr<i8>, #blocked1>, tensor<128x256x!tt.ptr<i8>, #blocked2>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.memdesc<256x8xi8, #shared, #smem, mutable>, !ttg.memdesc<256x8xi8, #shared, #smem, mutable>, !ttg.memdesc<256x128xi8, #shared1, #smem, mutable>, !ttg.memdesc<128x256xi8, #shared2, #smem, mutable>
    }
    %5 = ttg.async_wait %4#6, %4#7, %4#8, %4#9 {num = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<2x256x128xi8, #shared1, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<2x128x256xi8, #shared2, #smem, mutable>
    ttg.local_dealloc %2 : !ttg.memdesc<2x256x8xi8, #shared, #smem, mutable>
    ttg.local_dealloc %3 : !ttg.memdesc<2x256x8xi8, #shared, #smem, mutable>
    %6 = arith.truncf %4#0 : tensor<256x256xf32, #mma> to tensor<256x256xbf16, #mma>
    tt.store %arg28, %6, %arg29 : tensor<256x256x!tt.ptr<bf16>, #mma>
    tt.return
  }
}
