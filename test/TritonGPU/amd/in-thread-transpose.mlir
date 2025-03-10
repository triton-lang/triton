// RUN: triton-opt %s -split-input-file -tritonamdgpu-in-thread-transpose | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-DAG: [[$threadrake_layout1:#.*]] = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [32, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
// CHECK-DAG: [[$threadrake_layout2:#.*]] = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-DAG: [[$linear1:#.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [1, 0], [2, 0], [4, 0{{]]}}, lane = {{\[\[}}8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 4{{]]}}, warp = {{\[\[}}0, 8], [0, 16], [0, 0{{]]}}, block = []}>
// CHECK-DAG: [[$linear2:#.*]] = #ttg.linear<{register = {{\[\[}}1, 0], [2, 0], [0, 1], [0, 2], [0, 4{{]]}}, lane = {{\[\[}}0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [8, 0{{]]}}, warp = {{\[\[}}16, 0], [0, 0], [0, 0{{]]}}, block = []}>
// CHECK-DAG: [[$shared1:#.*]] = #ttg.amd_rotating_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK-DAG: [[$shared2:#.*]] = #ttg.amd_rotating_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>

// CHECK-LABEL: threadRake_simple

// CHECK-DAG: [[load_val1:%.*]] = tt.load {{.*}} : tensor<256x32x!tt.ptr<f16>, [[$threadrake_layout1]]>
// CHECK-DAG: [[load_val2:%.*]] = tt.load {{.*}} : tensor<32x128x!tt.ptr<f16>, [[$threadrake_layout2]]>

// CHECK-DAG: [[alloc1:%.*]] = ttg.local_alloc {{.*}} : (tensor<256x32xf16, [[$linear1]]>) -> !ttg.memdesc<256x32xf16, [[$shared1]], #smem>
// CHECK-DAG: ttg.local_load [[alloc1]] : !ttg.memdesc<256x32xf16, [[$shared1]], #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

// CHECK-DAG: [[alloc2:%.*]] = ttg.local_alloc {{.*}} : (tensor<32x128xf16, [[$linear2]]>) -> !ttg.memdesc<32x128xf16, [[$shared2]], #smem>
// CHECK-DAG: ttg.local_load [[alloc2]] : !ttg.memdesc<32x128xf16, [[$shared2]], #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  tt.func public @threadRake_simple(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>

    %4 = ttg.local_alloc %2 : (tensor<256x32xf16, #blocked>) -> !ttg.memdesc<256x32xf16, #shared, #smem>
    %5 = ttg.local_load %4 : !ttg.memdesc<256x32xf16, #shared, #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

    %6 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %7 = ttg.local_load %6 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

    %8 = tt.dot %5, %7, %cst_0 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-NOT: #ttg.amd_rotating_shared
// CHECK-NOT: #ttg.linear
// CHECK-DAG: [[$blocked1:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-DAG: [[$blocked2:#.*]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
// CHECK-NOT: #ttg.amd_rotating_shared
// CHECK-NOT: #ttg.linear
// CHECK-LABEL: threadRake_k_fast_neg
// CHECK-DAG: tt.load {{.*}} : tensor<256x32x!tt.ptr<f16>, [[$blocked1]]>
// CHECK-DAG: tt.load {{.*}} : tensor<32x128x!tt.ptr<f16>, [[$blocked2]]>
  tt.func public @threadRake_k_fast_neg(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>

    %4 = ttg.local_alloc %2 : (tensor<256x32xf16, #blocked>) -> !ttg.memdesc<256x32xf16, #shared, #smem>
    %5 = ttg.local_load %4 : !ttg.memdesc<256x32xf16, #shared, #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

    %6 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %7 = ttg.local_load %6 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

    %8 = tt.dot %5, %7, %cst_0 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-NOT: #ttg.amd_rotating_shared
// CHECK-NOT: #ttg.linear
// CHECK-DAG: [[$blocked1:#.*]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
// CHECK-DAG: [[$blocked2:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-NOT: #ttg.amd_rotating_shared
// CHECK-NOT: #ttg.linear
// CHECK-LABEL: threadRake_small_k_neg
// CHECK-DAG: tt.load {{.*}} : tensor<256x32x!tt.ptr<f16>, [[$blocked1]]>
// CHECK-DAG: tt.load {{.*}} : tensor<32x128x!tt.ptr<f16>, [[$blocked2]]>
  tt.func public @threadRake_small_k_neg(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>

    %4 = ttg.local_alloc %2 : (tensor<256x32xf16, #blocked>) -> !ttg.memdesc<256x32xf16, #shared, #smem>
    %5 = ttg.local_load %4 : !ttg.memdesc<256x32xf16, #shared, #smem> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

    %6 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %7 = ttg.local_load %6 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

    %8 = tt.dot %5, %7, %cst_0 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-DAG: [[$threadrake_layout:#.*]] = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: [[$linear:#.*]] = #ttg.linear<{register = {{\[\[}}1, 0], [2, 0], [0, 1], [0, 2], [0, 4], [32, 0{{]]}}, lane = {{\[\[}}0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0{{]]}}, warp = [], block = []}>
// CHECK-DAG: [[$shared:#.*]] = #ttg.amd_rotating_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>

// CHECK-LABEL: threadRake_with_cfg

// CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, [[$shared]], #smem, mutable>
// CHECK: [[load_val_preloop:%.*]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>, [[$threadrake_layout]]>
// CHECK: ttg.local_store {{.*}} : tensor<64x64xf16, [[$linear]]> -> !ttg.memdesc<64x64xf16, [[$shared]], #smem, mutable>
// CHECK: scf.for
// CHECK: [[load_val_loop:%.*]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>, [[$threadrake_layout]]>
// CHECK: ttg.local_store {{.*}} : tensor<64x64xf16, [[$linear]]> -> !ttg.memdesc<64x64xf16, [[$shared]], #smem, mutable>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @threadRake_with_cfg(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x64xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.divsi %0, %4 : i32
    %6 = arith.subi %2, %5 : i32
    %7 = arith.minsi %6, %c1_i32 : i32
    %8 = arith.remsi %0, %4 : i32
    %9 = arith.remsi %8, %7 : i32
    %10 = arith.addi %5, %9 : i32
    %11 = arith.divsi %8, %7 : i32
    %12 = arith.muli %10, %c64_i32 : i32
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.splat %12 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = arith.addi %15, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = arith.muli %11, %c64_i32 : i32
    %18 = tt.splat %17 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = arith.addi %18, %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %21 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #blocked>
    %22 = arith.muli %20, %21 : tensor<64x1xi32, #blocked>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %25 = tt.broadcast %22 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %26 = tt.broadcast %24 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %27 = arith.addi %25, %26 : tensor<64x64xi32, #blocked>
    %28 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %29 = tt.addptr %28, %27 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %30 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %31 = tt.expand_dims %30 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %32 = tt.splat %arg7 : i32 -> tensor<64x1xi32, #blocked>
    %33 = arith.muli %31, %32 : tensor<64x1xi32, #blocked>
    %34 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %35 = tt.broadcast %33 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %36 = tt.broadcast %34 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %37 = arith.addi %35, %36 : tensor<64x64xi32, #blocked>
    %38 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %39 = tt.addptr %38, %37 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %40 = arith.addi %arg5, %c63_i32 : i32
    %41 = arith.divsi %40, %c64_i32 : i32
    %42 = arith.muli %arg7, %c64_i32 : i32
    %43 = tt.splat %42 : i32 -> tensor<64x64xi32, #blocked>
    %44 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %45 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable>
    %46 = arith.cmpi sgt, %41, %c0_i32 : i32
    %47 = tt.splat %46 : i1 -> tensor<64x64xi1, #blocked>
    %48 = tt.load %29, %47 {OpIdx = #amdgpu.OpIdx<0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
    %49 = tt.splat %46 : i1 -> tensor<64x64xi1, #blocked>
    %50 = tt.load %39, %49 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x64x!tt.ptr<f16>, #blocked>
    %51 = ttg.memdesc_subview %44[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttg.local_store %48, %51 {OpIdx = #amdgpu.OpIdx<0>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %52 = ttg.memdesc_subview %45[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    ttg.local_store %50, %52 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    %53 = arith.subi %41, %c1_i32 : i32
    %54:6 = scf.for %arg9 = %c0_i32 to %53 step %c1_i32 iter_args(%arg10 = %cst_0, %arg11 = %29, %arg12 = %39, %arg13 = %c0_i32, %arg14 = %51, %arg15 = %52) -> (tensor<64x64xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>)  : i32 {
      %76 = tt.addptr %arg11, %cst : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %77 = tt.addptr %arg12, %43 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %78 = tt.load %76 {OpIdx = #amdgpu.OpIdx<0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %79 = ttg.local_load %arg14 : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %80 = tt.load %77 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %81 = ttg.local_load %arg15 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %82 = tt.dot %79, %81, %arg10, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      %83 = arith.addi %arg13, %c1_i32 : i32
      %84 = arith.cmpi slt, %83, %c1_i32 : i32
      %85 = arith.select %84, %83, %c0_i32 : i32
      %86 = ttg.memdesc_subview %44[%85, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttg.local_store %78, %86 {OpIdx = #amdgpu.OpIdx<0>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %87 = ttg.memdesc_subview %45[%85, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
      ttg.local_store %80, %87 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
      scf.yield %82, %76, %77, %85, %86, %87 : tensor<64x64xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
    }
    %55 = arith.cmpi sge, %41, %c1_i32 : i32
    %56 = ttg.local_load %54#4 : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %57 = ttg.local_load %54#5 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %58 = scf.if %55 -> (tensor<64x64xf32, #mma>) {
      %76 = tt.dot %56, %57, %54#0, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      scf.yield %76 : tensor<64x64xf32, #mma>
    } else {
      scf.yield %54#0 : tensor<64x64xf32, #mma>
    }
    %59 = arith.select %55, %58, %54#0 : tensor<64x64xf32, #mma>
    ttg.local_dealloc %44 : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %45 : !ttg.memdesc<1x64x64xf16, #shared1, #smem, mutable>
    %60 = arith.truncf %59 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
    %61 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #blocked>
    %62 = arith.muli %61, %20 : tensor<64x1xi32, #blocked>
    %63 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %64 = tt.addptr %63, %62 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %65 = tt.broadcast %64 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %66 = tt.addptr %65, %36 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %67 = tt.splat %arg3 : i32 -> tensor<64x1xi32, #blocked>
    %68 = arith.cmpi slt, %20, %67 : tensor<64x1xi32, #blocked>
    %69 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked>
    %70 = arith.cmpi slt, %34, %69 : tensor<1x64xi32, #blocked>
    %71 = tt.broadcast %68 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %72 = tt.broadcast %70 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %73 = arith.andi %71, %72 : tensor<64x64xi1, #blocked>
    %74 = ttg.convert_layout %66 : tensor<64x64x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #mma>
    %75 = ttg.convert_layout %73 : tensor<64x64xi1, #blocked> -> tensor<64x64xi1, #mma>
    tt.store %74, %60, %75 : tensor<64x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}
