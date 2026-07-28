// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s --check-prefixes=BASELINE,TREE
// RUN: triton-opt %s -split-input-file -tritongpu-optimize-layouts | FileCheck %s --check-prefixes=OPTIMIZED,TREE
// RUN: triton-opt %s -split-input-file -tritongpu-optimize-layouts -tritongpu-optimize-layouts | FileCheck %s --check-prefixes=OPTIMIZED,TREE
// RUN: triton-opt %s -split-input-file -tritongpu-optimize-layouts --mlir-print-ir-after-all -o /dev/null 2>&1 | FileCheck %s --check-prefix=PASS-PIPELINE

// PASS-PIPELINE-NOT: TritonGPURemoveLayoutConversions
// PASS-PIPELINE: IR Dump After TritonGPUOptimizeLayouts

// Reduced from the exact-order stochastic-rounding handoff. Unlike adding
// allow_reorder to the reshapes, optimizing the whole scalar-rooted expression
// must preserve every logical element position.
//
// BASELINE-LABEL: @stochastic_rounding_join_chain
// BASELINE-COUNT-8: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @stochastic_rounding_join_chain
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-NOT: allow_reorder
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 16, 2], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#linear1 = #ttg.linear<{register = [[0, 8, 0], [0, 16, 0]], lane = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 32, 0]], warp = [[0, 64, 0], [0, 128, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[16], [32]], lane = [[1], [2], [4], [8], [64]], warp = [[128], [256]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 16], [0, 32]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 64]], warp = [[0, 128], [0, 256]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 32, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 64, 0]], warp = [[0, 128, 0], [0, 256, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[1], [32], [64]], lane = [[2], [4], [8], [16], [128]], warp = [[256], [512]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1], [0, 32], [0, 64]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 128]], warp = [[0, 256], [0, 512]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 0, 1], [0, 1, 0], [0, 32, 0], [0, 64, 0]], lane = [[0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 128, 0]], warp = [[0, 256, 0], [0, 512, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @stochastic_rounding_join_chain(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> tensor<32x64xi32, #mma> {
    %a0 = tt.splat %arg0 : i32 -> tensor<128xi32, #blocked1>
    %a1 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked1>
    %x0 = tt.splat %arg2 : i32 -> tensor<128xi32, #blocked1>
    %x1 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked1>

    %0 = tt.reshape %a0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %1 = tt.reshape %a1 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %2 = tt.join %0, %1 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %3 = ttg.convert_layout %2 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %4 = tt.reshape %3 : tensor<1x128x2xi32, #blocked4> -> tensor<256xi32, #blocked1>

    %5 = tt.reshape %x0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %6 = tt.join %5, %5 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %7 = ttg.convert_layout %6 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %8 = tt.reshape %7 : tensor<1x128x2xi32, #blocked4> -> tensor<256xi32, #blocked1>

    %9 = arith.xori %4, %8 : tensor<256xi32, #blocked1>
    %10 = tt.reshape %3 : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
    %11 = tt.reshape %9 : tensor<256xi32, #blocked1> -> tensor<1x256xi32, #blocked2>
    %12 = tt.join %10, %11 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
    %13 = ttg.convert_layout %12 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
    %14 = tt.reshape %13 : tensor<1x256x2xi32, #linear1> -> tensor<512xi32, #linear2>

    %15 = tt.reshape %x1 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %16 = tt.join %15, %15 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %17 = ttg.convert_layout %16 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %18 = tt.reshape %17 : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
    %19 = tt.join %18, %18 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
    %20 = ttg.convert_layout %19 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
    %21 = tt.reshape %20 : tensor<1x256x2xi32, #linear1> -> tensor<512xi32, #linear2>

    %22 = arith.xori %14, %21 : tensor<512xi32, #linear2>
    %23 = tt.reshape %13 : tensor<1x256x2xi32, #linear1> -> tensor<1x512xi32, #linear3>
    %24 = tt.reshape %22 : tensor<512xi32, #linear2> -> tensor<1x512xi32, #linear3>
    %25 = tt.join %23, %24 : tensor<1x512xi32, #linear3> -> tensor<1x512x2xi32, #linear4>
    %26 = tt.reshape %25 : tensor<1x512x2xi32, #linear4> -> tensor<1024xi32, #linear5>

    %27 = tt.reshape %a0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %28 = tt.join %27, %27 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %29 = ttg.convert_layout %28 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %30 = tt.reshape %29 : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
    %31 = tt.join %30, %30 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
    %32 = ttg.convert_layout %31 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
    %33 = tt.reshape %32 : tensor<1x256x2xi32, #linear1> -> tensor<1x512xi32, #linear3>
    %34 = tt.join %33, %33 : tensor<1x512xi32, #linear3> -> tensor<1x512x2xi32, #linear4>
    %35 = tt.reshape %34 : tensor<1x512x2xi32, #linear4> -> tensor<1024xi32, #linear5>

    %36 = arith.xori %26, %35 : tensor<1024xi32, #linear5>
    %37 = tt.reshape %25 : tensor<1x512x2xi32, #linear4> -> tensor<1x1024xi32, #linear6>
    %38 = tt.reshape %36 : tensor<1024xi32, #linear5> -> tensor<1x1024xi32, #linear6>
    %39 = tt.join %37, %38 : tensor<1x1024xi32, #linear6> -> tensor<1x1024x2xi32, #linear7>
    %40 = tt.reshape %39 : tensor<1x1024x2xi32, #linear7> -> tensor<32x64xi32, #blocked>
    %41 = ttg.convert_layout %40 : tensor<32x64xi32, #blocked> -> tensor<32x64xi32, #mma>
    tt.return %41 : tensor<32x64xi32, #mma>
  }

  // Production stochastic rounding starts from a logical range and shared
  // random-number arithmetic in a nested region. The range and scalar splats
  // dominate the region from the enclosing function block.
  //
  // BASELINE-LABEL: @stochastic_rounding_indexed_join_chain
  // BASELINE-COUNT-8: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @stochastic_rounding_indexed_join_chain
  // OPTIMIZED-NOT: allow_reorder
  // OPTIMIZED: tt.make_range
  // OPTIMIZED: arith.shrui
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func @stochastic_rounding_indexed_join_chain(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> tensor<32x64xi32, #mma> {
    %zero = arith.constant 0 : i32
    %condition = arith.cmpi ne, %arg0, %zero : i32
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %seed_a0 = tt.splat %arg0 : i32 -> tensor<128xi32, #blocked1>
    %seed_a1 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked1>
    %seed_x0 = tt.splat %arg2 : i32 -> tensor<128xi32, #blocked1>
    %seed_x1 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked1>

    %result = scf.if %condition -> (tensor<32x64xi32, #mma>) {
      %a0 = arith.xori %seed_a0, %range : tensor<128xi32, #blocked1>
      %a1 = arith.addi %seed_a1, %range : tensor<128xi32, #blocked1>
      %x0 = arith.xori %seed_x0, %range : tensor<128xi32, #blocked1>
      %x1 = arith.xori %seed_x1, %range : tensor<128xi32, #blocked1>

      %0 = tt.reshape %a0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
      %1 = tt.reshape %a1 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
      %2 = tt.join %0, %1 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
      %3 = ttg.convert_layout %2 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
      %4 = tt.reshape %3 : tensor<1x128x2xi32, #blocked4> -> tensor<256xi32, #blocked1>

      %5 = tt.reshape %x0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
      %6 = tt.join %5, %5 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
      %7 = ttg.convert_layout %6 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
      %8 = tt.reshape %7 : tensor<1x128x2xi32, #blocked4> -> tensor<256xi32, #blocked1>

      %9 = arith.xori %4, %8 : tensor<256xi32, #blocked1>
      %10 = tt.reshape %3 : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
      %11 = tt.reshape %9 : tensor<256xi32, #blocked1> -> tensor<1x256xi32, #blocked2>
      %12 = tt.join %10, %11 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
      %13 = ttg.convert_layout %12 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
      %14 = tt.reshape %13 : tensor<1x256x2xi32, #linear1> -> tensor<512xi32, #linear2>

      %15 = tt.reshape %x1 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
      %16 = tt.join %15, %15 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
      %17 = ttg.convert_layout %16 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
      %18 = tt.reshape %17 : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
      %19 = tt.join %18, %18 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
      %20 = ttg.convert_layout %19 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
      %21 = tt.reshape %20 : tensor<1x256x2xi32, #linear1> -> tensor<512xi32, #linear2>

      %22 = arith.xori %14, %21 : tensor<512xi32, #linear2>
      %23 = tt.reshape %13 : tensor<1x256x2xi32, #linear1> -> tensor<1x512xi32, #linear3>
      %24 = tt.reshape %22 : tensor<512xi32, #linear2> -> tensor<1x512xi32, #linear3>
      %25 = tt.join %23, %24 : tensor<1x512xi32, #linear3> -> tensor<1x512x2xi32, #linear4>
      %26 = tt.reshape %25 : tensor<1x512x2xi32, #linear4> -> tensor<1024xi32, #linear5>

      %27 = tt.reshape %a0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
      %28 = tt.join %27, %27 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
      %29 = ttg.convert_layout %28 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
      %30 = tt.reshape %29 : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
      %31 = tt.join %30, %30 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
      %32 = ttg.convert_layout %31 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
      %33 = tt.reshape %32 : tensor<1x256x2xi32, #linear1> -> tensor<1x512xi32, #linear3>
      %34 = tt.join %33, %33 : tensor<1x512xi32, #linear3> -> tensor<1x512x2xi32, #linear4>
      %35 = tt.reshape %34 : tensor<1x512x2xi32, #linear4> -> tensor<1024xi32, #linear5>

      %36 = arith.xori %26, %35 : tensor<1024xi32, #linear5>
      %37 = tt.reshape %25 : tensor<1x512x2xi32, #linear4> -> tensor<1x1024xi32, #linear6>
      %38 = tt.reshape %36 : tensor<1024xi32, #linear5> -> tensor<1x1024xi32, #linear6>
      %39 = tt.join %37, %38 : tensor<1x1024xi32, #linear6> -> tensor<1x1024x2xi32, #linear7>
      %40 = tt.reshape %39 : tensor<1x1024x2xi32, #linear7> -> tensor<32x64xi32, #blocked>
      %41 = ttg.convert_layout %40 : tensor<32x64xi32, #blocked> -> tensor<32x64xi32, #mma>
      scf.yield %41 : tensor<32x64xi32, #mma>
    } else {
      %fallback = tt.splat %arg0 : i32 -> tensor<32x64xi32, #mma>
      scf.yield %fallback : tensor<32x64xi32, #mma>
    }

    tt.return %result : tensor<32x64xi32, #mma>
  }

  // A real frontend kernel can end at a store with the correct value layout,
  // leaving no final conversion to seed backward propagation. Start from the
  // store boundary so all interior joins are optimized together.
  //
  // BASELINE-LABEL: @stochastic_rounding_indexed_store_chain
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @stochastic_rounding_indexed_store_chain
  // OPTIMIZED: tt.make_range
  // OPTIMIZED: arith.shrui
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.store
  // OPTIMIZED: tt.return
  tt.func @stochastic_rounding_indexed_store_chain(%out: !tt.ptr<i32>, %arg0: i32, %arg1: i32, %arg2: i32) {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %seed0 = tt.splat %arg0 : i32 -> tensor<128xi32, #blocked1>
    %seed1 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked1>
    %seed2 = tt.splat %arg2 : i32 -> tensor<128xi32, #blocked1>
    %a0 = arith.xori %seed0, %range : tensor<128xi32, #blocked1>
    %a1 = arith.addi %seed1, %range : tensor<128xi32, #blocked1>
    %x0 = arith.xori %seed2, %range : tensor<128xi32, #blocked1>

    %lhs0 = tt.reshape %a0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %rhs0 = tt.reshape %a1 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %join0 = tt.join %lhs0, %rhs0 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %convert0 = ttg.convert_layout %join0 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %values0 = tt.reshape %convert0 : tensor<1x128x2xi32, #blocked4> -> tensor<256xi32, #blocked1>

    %lhs1 = tt.reshape %x0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %join1 = tt.join %lhs1, %lhs1 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %convert1 = ttg.convert_layout %join1 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %values1 = tt.reshape %convert1 : tensor<1x128x2xi32, #blocked4> -> tensor<256xi32, #blocked1>

    %mixed = arith.xori %values0, %values1 : tensor<256xi32, #blocked1>
    %lhs2 = tt.reshape %convert0 : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
    %rhs2 = tt.reshape %mixed : tensor<256xi32, #blocked1> -> tensor<1x256xi32, #blocked2>
    %join2 = tt.join %lhs2, %rhs2 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
    %convert2 = ttg.convert_layout %join2 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
    %result = tt.reshape %convert2 : tensor<1x256x2xi32, #linear1> -> tensor<512xi32, #linear2>

    %offsets = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #linear2>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>, #linear2>
    %address = tt.addptr %base, %offsets : tensor<512x!tt.ptr<i32>, #linear2>, tensor<512xi32, #linear2>
    tt.store %address, %result : tensor<512x!tt.ptr<i32>, #linear2>
    tt.return
  }
}

// -----

// The result encoding is a hard boundary. The remaining conversion must not
// be removed merely to improve a conversion count.
//
// TREE-LABEL: @tree_join_layout_pressure
// TREE-COUNT-1: ttg.convert_layout
// TREE-NOT: ttg.convert_layout
// TREE: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 2, 2], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [8, 2, 2, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [4, 2, 2, 2], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 2], threadsPerWarp = [4, 2, 2, 2, 1], warpsPerCTA = [4, 1, 1, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2, 2], threadsPerWarp = [8, 2, 2, 1, 1], warpsPerCTA = [4, 1, 1, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2, 2, 2], threadsPerWarp = [8, 2, 2, 1, 1, 1], warpsPerCTA = [4, 1, 1, 1, 1, 1], order = [5, 4, 3, 2, 1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tree_join_layout_pressure(%arg0: tensor<128x2xi16, #blocked>) -> tensor<128x32xi16, #blocked1> {
    %0 = tt.join %arg0, %arg0 : tensor<128x2xi16, #blocked> -> tensor<128x2x2xi16, #blocked2>
    %1 = ttg.convert_layout %0 : tensor<128x2x2xi16, #blocked2> -> tensor<128x2x2xi16, #blocked3>
    %2 = tt.join %arg0, %arg0 : tensor<128x2xi16, #blocked> -> tensor<128x2x2xi16, #blocked2>
    %3 = ttg.convert_layout %2 : tensor<128x2x2xi16, #blocked2> -> tensor<128x2x2xi16, #blocked3>
    %4 = tt.join %1, %3 : tensor<128x2x2xi16, #blocked3> -> tensor<128x2x2x2xi16, #blocked4>
    %5 = ttg.convert_layout %4 : tensor<128x2x2x2xi16, #blocked4> -> tensor<128x2x2x2xi16, #blocked5>
    %6 = tt.join %arg0, %arg0 : tensor<128x2xi16, #blocked> -> tensor<128x2x2xi16, #blocked2>
    %7 = ttg.convert_layout %6 : tensor<128x2x2xi16, #blocked2> -> tensor<128x2x2xi16, #blocked3>
    %8 = tt.join %arg0, %arg0 : tensor<128x2xi16, #blocked> -> tensor<128x2x2xi16, #blocked2>
    %9 = ttg.convert_layout %8 : tensor<128x2x2xi16, #blocked2> -> tensor<128x2x2xi16, #blocked3>
    %10 = tt.join %7, %9 : tensor<128x2x2xi16, #blocked3> -> tensor<128x2x2x2xi16, #blocked4>
    %11 = ttg.convert_layout %10 : tensor<128x2x2x2xi16, #blocked4> -> tensor<128x2x2x2xi16, #blocked5>
    %12 = tt.join %5, %11 : tensor<128x2x2x2xi16, #blocked5> -> tensor<128x2x2x2x2xi16, #blocked6>
    %13 = ttg.convert_layout %12 : tensor<128x2x2x2x2xi16, #blocked6> -> tensor<128x2x2x2x2xi16, #blocked7>
    %14 = tt.join %arg0, %arg0 : tensor<128x2xi16, #blocked> -> tensor<128x2x2xi16, #blocked2>
    %15 = ttg.convert_layout %14 : tensor<128x2x2xi16, #blocked2> -> tensor<128x2x2xi16, #blocked3>
    %16 = tt.join %arg0, %arg0 : tensor<128x2xi16, #blocked> -> tensor<128x2x2xi16, #blocked2>
    %17 = ttg.convert_layout %16 : tensor<128x2x2xi16, #blocked2> -> tensor<128x2x2xi16, #blocked3>
    %18 = tt.join %15, %17 : tensor<128x2x2xi16, #blocked3> -> tensor<128x2x2x2xi16, #blocked4>
    %19 = ttg.convert_layout %18 : tensor<128x2x2x2xi16, #blocked4> -> tensor<128x2x2x2xi16, #blocked5>
    %20 = tt.join %arg0, %arg0 : tensor<128x2xi16, #blocked> -> tensor<128x2x2xi16, #blocked2>
    %21 = ttg.convert_layout %20 : tensor<128x2x2xi16, #blocked2> -> tensor<128x2x2xi16, #blocked3>
    %22 = tt.join %arg0, %arg0 : tensor<128x2xi16, #blocked> -> tensor<128x2x2xi16, #blocked2>
    %23 = ttg.convert_layout %22 : tensor<128x2x2xi16, #blocked2> -> tensor<128x2x2xi16, #blocked3>
    %24 = tt.join %21, %23 : tensor<128x2x2xi16, #blocked3> -> tensor<128x2x2x2xi16, #blocked4>
    %25 = ttg.convert_layout %24 : tensor<128x2x2x2xi16, #blocked4> -> tensor<128x2x2x2xi16, #blocked5>
    %26 = tt.join %19, %25 : tensor<128x2x2x2xi16, #blocked5> -> tensor<128x2x2x2x2xi16, #blocked6>
    %27 = ttg.convert_layout %26 : tensor<128x2x2x2x2xi16, #blocked6> -> tensor<128x2x2x2x2xi16, #blocked7>
    %28 = tt.join %13, %27 : tensor<128x2x2x2x2xi16, #blocked7> -> tensor<128x2x2x2x2x2xi16, #blocked8>
    %29 = tt.reshape %28 : tensor<128x2x2x2x2x2xi16, #blocked8> -> tensor<128x32xi16, #blocked1>
    tt.return %29 : tensor<128x32xi16, #blocked1>
  }
}

// -----

// BASELINE-LABEL: @share_dominating_layout_conversions
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @share_dominating_layout_conversions
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return %[[CONVERT:.*]], %[[CONVERT]]

#source = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#target = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @share_dominating_layout_conversions(%arg0: tensor<128xi32, #source>) -> (tensor<128xi32, #target>, tensor<128xi32, #target>) {
    %0 = ttg.convert_layout %arg0 : tensor<128xi32, #source> -> tensor<128xi32, #target>
    %1 = ttg.convert_layout %arg0 : tensor<128xi32, #source> -> tensor<128xi32, #target>
    tt.return %0, %1 : tensor<128xi32, #target>, tensor<128xi32, #target>
  }
}
