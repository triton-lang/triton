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

// Both results of tt.split share one inferred parent layout. Optimize their
// fixed consumers together and keep the sole conversion on the source side.
//
// BASELINE-LABEL: @layout_conflict_split_fanout
// BASELINE: tt.split
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_split_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED: tt.split
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#split_target = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_split_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16x2xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %converted = ttg.convert_layout %source : tensor<16x16x2xf32, #source> -> tensor<16x16x2xf32, #split_target>
    %left, %right = tt.split %converted : tensor<16x16x2xf32, #split_target> -> tensor<16x16xf32, #target>
    %left_first = arith.addf %left, %target : tensor<16x16xf32, #target>
    %left_second = arith.mulf %left, %target : tensor<16x16xf32, #target>
    %right_first = arith.addf %right, %target : tensor<16x16xf32, #target>
    %right_second = arith.subf %right, %target : tensor<16x16xf32, #target>
    tt.return %left_first, %left_second, %right_first, %right_second : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// A value-and-index reduction must select one sliced layout for both outputs.
// Keep their fixed consumers in the same component and leave only the
// immutable source conversion.
//
// BASELINE-LABEL: @layout_conflict_multi_reduce_fanout
// BASELINE: "tt.reduce"
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_multi_reduce_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED: "tt.reduce"
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#reduced = #ttg.slice<{dim = 1, parent = #target}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_multi_reduce_fanout(%target: tensor<16x16xf32, #target>, %indices: tensor<16x16xi32, #target>, %source: tensor<16x16xf32, #source>, %reference: tensor<16xf32, #reduced>, %index_reference: tensor<16xi32, #reduced>) -> (tensor<16xf32, #reduced>, tensor<16xf32, #reduced>, tensor<16xi32, #reduced>, tensor<16xi32, #reduced>) {
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %mixed = arith.addf %converted, %target : tensor<16x16xf32, #target>
    %result:2 = "tt.reduce"(%mixed, %indices) <{axis = 1 : i32}> ({
      ^bb0(%left_value: f32, %left_index: i32, %right_value: f32, %right_index: i32):
        %take_left = arith.cmpf oge, %left_value, %right_value : f32
        %value = arith.select %take_left, %left_value, %right_value : f32
        %index = arith.select %take_left, %left_index, %right_index : i32
        tt.reduce.return %value, %index : f32, i32
    }) : (tensor<16x16xf32, #target>, tensor<16x16xi32, #target>) -> (tensor<16xf32, #reduced>, tensor<16xi32, #reduced>)
    %first = arith.addf %result#0, %reference : tensor<16xf32, #reduced>
    %second = arith.mulf %result#0, %reference : tensor<16xf32, #reduced>
    %third = arith.addi %result#1, %index_reference : tensor<16xi32, #reduced>
    %fourth = arith.subi %result#1, %index_reference : tensor<16xi32, #reduced>
    tt.return %first, %second, %third, %fourth : tensor<16xf32, #reduced>, tensor<16xf32, #reduced>, tensor<16xi32, #reduced>, tensor<16xi32, #reduced>
  }
}

// -----

// A while loop ties its initializer, before argument, condition yield, after
// argument, back-edge yield, and result. Keep the whole cycle in one encoding
// and place its only conversion at the immutable source boundary.
//
// BASELINE-LABEL: @layout_conflict_while_fanout
// BASELINE-COUNT-4: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_while_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_while_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %four = arith.constant 4 : i32
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result:2 = scf.while (%acc = %converted, %iteration = %zero) : (tensor<16x16xf32, #target>, i32) -> (tensor<16x16xf32, #target>, i32) {
      %continue = arith.cmpi slt, %iteration, %four : i32
      scf.condition(%continue) %acc, %iteration : tensor<16x16xf32, #target>, i32
    } do {
    ^bb0(%acc: tensor<16x16xf32, #target>, %iteration: i32):
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      %next_iteration = arith.addi %iteration, %one : i32
      scf.yield %next, %next_iteration : tensor<16x16xf32, #target>, i32
    }
    %first = arith.mulf %result#0, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result#0, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result#0, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// A store remains a fixed memory boundary inside the globally assigned while
// component. Preserve its pointer and value layouts without recreating a
// conversion for every loop-result consumer.
//
// BASELINE-LABEL: @layout_conflict_effectful_while_fanout
// BASELINE-COUNT-5: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_effectful_while_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_effectful_while_fanout(%ptr: !tt.ptr<f32>, %target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %four = arith.constant 4 : i32
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #target>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result:2 = scf.while (%acc = %converted, %iteration = %zero) : (tensor<16x16xf32, #target>, i32) -> (tensor<16x16xf32, #target>, i32) {
      %continue = arith.cmpi slt, %iteration, %four : i32
      scf.condition(%continue) %acc, %iteration : tensor<16x16xf32, #target>, i32
    } do {
    ^bb0(%acc: tensor<16x16xf32, #target>, %iteration: i32):
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      tt.store %ptrs, %next : tensor<16x16x!tt.ptr<f32>, #target>
      %next_iteration = arith.addi %iteration, %one : i32
      scf.yield %next, %next_iteration : tensor<16x16xf32, #target>, i32
    }
    %first = arith.mulf %result#0, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result#0, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result#0, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
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

// -----

// The target input and all three function results are fixed boundaries. The
// legacy first-layout choice performs the arithmetic in the other input's
// layout, converting the target and then each result independently. Global
// conflict resolution must keep the three consumers in their target layout and
// materialize the unavoidable conversion exactly once.
//
// BASELINE-LABEL: @layout_conflict_fanout
// BASELINE-COUNT-4: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %mixed = arith.addf %converted, %target : tensor<16x16xf32, #target>
    %first = arith.mulf %mixed, %target : tensor<16x16xf32, #target>
    %second = arith.addf %mixed, %target : tensor<16x16xf32, #target>
    %third = arith.subf %mixed, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// A layout-preserving transpose changes the encoding as well as the logical
// axes. It must not turn off global conflict resolution for the rest of the
// expression or move a shared-memory conversion to each fixed result.
//
// BASELINE-LABEL: @layout_conflict_transpose_fanout
// BASELINE-COUNT-4: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_transpose_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_transpose_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %mixed = arith.addf %converted, %target : tensor<16x16xf32, #target>
    %transposed = tt.trans %mixed {order = array<i32: 1, 0>} : tensor<16x16xf32, #target> -> tensor<16x16xf32, #source>
    %restored = tt.trans %transposed {order = array<i32: 1, 0>} : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %first = arith.mulf %restored, %target : tensor<16x16xf32, #target>
    %second = arith.addf %restored, %target : tensor<16x16xf32, #target>
    %third = arith.subf %restored, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// Flattening and restoring a tensor must preserve logical element order. The
// globally cheapest physical conversion is still the single conversion at the
// input, not one conversion for each result; no allow_reorder is introduced.
//
// BASELINE-LABEL: @layout_conflict_exact_reshape_fanout
// BASELINE-COUNT-5: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_exact_reshape_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-NOT: allow_reorder
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#flat_target = #ttg.linear<{register = [[16], [32], [64], [128]], lane = [[1], [2], [4], [8], [0]], warp = [[0], [0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_exact_reshape_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %mixed = arith.addf %converted, %target : tensor<16x16xf32, #target>
    %flattened = tt.reshape %mixed : tensor<16x16xf32, #target> -> tensor<256xf32, #flat_target>
    %restored = tt.reshape %flattened : tensor<256xf32, #flat_target> -> tensor<16x16xf32, #target>
    %first = arith.mulf %restored, %target : tensor<16x16xf32, #target>
    %second = arith.addf %restored, %target : tensor<16x16xf32, #target>
    %third = arith.subf %restored, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// Dimension expansion ties a sliced source encoding to its parent encoding.
// Resolve that constraint across the full producer/consumer graph rather than
// performing a shared-memory conversion separately for each expanded result.
//
// BASELINE-LABEL: @layout_conflict_expand_dims_fanout
// BASELINE-COUNT-4: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_expand_dims_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source_parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#target_parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 0, 1]}>
#source = #ttg.slice<{dim = 1, parent = #source_parent}>
#target = #ttg.slice<{dim = 1, parent = #target_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_expand_dims_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x1x16xf32, #target_parent>, tensor<16x1x16xf32, #target_parent>, tensor<16x1x16xf32, #target_parent>) {
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %mixed = arith.addf %converted, %target : tensor<16x16xf32, #target>
    %expanded = tt.expand_dims %mixed {axis = 1 : i32} : tensor<16x16xf32, #target> -> tensor<16x1x16xf32, #target_parent>
    %target_expanded = tt.expand_dims %target {axis = 1 : i32} : tensor<16x16xf32, #target> -> tensor<16x1x16xf32, #target_parent>
    %first = arith.mulf %expanded, %target_expanded : tensor<16x1x16xf32, #target_parent>
    %second = arith.addf %expanded, %target_expanded : tensor<16x1x16xf32, #target_parent>
    %third = arith.subf %expanded, %target_expanded : tensor<16x1x16xf32, #target_parent>
    tt.return %first, %second, %third : tensor<16x1x16xf32, #target_parent>, tensor<16x1x16xf32, #target_parent>, tensor<16x1x16xf32, #target_parent>
  }
}

// -----

// A single-result reduction uniquely determines its sliced result encoding.
// Keep that relation in the global problem so one input conversion can serve
// every fixed consumer instead of converting both the reduction input and its
// sliced result.
//
// BASELINE-LABEL: @layout_conflict_reduce_fanout
// BASELINE-COUNT-2: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_reduce_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#reduced = #ttg.slice<{dim = 1, parent = #target}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_reduce_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %reference: tensor<16xf32, #reduced>) -> (tensor<16xf32, #reduced>, tensor<16xf32, #reduced>, tensor<16xf32, #reduced>) {
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %mixed = arith.addf %converted, %target : tensor<16x16xf32, #target>
    %reduction = "tt.reduce"(%mixed) <{axis = 1 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        tt.reduce.return %sum : f32
    }) : (tensor<16x16xf32, #target>) -> tensor<16xf32, #reduced>
    %first = arith.mulf %reduction, %reference : tensor<16xf32, #reduced>
    %second = arith.addf %reduction, %reference : tensor<16xf32, #reduced>
    %third = arith.subf %reduction, %reference : tensor<16xf32, #reduced>
    tt.return %first, %second, %third : tensor<16xf32, #reduced>, tensor<16xf32, #reduced>, tensor<16xf32, #reduced>
  }
}

// -----

// Keep both branches and their three fixed consumers in the globally selected
// result layout. The source argument is an immutable layout boundary, so its
// required conversion must not be removed.
//
// BASELINE-LABEL: @layout_conflict_conditional_fanout
// BASELINE-COUNT-4: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_conditional_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_conditional_fanout(%condition: i1, %target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %selected = scf.if %condition -> (tensor<16x16xf32, #target>) {
      %then = arith.addf %converted, %target : tensor<16x16xf32, #target>
      scf.yield %then : tensor<16x16xf32, #target>
    } else {
      %else = arith.subf %converted, %target : tensor<16x16xf32, #target>
      scf.yield %else : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %selected, %target : tensor<16x16xf32, #target>
    %second = arith.addf %selected, %target : tensor<16x16xf32, #target>
    %third = arith.subf %selected, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// An exact-order join must use one legal encoding for both of its inputs.
// Price the complete join component and its three fixed consumers together;
// the source argument still requires one physical layout conversion.
//
// BASELINE-LABEL: @layout_conflict_join_fanout
// BASELINE-COUNT-4: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_join_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#joined = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_join_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16x2xf32, #joined>, tensor<16x16x2xf32, #joined>, tensor<16x16x2xf32, #joined>) {
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %mixed = tt.join %converted, %target : tensor<16x16xf32, #target> -> tensor<16x16x2xf32, #joined>
    %reference = tt.join %target, %target : tensor<16x16xf32, #target> -> tensor<16x16x2xf32, #joined>
    %first = arith.mulf %mixed, %reference : tensor<16x16x2xf32, #joined>
    %second = arith.addf %mixed, %reference : tensor<16x16x2xf32, #joined>
    %third = arith.subf %mixed, %reference : tensor<16x16x2xf32, #joined>
    tt.return %first, %second, %third : tensor<16x16x2xf32, #joined>, tensor<16x16x2xf32, #joined>, tensor<16x16x2xf32, #joined>
  }
}

// -----

// A loop result and its region iter_arg must be assigned as one component.
// Hoist the single required conversion to the fixed source boundary instead
// of recreating three conversions for the result's fixed consumers.
//
// BASELINE-LABEL: @layout_conflict_for_fanout
// BASELINE-COUNT-4: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_for_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_for_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// Memory and synchronization remain fixed hardware-protocol boundaries inside
// a globally assigned loop. Preserve the store's pointer and value layouts
// while sharing the loop-carried conversion across every result consumer.
//
// BASELINE-LABEL: @layout_conflict_effectful_for_fanout
// BASELINE-COUNT-5: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @layout_conflict_effectful_for_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @layout_conflict_effectful_for_fanout(%ptr: !tt.ptr<f32>, %target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #target>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      tt.store %ptrs, %next : tensor<16x16x!tt.ptr<f32>, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}
