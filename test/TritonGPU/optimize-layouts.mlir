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

  // Explicitly permitting a reshape to reorder does not turn it into an opaque
  // physical-layout boundary. The incumbent already removes every conversion
  // in this original handoff variant, so the global pass must retain that
  // zero-conversion result without dropping the existing permission.
  //
  // BASELINE-LABEL: @stochastic_rounding_allow_reorder_join_chain
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.reshape {{.*}} allow_reorder
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @stochastic_rounding_allow_reorder_join_chain
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.reshape {{.*}} allow_reorder
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func @stochastic_rounding_allow_reorder_join_chain(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> tensor<32x64xi32, #mma> {
    %a0 = tt.splat %arg0 : i32 -> tensor<128xi32, #blocked1>
    %a1 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked1>
    %x0 = tt.splat %arg2 : i32 -> tensor<128xi32, #blocked1>
    %x1 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked1>

    %0 = tt.reshape %a0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %1 = tt.reshape %a1 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %2 = tt.join %0, %1 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %3 = ttg.convert_layout %2 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %4 = tt.reshape %3 allow_reorder : tensor<1x128x2xi32, #blocked4> -> tensor<256xi32, #blocked1>

    %5 = tt.reshape %x0 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %6 = tt.join %5, %5 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %7 = ttg.convert_layout %6 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %8 = tt.reshape %7 allow_reorder : tensor<1x128x2xi32, #blocked4> -> tensor<256xi32, #blocked1>

    %9 = arith.xori %4, %8 : tensor<256xi32, #blocked1>
    %10 = tt.reshape %3 allow_reorder : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
    %11 = tt.reshape %9 : tensor<256xi32, #blocked1> -> tensor<1x256xi32, #blocked2>
    %12 = tt.join %10, %11 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
    %13 = ttg.convert_layout %12 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
    %14 = tt.reshape %13 : tensor<1x256x2xi32, #linear1> -> tensor<512xi32, #linear2>

    %15 = tt.reshape %x1 : tensor<128xi32, #blocked1> -> tensor<1x128xi32, #blocked2>
    %16 = tt.join %15, %15 : tensor<1x128xi32, #blocked2> -> tensor<1x128x2xi32, #blocked3>
    %17 = ttg.convert_layout %16 : tensor<1x128x2xi32, #blocked3> -> tensor<1x128x2xi32, #blocked4>
    %18 = tt.reshape %17 allow_reorder : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
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
    %30 = tt.reshape %29 allow_reorder : tensor<1x128x2xi32, #blocked4> -> tensor<1x256xi32, #blocked2>
    %31 = tt.join %30, %30 : tensor<1x256xi32, #blocked2> -> tensor<1x256x2xi32, #blocked3>
    %32 = ttg.convert_layout %31 : tensor<1x256x2xi32, #blocked3> -> tensor<1x256x2xi32, #linear1>
    %33 = tt.reshape %32 : tensor<1x256x2xi32, #linear1> -> tensor<1x512xi32, #linear3>
    %34 = tt.join %33, %33 : tensor<1x512xi32, #linear3> -> tensor<1x512x2xi32, #linear4>
    %35 = tt.reshape %34 : tensor<1x512x2xi32, #linear4> -> tensor<1024xi32, #linear5>

    %36 = arith.xori %26, %35 : tensor<1024xi32, #linear5>
    %37 = tt.reshape %25 : tensor<1x512x2xi32, #linear4> -> tensor<1x1024xi32, #linear6>
    %38 = tt.reshape %36 : tensor<1024xi32, #linear5> -> tensor<1x1024xi32, #linear6>
    %39 = tt.join %37, %38 : tensor<1x1024xi32, #linear6> -> tensor<1x1024x2xi32, #linear7>
    %40 = tt.reshape %39 allow_reorder : tensor<1x1024x2xi32, #linear7> -> tensor<32x64xi32, #blocked>
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

// Keep both coalesced inputs, conditional branches, and row reductions in
// their existing zero-copy layout. A competing vector-layout consumer must
// not move either 64x64 operand into an inter-warp reduction layout. The
// independent conflict still needs its one necessary source conversion.
//
// BASELINE-LABEL: @coalesced_row_reduction_independent_fanout
// BASELINE: tt.load
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.load
// BASELINE-NOT: ttg.convert_layout
// BASELINE: "tt.reduce"
// BASELINE-NOT: ttg.convert_layout
// BASELINE: "tt.reduce"
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @coalesced_row_reduction_independent_fanout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: "tt.reduce"
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: "tt.reduce"
// OPTIMIZED: tt.store
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#row = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#interwarp = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#interwarp_slice = #ttg.slice<{dim = 1, parent = #interwarp}>
#vector = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @coalesced_row_reduction_independent_fanout(%condition: i1, %x_ptr: tensor<64x64x!tt.ptr<f16>, #row>, %dy_ptr: tensor<64x64x!tt.ptr<f16>, #row>, %out_ptr: tensor<64x64x!tt.ptr<f32>, #row>, %reference: tensor<64xf32, #vector>, %target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %x_half = tt.load %x_ptr : tensor<64x64x!tt.ptr<f16>, #row>
    %dy_half = tt.load %dy_ptr : tensor<64x64x!tt.ptr<f16>, #row>
    %x_reduction = ttg.convert_layout %x_half : tensor<64x64xf16, #row> -> tensor<64x64xf16, #interwarp>
    %dy_reduction = ttg.convert_layout %dy_half : tensor<64x64xf16, #row> -> tensor<64x64xf16, #interwarp>
    %x = arith.extf %x_reduction : tensor<64x64xf16, #interwarp> to tensor<64x64xf32, #interwarp>
    %dy = arith.extf %dy_reduction : tensor<64x64xf16, #interwarp> to tensor<64x64xf32, #interwarp>
    %x_selected = scf.if %condition -> (tensor<64x64xf32, #interwarp>) {
      %factor = arith.constant dense<2.000000e+00> : tensor<64x64xf32, #interwarp>
      %scaled = arith.mulf %x, %factor : tensor<64x64xf32, #interwarp>
      scf.yield %scaled : tensor<64x64xf32, #interwarp>
    } else {
      scf.yield %x : tensor<64x64xf32, #interwarp>
    }
    %dy_selected = scf.if %condition -> (tensor<64x64xf32, #interwarp>) {
      %factor = arith.constant dense<2.000000e+00> : tensor<64x64xf32, #interwarp>
      %scaled = arith.mulf %dy, %factor : tensor<64x64xf32, #interwarp>
      scf.yield %scaled : tensor<64x64xf32, #interwarp>
    } else {
      scf.yield %dy : tensor<64x64xf32, #interwarp>
    }
    %squared = arith.mulf %x_selected, %x_selected : tensor<64x64xf32, #interwarp>
    %norm = "tt.reduce"(%squared) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<64x64xf32, #interwarp>) -> tensor<64xf32, #interwarp_slice>
    %product = arith.mulf %x_selected, %dy_selected : tensor<64x64xf32, #interwarp>
    %dot = "tt.reduce"(%product) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<64x64xf32, #interwarp>) -> tensor<64xf32, #interwarp_slice>
    %norm_vector = ttg.convert_layout %norm : tensor<64xf32, #interwarp_slice> -> tensor<64xf32, #vector>
    %dot_vector = ttg.convert_layout %dot : tensor<64xf32, #interwarp_slice> -> tensor<64xf32, #vector>
    %row_scale = arith.addf %norm_vector, %reference : tensor<64xf32, #vector>
    %scaled_dot = arith.mulf %dot_vector, %row_scale : tensor<64xf32, #vector>
    %reduction_scale = ttg.convert_layout %scaled_dot : tensor<64xf32, #vector> -> tensor<64xf32, #interwarp_slice>
    %expanded = tt.expand_dims %reduction_scale {axis = 1 : i32} : tensor<64xf32, #interwarp_slice> -> tensor<64x1xf32, #interwarp>
    %broadcast = tt.broadcast %expanded : tensor<64x1xf32, #interwarp> -> tensor<64x64xf32, #interwarp>
    %row_broadcast = ttg.convert_layout %broadcast : tensor<64x64xf32, #interwarp> -> tensor<64x64xf32, #row>
    %dy_row = arith.extf %dy_half : tensor<64x64xf16, #row> to tensor<64x64xf32, #row>
    %result = arith.addf %row_broadcast, %dy_row : tensor<64x64xf32, #row>
    tt.store %out_ptr, %result : tensor<64x64x!tt.ptr<f32>, #row>

    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %first = arith.mulf %converted, %target : tensor<16x16xf32, #target>
    %second = arith.addf %converted, %target : tensor<16x16xf32, #target>
    %third = arith.subf %converted, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// Pairwise reductions in a high-rank register network jointly own their
// distributed layout. Protect that network without falling back to the legacy
// assignment for an independent loop and its three consumers.
//
// BASELINE-LABEL: @pairwise_reduction_network_independent_fanout
// BASELINE-COUNT-2: "tt.reduce"
// BASELINE: scf.for
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @pairwise_reduction_network_independent_fanout
// OPTIMIZED-COUNT-2: "tt.reduce"
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED: scf.for
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#network = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [4, 8, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#network_slice = #ttg.slice<{dim = 3, parent = #network}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @pairwise_reduction_network_independent_fanout(%network_input: tensor<16x4x2x2xi32, #network>, %target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x4x2xi32, #network_slice>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %first_pair = "tt.reduce"(%network_input) <{axis = 3 : i32}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %selected = arith.maxui %lhs, %rhs : i32
      tt.reduce.return %selected : i32
    }) : (tensor<16x4x2x2xi32, #network>) -> tensor<16x4x2xi32, #network_slice>
    %expanded = tt.expand_dims %first_pair {axis = 3 : i32} : tensor<16x4x2xi32, #network_slice> -> tensor<16x4x2x1xi32, #network>
    %broadcast = tt.broadcast %expanded : tensor<16x4x2x1xi32, #network> -> tensor<16x4x2x2xi32, #network>
    %mixed = arith.xori %network_input, %broadcast : tensor<16x4x2x2xi32, #network>
    %second_pair = "tt.reduce"(%mixed) <{axis = 3 : i32}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %selected = arith.maxui %lhs, %rhs : i32
      tt.reduce.return %selected : i32
    }) : (tensor<16x4x2x2xi32, #network>) -> tensor<16x4x2xi32, #network_slice>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %second_pair : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x4x2xi32, #network_slice>
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

#wide = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#narrow = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // A scalar result has no distributed layout. The reduction still needs to
  // account for lane communication, but does not require an input conversion.
  //
  // BASELINE-LABEL: @predicate_reduction_avoids_shared_conversion
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: arith.extui
  // BASELINE: "tt.reduce"
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @predicate_reduction_avoids_shared_conversion
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: arith.extui
  // OPTIMIZED: "tt.reduce"
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @predicate_reduction_avoids_shared_conversion(
      %predicate: tensor<4096xi1, #wide>) -> i32 {
    %converted = ttg.convert_layout %predicate : tensor<4096xi1, #wide> -> tensor<4096xi1, #narrow>
    %extended = arith.extui %converted : tensor<4096xi1, #narrow> to tensor<4096xi32, #narrow>
    %result = "tt.reduce"(%extended) <{axis = 0 : i32}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      tt.reduce.return %sum : i32
    }) : (tensor<4096xi32, #narrow>) -> i32
    tt.return %result : i32
  }

  // Scalar consumers must not weaken the atomic flag's fixed memory contract.
  //
  // BASELINE-LABEL: @predicate_comparison_reduction_avoids_shared_conversion
  // BASELINE: arith.cmpf
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: arith.extui
  // BASELINE: "tt.reduce"
  // BASELINE: tt.atomic_rmw or
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @predicate_comparison_reduction_avoids_shared_conversion
  // OPTIMIZED: arith.cmpf
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: arith.extui
  // OPTIMIZED: "tt.reduce"
  // OPTIMIZED: tt.atomic_rmw or
  // OPTIMIZED: tt.return
  tt.func public @predicate_comparison_reduction_avoids_shared_conversion(
      %lhs: tensor<4096xbf16, #wide>,
      %rhs: tensor<4096xbf16, #wide>,
      %flag: !tt.ptr<i32>) {
    %ones = arith.constant dense<true> : tensor<4096xi1, #wide>
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %enabled = arith.constant true
    %lhs_f32 = arith.extf %lhs : tensor<4096xbf16, #wide> to tensor<4096xf32, #wide>
    %rhs_f32 = arith.extf %rhs : tensor<4096xbf16, #wide> to tensor<4096xf32, #wide>
    %equal = arith.cmpf oeq, %lhs_f32, %rhs_f32 : tensor<4096xf32, #wide>
    %different = arith.xori %equal, %ones : tensor<4096xi1, #wide>
    %converted = ttg.convert_layout %different : tensor<4096xi1, #wide> -> tensor<4096xi1, #narrow>
    %extended = arith.extui %converted : tensor<4096xi1, #narrow> to tensor<4096xi32, #narrow>
    %result = "tt.reduce"(%extended) <{axis = 0 : i32}> ({
    ^bb0(%lhs_sum: i32, %rhs_sum: i32):
      %sum = arith.addi %lhs_sum, %rhs_sum : i32
      tt.reduce.return %sum : i32
    }) : (tensor<4096xi32, #narrow>) -> i32
    %any = arith.cmpi ne, %result, %zero : i32
    scf.if %any {
      %old = tt.atomic_rmw or, acq_rel, gpu, %flag, %one, %enabled : (!tt.ptr<i32>, i32, i1) -> i32
    }
    tt.return
  }

  // Multi-result scalar reductions still require all their tensor inputs to
  // share one encoding. Remove both conversions only when both fixed sources
  // already have the same layout.
  //
  // BASELINE-LABEL: @paired_scalar_reduction_removes_both_conversions
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: "tt.reduce"
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @paired_scalar_reduction_removes_both_conversions
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: "tt.reduce"
  // OPTIMIZED: tt.return
  tt.func public @paired_scalar_reduction_removes_both_conversions(
      %values: tensor<4096xf32, #wide>,
      %indices: tensor<4096xi32, #wide>) -> (f32, i32) {
    %narrow_values = ttg.convert_layout %values : tensor<4096xf32, #wide> -> tensor<4096xf32, #narrow>
    %narrow_indices = ttg.convert_layout %indices : tensor<4096xi32, #wide> -> tensor<4096xi32, #narrow>
    %result:2 = "tt.reduce"(%narrow_values, %narrow_indices) <{axis = 0 : i32}> ({
    ^bb0(%left_value: f32, %left_index: i32, %right_value: f32, %right_index: i32):
      %take_left = arith.cmpf oge, %left_value, %right_value : f32
      %value = arith.select %take_left, %left_value, %right_value : f32
      %index = arith.select %take_left, %left_index, %right_index : i32
      tt.reduce.return %value, %index : f32, i32
    }) : (tensor<4096xf32, #narrow>, tensor<4096xi32, #narrow>) -> (f32, i32)
    tt.return %result#0, %result#1 : f32, i32
  }

  // Inputs with genuinely different fixed layouts retain the one conversion
  // needed to satisfy the reduction's same-operand-encoding invariant.
  //
  // BASELINE-LABEL: @paired_scalar_reduction_preserves_required_conversion
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: "tt.reduce"
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @paired_scalar_reduction_preserves_required_conversion
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED: "tt.reduce"
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @paired_scalar_reduction_preserves_required_conversion(
      %values: tensor<4096xf32, #wide>,
      %indices: tensor<4096xi32, #narrow>) -> (f32, i32) {
    %narrow_values = ttg.convert_layout %values : tensor<4096xf32, #wide> -> tensor<4096xf32, #narrow>
    %result:2 = "tt.reduce"(%narrow_values, %indices) <{axis = 0 : i32}> ({
    ^bb0(%left_value: f32, %left_index: i32, %right_value: f32, %right_index: i32):
      %take_left = arith.cmpf oge, %left_value, %right_value : f32
      %value = arith.select %take_left, %left_value, %right_value : f32
      %index = arith.select %take_left, %left_index, %right_index : i32
      tt.reduce.return %value, %index : f32, i32
    }) : (tensor<4096xf32, #narrow>, tensor<4096xi32, #narrow>) -> (f32, i32)
    tt.return %result#0, %result#1 : f32, i32
  }

  // Both reductions consume the same producer, so removing the conversion
  // must not duplicate that producer or insert one conversion per reduction.
  //
  // BASELINE-LABEL: @shared_scalar_reduction_fanout_avoids_duplication
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE-COUNT-2: "tt.reduce"
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @shared_scalar_reduction_fanout_avoids_duplication
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED-COUNT-2: "tt.reduce"
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @shared_scalar_reduction_fanout_avoids_duplication(
      %predicate: tensor<4096xi1, #wide>) -> (i32, i32) {
    %converted = ttg.convert_layout %predicate : tensor<4096xi1, #wide> -> tensor<4096xi1, #narrow>
    %extended = arith.extui %converted : tensor<4096xi1, #narrow> to tensor<4096xi32, #narrow>
    %sum = "tt.reduce"(%extended) <{axis = 0 : i32}> ({
    ^bb0(%sum_lhs: i32, %sum_rhs: i32):
      %result = arith.addi %sum_lhs, %sum_rhs : i32
      tt.reduce.return %result : i32
    }) : (tensor<4096xi32, #narrow>) -> i32
    %maximum = "tt.reduce"(%extended) <{axis = 0 : i32}> ({
    ^bb0(%max_lhs: i32, %max_rhs: i32):
      %result = arith.maxui %max_lhs, %max_rhs : i32
      tt.reduce.return %result : i32
    }) : (tensor<4096xi32, #narrow>) -> i32
    tt.return %sum, %maximum : i32, i32
  }

  // Hoisting across a scalar loop must retain its execution semantics without
  // charging the reduction for a nonexistent result layout.
  //
  // BASELINE-LABEL: @loop_scalar_reduction_avoids_conversion
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: scf.for
  // BASELINE: "tt.reduce"
  // BASELINE: tt.return
  //
  // OPTIMIZED-LABEL: @loop_scalar_reduction_avoids_conversion
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: scf.for
  // OPTIMIZED: "tt.reduce"
  // OPTIMIZED: tt.return
  tt.func public @loop_scalar_reduction_avoids_conversion(
      %predicate: tensor<4096xi1, #wide>) -> i32 {
    %zero = arith.constant 0 : i32
    %start = arith.constant 0 : index
    %stop = arith.constant 4 : index
    %step = arith.constant 1 : index
    %converted = ttg.convert_layout %predicate : tensor<4096xi1, #wide> -> tensor<4096xi1, #narrow>
    %extended = arith.extui %converted : tensor<4096xi1, #narrow> to tensor<4096xi32, #narrow>
    %result = scf.for %iteration = %start to %stop step %step iter_args(%accumulator = %zero) -> i32 {
      %sum = "tt.reduce"(%extended) <{axis = 0 : i32}> ({
      ^bb0(%lhs: i32, %rhs: i32):
        %combined = arith.addi %lhs, %rhs : i32
        tt.reduce.return %combined : i32
      }) : (tensor<4096xi32, #narrow>) -> i32
      %next = arith.addi %accumulator, %sum : i32
      scf.yield %next : i32
    }
    tt.return %result : i32
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

// -----

// A tensor-core accumulation loop owns the dot operand and accumulator
// layouts. Protect that complete protocol without forcing an independent
// loop-carried fanout in the same function back to legacy layout assignment.
//
// BASELINE-LABEL: @hardware_dot_loop_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.dot
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_dot_loop_independent_fanout
// OPTIMIZED: tt.dot
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_dot_loop_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %lhs: tensor<16x16xf16, #dot_a>, %rhs: tensor<16x16xf16, #dot_b>, %initial: tensor<16x16xf32, #mma>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #mma>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %hardware = scf.for %iv = %zero to %four step %one iter_args(%acc = %initial) -> (tensor<16x16xf32, #mma>) {
      %next = tt.dot %lhs, %rhs, %acc : tensor<16x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<16x16xf32, #mma>
      scf.yield %next : tensor<16x16xf32, #mma>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #mma>
  }
}

// -----

// The index range used by a tensor-core store also has an independent
// consumer in a conflicting layout. Rematerialize the range for that consumer
// instead of converting the coalesced store pointer or mask. Keep the one
// required accumulator conversion and optimize the independent fanout.
//
// BASELINE-LABEL: @hardware_dot_store_address_independent_fanout
// BASELINE: tt.dot
// BASELINE: ttg.convert_layout
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_dot_store_address_independent_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED: tt.dot
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#source_row = #ttg.slice<{dim = 1, parent = #source}>
#target_row = #ttg.slice<{dim = 1, parent = #target}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_dot_store_address_independent_fanout(%ptr: !tt.ptr<f32>, %bound: i32, %target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %lhs: tensor<16x16xf16, #dot_a>, %rhs: tensor<16x16xf16, #dot_b>, %initial: tensor<16x16xf32, #mma>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #mma>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %hardware = scf.for %iv = %zero to %four step %one iter_args(%acc = %initial) -> (tensor<16x16xf32, #mma>) {
      %next = tt.dot %lhs, %rhs, %acc : tensor<16x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<16x16xf32, #mma>
      scf.yield %next : tensor<16x16xf32, #mma>
    }

    %offsets = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #target_row>
    %expanded_offsets = tt.expand_dims %offsets {axis = 1 : i32} : tensor<16xi32, #target_row> -> tensor<16x1xi32, #target>
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #target>
    %row_ptrs = tt.addptr %base, %expanded_offsets : tensor<16x1x!tt.ptr<f32>, #target>, tensor<16x1xi32, #target>
    %ptrs = tt.broadcast %row_ptrs : tensor<16x1x!tt.ptr<f32>, #target> -> tensor<16x16x!tt.ptr<f32>, #target>
    %bounds = tt.splat %bound : i32 -> tensor<16xi32, #target_row>
    %row_mask = arith.cmpi slt, %offsets, %bounds : tensor<16xi32, #target_row>
    %expanded_mask = tt.expand_dims %row_mask {axis = 1 : i32} : tensor<16xi1, #target_row> -> tensor<16x1xi1, #target>
    %mask = tt.broadcast %expanded_mask : tensor<16x1xi1, #target> -> tensor<16x16xi1, #target>
    %epilogue = ttg.convert_layout %hardware : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #target>
    tt.store %ptrs, %epilogue, %mask : tensor<16x16x!tt.ptr<f32>, #target>

    %source_offsets = ttg.convert_layout %offsets : tensor<16xi32, #target_row> -> tensor<16xi32, #source_row>
    %source_column = tt.expand_dims %source_offsets {axis = 1 : i32} : tensor<16xi32, #source_row> -> tensor<16x1xi32, #source>
    %source_grid = tt.broadcast %source_column : tensor<16x1xi32, #source> -> tensor<16x16xi32, #source>
    %source_grid_float = arith.sitofp %source_grid : tensor<16x16xi32, #source> to tensor<16x16xf32, #source>
    %indexed_source = arith.addf %source, %source_grid_float : tensor<16x16xf32, #source>
    %converted = ttg.convert_layout %indexed_source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #mma>
  }
}

// -----

// A tensor-memory load has a hardware-owned result encoding. Keep that load
// and its loop-carried accumulator unchanged while globally optimizing an
// unrelated expression in the same function.
//
// BASELINE-LABEL: @hardware_tmem_loop_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: ttng.tmem_load
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_tmem_loop_independent_fanout
// OPTIMIZED: ttng.tmem_load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#tmem_result = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_tmem_loop_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %memory: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %initial: tensor<128x128xf32, #tmem_result>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<128x128xf32, #tmem_result>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %hardware = scf.for %iv = %zero to %four step %one iter_args(%acc = %initial) -> (tensor<128x128xf32, #tmem_result>) {
      %loaded = ttng.tmem_load %memory : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #tmem_result>
      %next = arith.addf %acc, %loaded : tensor<128x128xf32, #tmem_result>
      scf.yield %next : tensor<128x128xf32, #tmem_result>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<128x128xf32, #tmem_result>
  }
}

// -----

// Atomic pointer, value, mask, result, and loop-carried layouts are one
// protocol. Protect them without disabling the independent global component.
//
// BASELINE-LABEL: @hardware_atomic_loop_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.atomic_rmw
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_atomic_loop_independent_fanout
// OPTIMIZED: tt.atomic_rmw
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_atomic_loop_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %counter: !tt.ptr<i32>, %initial: tensor<16x16xi32, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xi32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<16x16xi1, #target>
    %ptrs = tt.splat %counter : !tt.ptr<i32> -> tensor<16x16x!tt.ptr<i32>, #target>
    %hardware = scf.for %iv = %zero to %four step %one iter_args(%acc = %initial) -> (tensor<16x16xi32, #target>) {
      %next = tt.atomic_rmw add, relaxed, gpu, %ptrs, %acc, %mask : (tensor<16x16x!tt.ptr<i32>, #target>, tensor<16x16xi32, #target>, tensor<16x16xi1, #target>) -> tensor<16x16xi32, #target>
      scf.yield %next : tensor<16x16xi32, #target>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xi32, #target>
  }
}

// -----

// Descriptor loads own both their TMA descriptor and result encoding. Freeze
// the complete descriptor loop without sacrificing an independent component.
//
// BASELINE-LABEL: @hardware_descriptor_loop_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.descriptor_load
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_descriptor_loop_independent_fanout
// OPTIMIZED: tt.descriptor_load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_descriptor_loop_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %descriptor: !tt.tensordesc<128x64xf16, #nvmma_128>, %initial: tensor<128x64xf16, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<128x64xf16, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %coord = arith.constant 0 : i32
    %hardware = scf.for %iv = %zero to %four step %one iter_args(%acc = %initial) -> (tensor<128x64xf16, #target>) {
      %loaded = tt.descriptor_load %descriptor[%coord, %coord] : !tt.tensordesc<128x64xf16, #nvmma_128> -> tensor<128x64xf16, #target>
      %next = arith.addf %acc, %loaded : tensor<128x64xf16, #target>
      scf.yield %next : tensor<128x64xf16, #target>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<128x64xf16, #target>
  }
}

// -----

// Descriptor stores also participate in the hardware-owned protocol. Keep
// the descriptor load, store, and loop-carried layout together while still
// globally optimizing an independent loop and its three consumers.
//
// BASELINE-LABEL: @hardware_descriptor_store_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.descriptor_load
// BASELINE: tt.descriptor_store
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_descriptor_store_independent_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED: tt.descriptor_load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.descriptor_store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_descriptor_store_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %input: !tt.tensordesc<128x64xf16, #nvmma_128>, %output: !tt.tensordesc<128x64xf16, #nvmma_128>, %initial: tensor<128x64xf16, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<128x64xf16, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %coord = arith.constant 0 : i32
    %hardware = scf.for %iv = %zero to %four step %one iter_args(%acc = %initial) -> (tensor<128x64xf16, #target>) {
      %loaded = tt.descriptor_load %input[%coord, %coord] : !tt.tensordesc<128x64xf16, #nvmma_128> -> tensor<128x64xf16, #target>
      %next = arith.addf %acc, %loaded : tensor<128x64xf16, #target>
      tt.descriptor_store %output[%coord, %coord], %next : !tt.tensordesc<128x64xf16, #nvmma_128>, tensor<128x64xf16, #target>
      scf.yield %next : tensor<128x64xf16, #target>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<128x64xf16, #target>
  }
}

// -----

// TMA lowering replaces descriptor stores with a local store, a proxy fence,
// and an asynchronous shared-to-global copy before layout assignment runs
// again. Preserve the lowered store protocol and the independent fanout.
//
// BASELINE-LABEL: @hardware_lowered_tma_store_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: ttg.local_store
// BASELINE: ttng.async_tma_copy_local_to_global
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_lowered_tma_store_independent_fanout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED: ttg.local_store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: ttng.async_tma_copy_local_to_global
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_lowered_tma_store_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %output: !tt.tensordesc<128x64xf16, #nvmma_128>, %initial: tensor<128x64xf16, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<128x64xf16, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %coord = arith.constant 0 : i32
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #nvmma_128, #smem, mutable>
    %hardware = scf.for %iv = %zero to %four step %one iter_args(%acc = %initial) -> (tensor<128x64xf16, #target>) {
      %next = arith.addf %acc, %initial : tensor<128x64xf16, #target>
      ttg.local_store %next, %buffer : tensor<128x64xf16, #target> -> !ttg.memdesc<128x64xf16, #nvmma_128, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttng.async_tma_copy_local_to_global %output[%coord, %coord] %buffer : !tt.tensordesc<128x64xf16, #nvmma_128>, !ttg.memdesc<128x64xf16, #nvmma_128, #smem, mutable>
      scf.yield %next : tensor<128x64xf16, #target>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<128x64xf16, #target>
  }
}

// -----

// Synchronization cannot be rematerialized, moved across the loop, or treated
// as an ordinary memory boundary. Preserve the barrier and its loop while
// reducing the independent fanout to its single necessary conversion.
//
// BASELINE-LABEL: @hardware_barrier_loop_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: ttg.barrier local
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_barrier_loop_independent_fanout
// OPTIMIZED: ttg.barrier local
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_barrier_loop_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %initial: tensor<16x16xf32, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %hardware = scf.for %iv = %zero to %four step %one iter_args(%acc = %initial) -> (tensor<16x16xf32, #target>) {
      ttg.barrier local
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }
}

// -----

// A protected while protocol ties the initializer, before and after arguments,
// condition, back-edge, atomic result, and loop result. Protect that entire
// cycle while independently optimizing the ordinary for-loop component.
//
// BASELINE-LABEL: @hardware_atomic_while_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.atomic_rmw
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @hardware_atomic_while_independent_fanout
// OPTIMIZED: tt.atomic_rmw
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hardware_atomic_while_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %counter: !tt.ptr<i32>, %initial: tensor<16x16xi32, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xi32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %zero_i32 = arith.constant 0 : i32
    %one_i32 = arith.constant 1 : i32
    %four_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<16x16xi1, #target>
    %ptrs = tt.splat %counter : !tt.ptr<i32> -> tensor<16x16x!tt.ptr<i32>, #target>
    %hardware:2 = scf.while (%acc = %initial, %iteration = %zero_i32) : (tensor<16x16xi32, #target>, i32) -> (tensor<16x16xi32, #target>, i32) {
      %continue = arith.cmpi slt, %iteration, %four_i32 : i32
      scf.condition(%continue) %acc, %iteration : tensor<16x16xi32, #target>, i32
    } do {
    ^bb0(%acc: tensor<16x16xi32, #target>, %iteration: i32):
      %next = tt.atomic_rmw add, relaxed, gpu, %ptrs, %acc, %mask : (tensor<16x16x!tt.ptr<i32>, #target>, tensor<16x16xi32, #target>, tensor<16x16xi1, #target>) -> tensor<16x16xi32, #target>
      %next_iteration = arith.addi %iteration, %one_i32 : i32
      scf.yield %next, %next_iteration : tensor<16x16xi32, #target>, i32
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %hardware#0 : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xi32, #target>
  }
}

// -----

// Existing permuting reshapes retain their exact source, result, and
// allow_reorder contract. They must not force an independent loop component
// back to the legacy assignment or introduce new permutation permissions.
//
// BASELINE-LABEL: @opaque_reorder_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.reshape {{.*}} allow_reorder
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @opaque_reorder_independent_fanout
// OPTIMIZED: tt.reshape {{.*}} allow_reorder
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#flat = #ttg.linear<{register = [[16], [32], [64], [128]], lane = [[1], [2], [4], [8], [0]], warp = [[0], [0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @opaque_reorder_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<256xf32, #flat>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = tt.reshape %target allow_reorder : tensor<16x16xf32, #target> -> tensor<256xf32, #flat>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<256xf32, #flat>
  }
}

// -----

// An efficient exact-order reshape is a local boundary as well. Preserve its
// efficient_layout attribute without introducing allow_reorder.
//
// BASELINE-LABEL: @opaque_efficient_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.reshape {{.*}} efficient_layout
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @opaque_efficient_independent_fanout
// OPTIMIZED: tt.reshape {{.*}} efficient_layout
// OPTIMIZED-NOT: allow_reorder
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#flat = #ttg.linear<{register = [[16], [32], [64], [128]], lane = [[1], [2], [4], [8], [0]], warp = [[0], [0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @opaque_efficient_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<256xf32, #flat>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = tt.reshape %target efficient_layout : tensor<16x16xf32, #target> -> tensor<256xf32, #flat>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<256xf32, #flat>
  }
}

// -----

// Preserve a reshape carrying both contracts without propagating either
// permission into the independent globally optimized loop.
//
// BASELINE-LABEL: @opaque_reorder_efficient_independent_fanout
// BASELINE-COUNT-1: ttg.convert_layout
// BASELINE: tt.reshape {{.*}} allow_reorder efficient_layout
// BASELINE-COUNT-3: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @opaque_reorder_efficient_independent_fanout
// OPTIMIZED: tt.reshape {{.*}} allow_reorder efficient_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#flat = #ttg.linear<{register = [[16], [32], [64], [128]], lane = [[1], [2], [4], [8], [0]], warp = [[0], [0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @opaque_reorder_efficient_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<256xf32, #flat>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = tt.reshape %target allow_reorder efficient_layout : tensor<16x16xf32, #target> -> tensor<256xf32, #flat>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<256xf32, #flat>
  }
}

// -----

// Keep unsupported operations and protected control-flow protocols local.
// A legal gather, concatenation, histogram, or unusual while shape must not
// force unrelated components back to legacy layout assignment.

#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#cat_source = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#cat_result = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // BASELINE-LABEL: @opaque_gather_independent_fanout
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: tt.gather
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  // OPTIMIZED-LABEL: @opaque_gather_independent_fanout
  // OPTIMIZED: tt.gather
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @opaque_gather_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %indices: tensor<16x16xi32, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = tt.gather %target[%indices] {axis = 1 : i32} : (tensor<16x16xf32, #target>, tensor<16x16xi32, #target>) -> tensor<16x16xf32, #target>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }

  // BASELINE-LABEL: @opaque_efficient_gather_independent_fanout
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: tt.gather {{.*}}efficient_layout
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  // OPTIMIZED-LABEL: @opaque_efficient_gather_independent_fanout
  // OPTIMIZED: tt.gather {{.*}}efficient_layout
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @opaque_efficient_gather_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %indices: tensor<16x16xi32, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = tt.gather %target[%indices] {axis = 1 : i32, efficient_layout} : (tensor<16x16xf32, #target>, tensor<16x16xi32, #target>) -> tensor<16x16xf32, #target>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }

  // BASELINE-LABEL: @opaque_scan_independent_fanout
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: tt.scan
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  // OPTIMIZED-LABEL: @opaque_scan_independent_fanout
  // OPTIMIZED: tt.scan
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @opaque_scan_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = "tt.scan"(%target) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.scan.return %sum : f32
    }) : (tensor<16x16xf32, #target>) -> tensor<16x16xf32, #target>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }

  // BASELINE-LABEL: @opaque_cat_independent_fanout
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: tt.cat
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  // OPTIMIZED-LABEL: @opaque_cat_independent_fanout
  // OPTIMIZED: tt.cat
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @opaque_cat_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %cat_input: tensor<16xf32, #cat_source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<32xf32, #cat_result>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = tt.cat %cat_input, %cat_input : tensor<16xf32, #cat_source> -> tensor<32xf32, #cat_result>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<32xf32, #cat_result>
  }

  // BASELINE-LABEL: @opaque_histogram_independent_fanout
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: tt.histogram
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  // OPTIMIZED-LABEL: @opaque_histogram_independent_fanout
  // OPTIMIZED: tt.histogram
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @opaque_histogram_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %indices: tensor<128xi32, #cat_source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16xi32, #cat_source>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = tt.histogram %indices : tensor<128xi32, #cat_source> -> tensor<16xi32, #cat_source>
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16xi32, #cat_source>
  }

  // BASELINE-LABEL: @opaque_gather_loop_independent_fanout
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: tt.gather
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  // OPTIMIZED-LABEL: @opaque_gather_loop_independent_fanout
  // OPTIMIZED: tt.gather
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @opaque_gather_loop_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %indices: tensor<16x16xi32, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = scf.for %iv = %zero to %four step %one iter_args(%acc = %target) -> (tensor<16x16xf32, #target>) {
      %gathered = tt.gather %acc[%indices] {axis = 1 : i32, efficient_layout} : (tensor<16x16xf32, #target>, tensor<16x16xi32, #target>) -> tensor<16x16xf32, #target>
      scf.yield %gathered : tensor<16x16xf32, #target>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }

  // BASELINE-LABEL: @opaque_multi_scan_independent_fanout
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: tt.scan
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  // OPTIMIZED-LABEL: @opaque_multi_scan_independent_fanout
  // OPTIMIZED: tt.scan
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @opaque_multi_scan_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque:2 = "tt.scan"(%target, %target) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs0: f32, %lhs1: f32, %rhs0: f32, %rhs1: f32):
      %sum0 = arith.addf %lhs0, %rhs0 : f32
      %sum1 = arith.addf %lhs1, %rhs1 : f32
      tt.scan.return %sum0, %sum1 : f32, f32
    }) : (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>)
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque#0, %opaque#1 : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }

  // A before-region tensor without a corresponding result is valid SCF.
  // Both layout passes must preserve its loop contract without indexing past
  // the while results or disabling the independent global fanout.
  // BASELINE-LABEL: @opaque_while_extra_tensor_independent_fanout
  // BASELINE-COUNT-1: ttg.convert_layout
  // BASELINE: scf.while
  // BASELINE-COUNT-3: ttg.convert_layout
  // BASELINE-NOT: ttg.convert_layout
  // BASELINE: tt.return
  // OPTIMIZED-LABEL: @opaque_while_extra_tensor_independent_fanout
  // OPTIMIZED: scf.while
  // OPTIMIZED-COUNT-1: ttg.convert_layout
  // OPTIMIZED-NOT: ttg.convert_layout
  // OPTIMIZED: tt.return
  tt.func public @opaque_while_extra_tensor_independent_fanout(%target: tensor<16x16xf32, #target>, %source: tensor<16x16xf32, #source>, %initial: tensor<16x16xf32, #target>, %condition: i1) -> (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %opaque = scf.while (%acc = %initial, %extra = %target) : (tensor<16x16xf32, #target>, tensor<16x16xf32, #target>) -> tensor<16x16xf32, #target> {
      scf.condition(%condition) %acc : tensor<16x16xf32, #target>
    } do {
    ^bb0(%acc: tensor<16x16xf32, #target>):
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next, %target : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
    }
    %converted = ttg.convert_layout %source : tensor<16x16xf32, #source> -> tensor<16x16xf32, #target>
    %result = scf.for %iv = %zero to %four step %one iter_args(%acc = %converted) -> (tensor<16x16xf32, #target>) {
      %next = arith.addf %acc, %target : tensor<16x16xf32, #target>
      scf.yield %next : tensor<16x16xf32, #target>
    }
    %first = arith.mulf %result, %target : tensor<16x16xf32, #target>
    %second = arith.addf %result, %target : tensor<16x16xf32, #target>
    %third = arith.subf %result, %target : tensor<16x16xf32, #target>
    tt.return %first, %second, %third, %opaque : tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>, tensor<16x16xf32, #target>
  }

}

// -----

// Packed inline assembly operates on groups of register elements, so moving
// its layout conversion across the instruction can change the resulting
// logical values. Preserve both packed results in their original layout.
//
// BASELINE-LABEL: @packed_inline_asm_register_grouping
// BASELINE-SAME: %[[BASE_INPUT:[a-zA-Z0-9_]+]]: tensor<16x16xi8, #[[BASE_LAYOUT:[a-zA-Z0-9_]+]]>
// BASELINE: tt.elementwise_inline_asm {{.*}}packed_element = 4{{.*}} %[[BASE_INPUT]] : tensor<16x16xi8, #[[BASE_LAYOUT]]> -> tensor<16x16xi8, #[[BASE_LAYOUT]]>, tensor<16x16xi8, #[[BASE_LAYOUT]]>
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @packed_inline_asm_register_grouping
// OPTIMIZED-SAME: %[[PACKED_INPUT:[a-zA-Z0-9_]+]]: tensor<16x16xi8, #[[PACKED_LAYOUT:[a-zA-Z0-9_]+]]>
// OPTIMIZED: tt.elementwise_inline_asm {{.*}}packed_element = 4{{.*}} %[[PACKED_INPUT]] : tensor<16x16xi8, #[[PACKED_LAYOUT]]> -> tensor<16x16xi8, #[[PACKED_LAYOUT]]>, tensor<16x16xi8, #[[PACKED_LAYOUT]]>
// OPTIMIZED: tt.return

#source = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#target = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @packed_inline_asm_register_grouping(%source: tensor<16x16xi8, #source>, %target: tensor<16x16xi8, #target>) -> (tensor<16x16xi8, #target>, tensor<16x16xi8, #target>) {
    %packed:2 = tt.elementwise_inline_asm "prmt.b32 $0, $2, 0, 0x5140; prmt.b32 $1, $2, 0, 0x7362;" {constraints = "=r,=r,r", packed_element = 4 : i32, pure = true} %source : tensor<16x16xi8, #source> -> tensor<16x16xi8, #source>, tensor<16x16xi8, #source>
    %left = ttg.convert_layout %packed#0 : tensor<16x16xi8, #source> -> tensor<16x16xi8, #target>
    %right = ttg.convert_layout %packed#1 : tensor<16x16xi8, #source> -> tensor<16x16xi8, #target>
    %left_result = arith.xori %left, %target : tensor<16x16xi8, #target>
    %right_result = arith.addi %right, %target : tensor<16x16xi8, #target>
    tt.return %left_result, %right_result : tensor<16x16xi8, #target>, tensor<16x16xi8, #target>
  }
}

// -----

// MX quantization uses one reduction to produce both a compact scale and the
// quantized tensor. Keep the reduction shared, place the required scale
// conversion after narrowing, and still optimize the independent fanout.
//
// BASELINE-LABEL: @shared_quantization_reduction_independent_fanout
// BASELINE: "tt.reduce"
// BASELINE: tt.store
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @shared_quantization_reduction_independent_fanout
// OPTIMIZED-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<32x128xf32
// OPTIMIZED-SAME: %[[SCALE_PTR:[a-zA-Z0-9_]+]]: tensor<32x1x4x!tt.ptr<i8>
// OPTIMIZED-SAME: %[[OUTPUT_PTR:[a-zA-Z0-9_]+]]: tensor<32x128x!tt.ptr<f32>
// OPTIMIZED-SAME: %[[TARGET:[a-zA-Z0-9_]+]]: tensor<16x16xf32
// OPTIMIZED-SAME: %[[SOURCE:[a-zA-Z0-9_]+]]: tensor<16x16xf32
// OPTIMIZED-COUNT-1: %[[MAXIMUM:[a-zA-Z0-9_]+]] = "tt.reduce"(%[[NETWORK:[a-zA-Z0-9_]+]])
// OPTIMIZED: tt.reduce.return
// OPTIMIZED-NEXT: }) :
// OPTIMIZED-NEXT: %[[NARROW_SCALE:[a-zA-Z0-9_]+]] = arith.fptoui %[[MAXIMUM]]
// OPTIMIZED-NEXT: %[[STORED_SCALE:[a-zA-Z0-9_]+]] = ttg.convert_layout %[[NARROW_SCALE]]
// OPTIMIZED-NEXT: tt.store %[[SCALE_PTR]], %[[STORED_SCALE]]
// OPTIMIZED-NEXT: %[[EXPANDED:[a-zA-Z0-9_]+]] = tt.expand_dims %[[MAXIMUM]]
// OPTIMIZED-NEXT: %[[BROADCAST:[a-zA-Z0-9_]+]] = tt.broadcast %[[EXPANDED]]
// OPTIMIZED-NEXT: %[[PRODUCT:[a-zA-Z0-9_]+]] = arith.mulf %[[NETWORK]], %[[BROADCAST]]
// OPTIMIZED-NEXT: %[[QUANTIZED:[a-zA-Z0-9_]+]] = tt.reshape %[[PRODUCT]]
// OPTIMIZED-NEXT: tt.store %[[OUTPUT_PTR]], %[[QUANTIZED]]
// OPTIMIZED-NEXT: %[[FANOUT:[a-zA-Z0-9_]+]] = ttg.convert_layout %[[SOURCE]]
// OPTIMIZED-NEXT: %[[FIRST:[a-zA-Z0-9_]+]] = arith.addf %[[FANOUT]], %[[TARGET]]
// OPTIMIZED-NEXT: %[[SECOND:[a-zA-Z0-9_]+]] = arith.mulf %[[FANOUT]], %[[TARGET]]
// OPTIMIZED-NEXT: %[[THIRD:[a-zA-Z0-9_]+]] = arith.subf %[[FANOUT]], %[[TARGET]]
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-NOT: "tt.reduce"
// OPTIMIZED: tt.return %[[FIRST]], %[[SECOND]], %[[THIRD]]

#mx_data = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mx_network = #ttg.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [4, 1, 4, 2], warpsPerCTA = [8, 1, 1, 1], order = [3, 2, 1, 0]}>
#mx_reduced = #ttg.slice<{dim = 3, parent = #mx_network}>
#mx_scale_parent = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [8, 1, 4, 1], warpsPerCTA = [8, 1, 1, 1], order = [3, 2, 1, 0]}>
#mx_scale = #ttg.slice<{dim = 3, parent = #mx_scale_parent}>
#fanout_source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#fanout_target = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @shared_quantization_reduction_independent_fanout(%input: tensor<32x128xf32, #mx_data>, %scale_ptr: tensor<32x1x4x!tt.ptr<i8>, #mx_scale>, %output_ptr: tensor<32x128x!tt.ptr<f32>, #mx_data>, %target: tensor<16x16xf32, #fanout_target>, %source: tensor<16x16xf32, #fanout_source>) -> (tensor<16x16xf32, #fanout_target>, tensor<16x16xf32, #fanout_target>, tensor<16x16xf32, #fanout_target>) {
    %network = tt.reshape %input : tensor<32x128xf32, #mx_data> -> tensor<32x1x4x32xf32, #mx_network>
    %maximum = "tt.reduce"(%network) <{axis = 3 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %larger = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %larger : f32
    }) : (tensor<32x1x4x32xf32, #mx_network>) -> tensor<32x1x4xf32, #mx_reduced>

    %narrow_scale = arith.fptoui %maximum : tensor<32x1x4xf32, #mx_reduced> to tensor<32x1x4xi8, #mx_reduced>
    %stored_scale = ttg.convert_layout %narrow_scale : tensor<32x1x4xi8, #mx_reduced> -> tensor<32x1x4xi8, #mx_scale>
    tt.store %scale_ptr, %stored_scale : tensor<32x1x4x!tt.ptr<i8>, #mx_scale>

    %expanded = tt.expand_dims %maximum {axis = 3 : i32} : tensor<32x1x4xf32, #mx_reduced> -> tensor<32x1x4x1xf32, #mx_network>
    %broadcast = tt.broadcast %expanded : tensor<32x1x4x1xf32, #mx_network> -> tensor<32x1x4x32xf32, #mx_network>
    %product = arith.mulf %network, %broadcast : tensor<32x1x4x32xf32, #mx_network>
    %quantized = tt.reshape %product : tensor<32x1x4x32xf32, #mx_network> -> tensor<32x128xf32, #mx_data>
    tt.store %output_ptr, %quantized : tensor<32x128x!tt.ptr<f32>, #mx_data>

    %converted = ttg.convert_layout %source : tensor<16x16xf32, #fanout_source> -> tensor<16x16xf32, #fanout_target>
    %first = arith.addf %converted, %target : tensor<16x16xf32, #fanout_target>
    %second = arith.mulf %converted, %target : tensor<16x16xf32, #fanout_target>
    %third = arith.subf %converted, %target : tensor<16x16xf32, #fanout_target>
    tt.return %first, %second, %third : tensor<16x16xf32, #fanout_target>, tensor<16x16xf32, #fanout_target>, tensor<16x16xf32, #fanout_target>
  }
}

// -----

// This is the exact coalesced pre-layout IR of the original production
// _reduce_forward_default eight-load kernel, not a synthetic approximation.
// The published global optimizer leaves 18 conversions and the legacy pass
// leaves five. Globally assign the shared 64-bit pointer, mask, and value
// components once so that all eight coalesced loads and the final store need
// no conversions. The same contract must hold when the pass is run twice.
// Original TTIR SHA-256:
// 02c654bc7da0de9f741ffae91d8ddce47e118ff0aceb4c444a1909ec77d455c4
//
// BASELINE-LABEL: tt.func private @"triton_kernels.reduce._reduce_forward_inner
// BASELINE-COUNT-4: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE-COUNT-8: tt.load
// BASELINE: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: tt.func private @"triton_kernels.reduce._reduce_forward_inner
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_reduce_forward(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i64 {tt.divisibility = 16 : i32}, %arg2: i64, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: i64, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %0 = arith.addi %arg11, %c31_i32 : i32
    %1 = arith.divsi %0, %c32_i32 : i32
    %2 = tt.get_program_id x : i32
    %3 = arith.divsi %2, %1 : i32
    %4 = arith.remsi %2, %1 : i32
    tt.call @"triton_kernels.reduce._reduce_forward_inner__i32_i32_triton_kernels.reduce.ReduceForwardCommonArgs<Pfp32, i64, i64, c1, cNone, cNone, cNone, cNone, Pfp32, i64, c1, cNone, cNone, cNone, cNone, i32, i32, i32, cNone, i32, i32, i32, c8, i32, i32, i32, c8, cNone, TT, cNone, TT, cNone, TT, cNone, cNone, cNone, cNone, cNone, cTrue, cNone, cNone, cTrue, cTrue, cTrue, cTrue, cTrue, cTrue, cTrue, c128, c128, c1, cNone, c1>_c32_cTrue_c8"(%4, %3, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13) : (i32, i32, !tt.ptr<f32>, i64, i64, !tt.ptr<f32>, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    tt.return
  }
  tt.func private @"triton_kernels.reduce._reduce_forward_inner__i32_i32_triton_kernels.reduce.ReduceForwardCommonArgs<Pfp32, i64, i64, c1, cNone, cNone, cNone, cNone, Pfp32, i64, c1, cNone, cNone, cNone, cNone, i32, i32, i32, cNone, i32, i32, i32, c8, i32, i32, i32, c8, cNone, TT, cNone, TT, cNone, TT, cNone, cNone, cNone, cNone, cNone, cTrue, cNone, cNone, cTrue, cTrue, cTrue, cTrue, cTrue, cTrue, cTrue, c128, c128, c1, cNone, c1>_c32_cTrue_c8"(%arg0: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg1: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg2: !tt.ptr<f32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg3: i64 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg4: i64 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg5: !tt.ptr<f32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg6: i64 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg7: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg8: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg9: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg10: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg11: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg12: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg13: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg14: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg15: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}) attributes {noinline = true} {
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.muli %arg0, %c32_i32 : i32
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %2 = tt.splat %0 : i32 -> tensor<32xi32, #blocked1>
    %3 = arith.addi %2, %1 : tensor<32xi32, #blocked1>
    %4 = tt.splat %arg13 : i32 -> tensor<32xi32, #blocked1>
    %5 = arith.cmpi slt, %3, %4 : tensor<32xi32, #blocked1>
    %6 = arith.muli %arg1, %c128_i32 : i32
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %8 = tt.splat %6 : i32 -> tensor<128xi32, #blocked1>
    %9 = arith.addi %8, %7 : tensor<128xi32, #blocked1>
    %10 = tt.splat %arg14 : i32 -> tensor<128xi32, #blocked1>
    %11 = arith.cmpi slt, %9, %10 : tensor<128xi32, #blocked1>
    %12 = ttg.convert_layout %5 : tensor<32xi1, #blocked1> -> tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi1, #blocked2>
    %14 = ttg.convert_layout %13 : tensor<32x1xi1, #blocked2> -> tensor<32x1xi1, #blocked3>
    %15 = ttg.convert_layout %11 : tensor<128xi1, #blocked1> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi1, #blocked4>
    %17 = ttg.convert_layout %16 : tensor<1x128xi1, #blocked4> -> tensor<1x128xi1, #blocked>
    %18 = tt.broadcast %14 : tensor<32x1xi1, #blocked3> -> tensor<32x128xi1, #blocked3>
    %19 = ttg.convert_layout %18 : tensor<32x128xi1, #blocked3> -> tensor<32x128xi1, #blocked>
    %20 = tt.broadcast %17 : tensor<1x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %21 = arith.andi %19, %20 : tensor<32x128xi1, #blocked>
    %22 = ttg.convert_layout %3 : tensor<32xi32, #blocked1> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %23 = tt.expand_dims %22 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %24 = ttg.convert_layout %23 : tensor<32x1xi32, #blocked2> -> tensor<32x1xi32, #blocked3>
    %25 = arith.extsi %24 : tensor<32x1xi32, #blocked3> to tensor<32x1xi64, #blocked3>
    %26 = tt.splat %arg4 : i64 -> tensor<32x1xi64, #blocked3>
    %27 = arith.muli %25, %26 : tensor<32x1xi64, #blocked3>
    %28 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %29 = tt.addptr %28, %27 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %30 = ttg.convert_layout %9 : tensor<128xi32, #blocked1> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %31 = tt.expand_dims %30 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi32, #blocked4>
    %32 = ttg.convert_layout %31 : tensor<1x128xi32, #blocked4> -> tensor<1x128xi32, #blocked>
    %33 = tt.broadcast %29 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %34 = ttg.convert_layout %33 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %35 = tt.broadcast %32 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %36 = tt.addptr %34, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %37 = ttg.convert_layout %36 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %38 = ttg.convert_layout %21 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %39 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %40 = tt.load %37, %38, %39 : tensor<32x128x!tt.ptr<f32>, #blocked>
    %41 = tt.addptr %arg2, %arg3 : !tt.ptr<f32>, i64
    %42 = tt.splat %41 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %43 = tt.addptr %42, %27 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %44 = tt.broadcast %43 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %45 = ttg.convert_layout %44 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %46 = tt.addptr %45, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %47 = ttg.convert_layout %46 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %48 = ttg.convert_layout %21 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %49 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %50 = tt.load %47, %48, %49 : tensor<32x128x!tt.ptr<f32>, #blocked>
    %51 = arith.addf %40, %50 : tensor<32x128xf32, #blocked>
    %52 = arith.muli %arg3, %c2_i64 : i64
    %53 = tt.addptr %arg2, %52 : !tt.ptr<f32>, i64
    %54 = tt.splat %53 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %55 = tt.addptr %54, %27 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %56 = tt.broadcast %55 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %57 = ttg.convert_layout %56 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %58 = tt.addptr %57, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %59 = ttg.convert_layout %58 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %60 = ttg.convert_layout %21 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %61 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %62 = tt.load %59, %60, %61 : tensor<32x128x!tt.ptr<f32>, #blocked>
    %63 = arith.addf %51, %62 : tensor<32x128xf32, #blocked>
    %64 = arith.muli %arg3, %c3_i64 : i64
    %65 = tt.addptr %arg2, %64 : !tt.ptr<f32>, i64
    %66 = tt.splat %65 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %67 = tt.addptr %66, %27 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %68 = tt.broadcast %67 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %69 = ttg.convert_layout %68 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %70 = tt.addptr %69, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %71 = ttg.convert_layout %70 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %72 = ttg.convert_layout %21 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %73 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %74 = tt.load %71, %72, %73 : tensor<32x128x!tt.ptr<f32>, #blocked>
    %75 = arith.addf %63, %74 : tensor<32x128xf32, #blocked>
    %76 = arith.muli %arg3, %c4_i64 : i64
    %77 = tt.addptr %arg2, %76 : !tt.ptr<f32>, i64
    %78 = tt.splat %77 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %79 = tt.addptr %78, %27 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %80 = tt.broadcast %79 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %81 = ttg.convert_layout %80 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %82 = tt.addptr %81, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %83 = ttg.convert_layout %82 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %84 = ttg.convert_layout %21 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %85 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %86 = tt.load %83, %84, %85 : tensor<32x128x!tt.ptr<f32>, #blocked>
    %87 = arith.addf %75, %86 : tensor<32x128xf32, #blocked>
    %88 = arith.muli %arg3, %c5_i64 : i64
    %89 = tt.addptr %arg2, %88 : !tt.ptr<f32>, i64
    %90 = tt.splat %89 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %91 = tt.addptr %90, %27 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %92 = tt.broadcast %91 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %93 = ttg.convert_layout %92 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %94 = tt.addptr %93, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %95 = ttg.convert_layout %94 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %96 = ttg.convert_layout %21 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %97 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %98 = tt.load %95, %96, %97 : tensor<32x128x!tt.ptr<f32>, #blocked>
    %99 = arith.addf %87, %98 : tensor<32x128xf32, #blocked>
    %100 = arith.muli %arg3, %c6_i64 : i64
    %101 = tt.addptr %arg2, %100 : !tt.ptr<f32>, i64
    %102 = tt.splat %101 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %103 = tt.addptr %102, %27 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %104 = tt.broadcast %103 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %105 = ttg.convert_layout %104 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %106 = tt.addptr %105, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %107 = ttg.convert_layout %106 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %108 = ttg.convert_layout %21 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %109 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %110 = tt.load %107, %108, %109 : tensor<32x128x!tt.ptr<f32>, #blocked>
    %111 = arith.addf %99, %110 : tensor<32x128xf32, #blocked>
    %112 = arith.muli %arg3, %c7_i64 : i64
    %113 = tt.addptr %arg2, %112 : !tt.ptr<f32>, i64
    %114 = tt.splat %113 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %115 = tt.addptr %114, %27 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %116 = tt.broadcast %115 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %117 = ttg.convert_layout %116 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %118 = tt.addptr %117, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %119 = ttg.convert_layout %118 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %120 = ttg.convert_layout %21 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %121 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %122 = tt.load %119, %120, %121 : tensor<32x128x!tt.ptr<f32>, #blocked>
    %123 = arith.addf %111, %122 : tensor<32x128xf32, #blocked>
    %124 = tt.splat %arg15 : i32 -> tensor<128xi32, #blocked1>
    %125 = arith.cmpi slt, %9, %124 : tensor<128xi32, #blocked1>
    %126 = tt.splat %arg6 : i64 -> tensor<32x1xi64, #blocked3>
    %127 = arith.muli %25, %126 : tensor<32x1xi64, #blocked3>
    %128 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
    %129 = tt.addptr %128, %127 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
    %130 = tt.broadcast %129 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %131 = ttg.convert_layout %130 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %132 = tt.addptr %131, %35 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
    %133 = ttg.convert_layout %125 : tensor<128xi1, #blocked1> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %134 = tt.expand_dims %133 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi1, #blocked4>
    %135 = ttg.convert_layout %134 : tensor<1x128xi1, #blocked4> -> tensor<1x128xi1, #blocked>
    %136 = tt.broadcast %135 : tensor<1x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    %137 = arith.andi %19, %136 : tensor<32x128xi1, #blocked>
    %138 = ttg.convert_layout %132 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
    %139 = ttg.convert_layout %123 : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
    %140 = ttg.convert_layout %137 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
    tt.store %138, %139, %140 : tensor<32x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// The exact first-layout input from KI expert gather/transposition. Keep the
// complete masked loop-store address together instead of introducing four
// independent row/column layout conversions. Original TTIR SHA-256:
// a42eb55f6510689306adbfa56521f29849f7fd000d3b12ddbfe86d80c8feaa06.
//
// BASELINE-LABEL: @production_masked_gather_store_layout
// BASELINE: scf.for
// BASELINE-COUNT-5: ttg.convert_layout
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_masked_gather_store_layout
// OPTIMIZED: scf.for
// OPTIMIZED-COUNT-1: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @production_masked_gather_store_layout(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg14: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c0_i32 = arith.constant 0 : i32
    %c24_i32 = arith.constant 24 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %cst_0 = arith.constant dense<0> : tensor<32xi32, #blocked1>
    %c32_i64 = arith.constant 32 : i64
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg9, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.minsi %2, %c24_i32 : i32
    %4 = arith.remsi %0, %arg10 : i32
    %5 = arith.divsi %0, %arg10 : i32
    %6 = tt.addptr %arg13, %c4_i32 : !tt.ptr<i32>, i32
    %7 = tt.load %6 : !tt.ptr<i32>
    %8 = arith.cmpi sge, %4, %7 : i32
    cf.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %9 = tt.addptr %arg14, %4 : !tt.ptr<i32>, i32
    %10 = tt.load %9 : !tt.ptr<i32>
    %11 = arith.andi %10, %c65535_i32 : i32
    %12 = arith.shrsi %10, %c16_i32 : i32
    %13 = tt.addptr %arg12, %11 : !tt.ptr<i32>, i32
    %14 = tt.load %13 : !tt.ptr<i32>
    %15 = arith.muli %12, %c32_i32 : i32
    %16 = arith.addi %14, %15 : i32
    %17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %18 = tt.splat %16 : i32 -> tensor<32xi32, #blocked1>
    %19 = arith.addi %18, %17 : tensor<32xi32, #blocked1>
    %20 = tt.addptr %13, %c1_i32 : !tt.ptr<i32>, i32
    %21 = tt.load %20 : !tt.ptr<i32>
    %22 = tt.splat %21 : i32 -> tensor<32xi32, #blocked1>
    %23 = arith.cmpi slt, %19, %22 : tensor<32xi32, #blocked1>
    %24 = tt.addptr %arg13, %11 : !tt.ptr<i32>, i32
    %25 = tt.load %24 : !tt.ptr<i32>
    %26 = arith.extsi %25 : i32 to i64
    %27 = arith.muli %26, %c32_i64 : i64
    %28 = arith.extsi %15 : i32 to i64
    %29 = arith.addi %27, %28 : i64
    %30 = arith.extsi %17 : tensor<32xi32, #blocked1> to tensor<32xi64, #blocked1>
    %31 = tt.splat %29 : i64 -> tensor<32xi64, #blocked1>
    %32 = arith.addi %31, %30 : tensor<32xi64, #blocked1>
    %33 = tt.splat %arg6 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked1>
    %34 = tt.addptr %33, %19 : tensor<32x!tt.ptr<i32>, #blocked1>, tensor<32xi32, #blocked1>
    %35 = ttg.convert_layout %34 : tensor<32x!tt.ptr<i32>, #blocked1> -> tensor<32x!tt.ptr<i32>, #blocked1>
    %36 = ttg.convert_layout %23 : tensor<32xi1, #blocked1> -> tensor<32xi1, #blocked1>
    %37 = ttg.convert_layout %cst_0 : tensor<32xi32, #blocked1> -> tensor<32xi32, #blocked1>
    %38 = tt.load %35, %36, %37 : tensor<32x!tt.ptr<i32>, #blocked1>
    %39 = arith.extsi %38 : tensor<32xi32, #blocked1> to tensor<32xi64, #blocked1>
    scf.for %arg21 = %c0_i32 to %3 step %c1_i32  : i32 {
      %40 = arith.muli %arg21, %c128_i32 : i32
      %41 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
      %42 = tt.splat %40 : i32 -> tensor<128xi32, #blocked1>
      %43 = arith.addi %42, %41 : tensor<128xi32, #blocked1>
      %44 = arith.extsi %43 : tensor<128xi32, #blocked1> to tensor<128xi64, #blocked1>
      %45 = arith.extsi %arg9 : i32 to i64
      %46 = tt.splat %45 : i64 -> tensor<128xi64, #blocked1>
      %47 = arith.cmpi slt, %44, %46 : tensor<128xi64, #blocked1>
      %48 = ttg.convert_layout %23 : tensor<32xi1, #blocked1> -> tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %49 = tt.expand_dims %48 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi1, #blocked2>
      %50 = ttg.convert_layout %49 : tensor<32x1xi1, #blocked2> -> tensor<32x1xi1, #blocked3>
      %51 = ttg.convert_layout %47 : tensor<128xi1, #blocked1> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi1, #blocked4>
      %53 = ttg.convert_layout %52 : tensor<1x128xi1, #blocked4> -> tensor<1x128xi1, #blocked>
      %54 = tt.broadcast %50 : tensor<32x1xi1, #blocked3> -> tensor<32x128xi1, #blocked3>
      %55 = ttg.convert_layout %54 : tensor<32x128xi1, #blocked3> -> tensor<32x128xi1, #blocked>
      %56 = tt.broadcast %53 : tensor<1x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
      %57 = arith.andi %55, %56 : tensor<32x128xi1, #blocked>
      %58 = arith.muli %5, %arg4 : i32
      %59 = arith.extsi %arg5 : i32 to i64
      %60 = tt.splat %59 : i64 -> tensor<32xi64, #blocked1>
      %61 = arith.muli %39, %60 : tensor<32xi64, #blocked1>
      %62 = arith.extsi %58 : i32 to i64
      %63 = tt.splat %62 : i64 -> tensor<32xi64, #blocked1>
      %64 = arith.addi %63, %61 : tensor<32xi64, #blocked1>
      %65 = ttg.convert_layout %64 : tensor<32xi64, #blocked1> -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %66 = tt.expand_dims %65 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi64, #blocked2>
      %67 = ttg.convert_layout %66 : tensor<32x1xi64, #blocked2> -> tensor<32x1xi64, #blocked3>
      %68 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
      %69 = tt.addptr %68, %67 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
      %70 = ttg.convert_layout %44 : tensor<128xi64, #blocked1> -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %71 = tt.expand_dims %70 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi64, #blocked4>
      %72 = ttg.convert_layout %71 : tensor<1x128xi64, #blocked4> -> tensor<1x128xi64, #blocked>
      %73 = tt.broadcast %69 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
      %74 = ttg.convert_layout %73 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
      %75 = tt.broadcast %72 : tensor<1x128xi64, #blocked> -> tensor<32x128xi64, #blocked>
      %76 = tt.addptr %74, %75 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi64, #blocked>
      %77 = ttg.convert_layout %76 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked>
      %78 = ttg.convert_layout %57 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
      %79 = ttg.convert_layout %cst : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked>
      %80 = tt.load %77, %78, %79 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %81 = arith.muli %5, %arg1 : i32
      %82 = arith.extsi %81 : i32 to i64
      %83 = tt.splat %82 : i64 -> tensor<32xi64, #blocked1>
      %84 = arith.addi %83, %32 : tensor<32xi64, #blocked1>
      %85 = ttg.convert_layout %84 : tensor<32xi64, #blocked1> -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %86 = tt.expand_dims %85 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi64, #blocked2>
      %87 = ttg.convert_layout %86 : tensor<32x1xi64, #blocked2> -> tensor<32x1xi64, #blocked3>
      %88 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked3>
      %89 = tt.addptr %88, %87 : tensor<32x1x!tt.ptr<f32>, #blocked3>, tensor<32x1xi64, #blocked3>
      %90 = arith.extsi %arg2 : i32 to i64
      %91 = tt.splat %90 : i64 -> tensor<128xi64, #blocked1>
      %92 = arith.muli %44, %91 : tensor<128xi64, #blocked1>
      %93 = ttg.convert_layout %92 : tensor<128xi64, #blocked1> -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %94 = tt.expand_dims %93 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi64, #blocked4>
      %95 = ttg.convert_layout %94 : tensor<1x128xi64, #blocked4> -> tensor<1x128xi64, #blocked>
      %96 = tt.broadcast %89 : tensor<32x1x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
      %97 = ttg.convert_layout %96 : tensor<32x128x!tt.ptr<f32>, #blocked3> -> tensor<32x128x!tt.ptr<f32>, #blocked>
      %98 = tt.broadcast %95 : tensor<1x128xi64, #blocked> -> tensor<32x128xi64, #blocked>
      %99 = tt.addptr %97, %98 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi64, #blocked>
      %100 = ttg.convert_layout %99 : tensor<32x128x!tt.ptr<f32>, #blocked> -> tensor<32x128x!tt.ptr<f32>, #blocked5>
      %101 = ttg.convert_layout %80 : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked5>
      %102 = ttg.convert_layout %56 : tensor<32x128xi1, #blocked> -> tensor<32x128xi1, #blocked5>
      tt.store %100, %101, %102 : tensor<32x128x!tt.ptr<f32>, #blocked5>
    }
    tt.return
  }
}

// -----

// The exact first-layout input from KI MX upcast. Preserve the hardware
// descriptor load and its scale/output component in the incumbent layout.
// Original TTIR SHA-256:
// 2b7c5682341b8564c2a7bb6810e0c87dad05dfcbbbcf489eee2571b5df6256a8.
//
// BASELINE-LABEL: @production_mx_descriptor_upcast_layout
// BASELINE: tt.descriptor_load
// BASELINE-COUNT-2: ttg.convert_layout
// BASELINE: tt.descriptor_store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_mx_descriptor_upcast_layout
// OPTIMIZED: tt.descriptor_load
// OPTIMIZED-COUNT-2: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.descriptor_store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 8, 4, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [0, 1, 2]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked12 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked13 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @production_mx_descriptor_upcast_layout(%arg0: !tt.tensordesc<1x64x128xbf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: !tt.tensordesc<1x64x128xf8E4M3FN>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32) attributes {noinline = false} {
    %cst = arith.constant dense<7> : tensor<1x64x4xi16, #blocked>
    %cst_0 = arith.constant dense<3.38953139E+38> : tensor<1x64x4x32xf32, #blocked1>
    %cst_1 = arith.constant dense<255> : tensor<1x64x4x1xi32, #blocked2>
    %cst_2 = arith.constant dense<0x7FC00000> : tensor<1x64x4x32xf32, #blocked1>
    %cst_3 = arith.constant dense<-3.38953139E+38> : tensor<1x64x4x32xf32, #blocked1>
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4_i64 = arith.constant 4 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_num_programs x : i32
    %3 = arith.cmpi ne, %arg27, %2 : i32
    %4:3 = scf.if %3 -> (i64, i64, i64) {
      %89 = arith.muli %arg26, %arg27 : i32
      %90 = arith.extsi %89 : i32 to i64
      %91 = arith.divsi %1, %90 : i64
      %92 = arith.remsi %1, %90 : i64
      %93 = arith.extsi %arg27 : i32 to i64
      %94 = arith.divsi %92, %93 : i64
      %95 = arith.remsi %1, %93 : i64
      scf.yield %95, %91, %94 : i64, i64, i64
    } else {
      %89 = tt.get_program_id y : i32
      %90 = arith.extsi %89 : i32 to i64
      %91 = tt.get_program_id z : i32
      %92 = arith.extsi %91 : i32 to i64
      scf.yield %1, %92, %90 : i64, i64, i64
    }
    %5 = arith.muli %4#2, %c64_i64 : i64
    %6 = arith.muli %4#0, %c128_i64 : i64
    %7 = arith.muli %4#0, %c4_i64 : i64
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %9 = ttg.convert_layout %8 : tensor<64xi32, #blocked3> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi32, #blocked4>
    %11 = ttg.convert_layout %10 : tensor<1x64xi32, #blocked4> -> tensor<1x64xi32, #blocked5>
    %12 = ttg.convert_layout %11 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked6}>>
    %13 = tt.expand_dims %12 {axis = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked6}>> -> tensor<1x64x1xi32, #blocked6>
    %14 = ttg.convert_layout %13 : tensor<1x64x1xi32, #blocked6> -> tensor<1x64x1xi32, #blocked7>
    %15 = arith.extsi %14 : tensor<1x64x1xi32, #blocked7> to tensor<1x64x1xi64, #blocked7>
    %16 = arith.extsi %arg21 : i32 to i64
    %17 = arith.cmpi slt, %4#1, %16 : i64
    %18 = tt.splat %5 : i64 -> tensor<1x64x1xi64, #blocked7>
    %19 = arith.addi %18, %15 : tensor<1x64x1xi64, #blocked7>
    %20 = arith.extsi %arg22 : i32 to i64
    %21 = tt.splat %20 : i64 -> tensor<1x64x1xi64, #blocked7>
    %22 = arith.cmpi slt, %19, %21 : tensor<1x64x1xi64, #blocked7>
    %23 = arith.trunci %4#1 : i64 to i32
    %24 = arith.trunci %5 : i64 to i32
    %25 = arith.trunci %6 : i64 to i32
    %26 = tt.descriptor_load %arg9[%23, %24, %25] : !tt.tensordesc<1x64x128xf8E4M3FN> -> tensor<1x64x128xf8E4M3FN, #blocked8>
    %27 = ttg.convert_layout %26 : tensor<1x64x128xf8E4M3FN, #blocked8> -> tensor<1x64x128xf8E4M3FN, #blocked9>
    %28 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked3>
    %29 = ttg.convert_layout %28 : tensor<4xi32, #blocked3> -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x4xi32, #blocked4>
    %31 = ttg.convert_layout %30 : tensor<1x4xi32, #blocked4> -> tensor<1x4xi32, #blocked10>
    %32 = ttg.convert_layout %31 : tensor<1x4xi32, #blocked10> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked11}>>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked11}>> -> tensor<1x1x4xi32, #blocked11>
    %34 = ttg.convert_layout %33 : tensor<1x1x4xi32, #blocked11> -> tensor<1x1x4xi32, #blocked12>
    %35 = arith.extsi %34 : tensor<1x1x4xi32, #blocked12> to tensor<1x1x4xi64, #blocked12>
    %36 = arith.extsi %arg19 : i32 to i64
    %37 = arith.muli %4#1, %36 : i64
    %38 = arith.extsi %arg20 : i32 to i64
    %39 = arith.muli %5, %38 : i64
    %40 = arith.addi %39, %7 : i64
    %41 = arith.addi %37, %40 : i64
    %42 = tt.addptr %arg18, %41 : !tt.ptr<i8>, i64
    %43 = tt.splat %7 : i64 -> tensor<1x1x4xi64, #blocked12>
    %44 = arith.addi %43, %35 : tensor<1x1x4xi64, #blocked12>
    %45 = arith.addi %arg23, %c31_i32 : i32
    %46 = arith.divsi %45, %c32_i32 : i32
    %47 = arith.extsi %46 : i32 to i64
    %48 = tt.splat %47 : i64 -> tensor<1x1x4xi64, #blocked12>
    %49 = arith.cmpi slt, %44, %48 : tensor<1x1x4xi64, #blocked12>
    %50 = tt.splat %17 : i1 -> tensor<1x64x1xi1, #blocked7>
    %51 = arith.andi %50, %22 : tensor<1x64x1xi1, #blocked7>
    %52 = tt.broadcast %51 : tensor<1x64x1xi1, #blocked7> -> tensor<1x64x4xi1, #blocked7>
    %53 = ttg.convert_layout %52 : tensor<1x64x4xi1, #blocked7> -> tensor<1x64x4xi1, #blocked>
    %54 = tt.broadcast %49 : tensor<1x1x4xi1, #blocked12> -> tensor<1x64x4xi1, #blocked12>
    %55 = ttg.convert_layout %54 : tensor<1x64x4xi1, #blocked12> -> tensor<1x64x4xi1, #blocked>
    %56 = arith.andi %53, %55 : tensor<1x64x4xi1, #blocked>
    %57 = tt.splat %38 : i64 -> tensor<1x64x1xi64, #blocked7>
    %58 = arith.muli %15, %57 : tensor<1x64x1xi64, #blocked7>
    %59 = tt.broadcast %58 : tensor<1x64x1xi64, #blocked7> -> tensor<1x64x4xi64, #blocked7>
    %60 = ttg.convert_layout %59 : tensor<1x64x4xi64, #blocked7> -> tensor<1x64x4xi64, #blocked>
    %61 = tt.broadcast %35 : tensor<1x1x4xi64, #blocked12> -> tensor<1x64x4xi64, #blocked12>
    %62 = ttg.convert_layout %61 : tensor<1x64x4xi64, #blocked12> -> tensor<1x64x4xi64, #blocked>
    %63 = arith.addi %60, %62 : tensor<1x64x4xi64, #blocked>
    %64 = tt.splat %42 : !tt.ptr<i8> -> tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %65 = tt.addptr %64, %63 : tensor<1x64x4x!tt.ptr<i8>, #blocked>, tensor<1x64x4xi64, #blocked>
    %66 = ttg.convert_layout %65 : tensor<1x64x4x!tt.ptr<i8>, #blocked> -> tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %67 = ttg.convert_layout %56 : tensor<1x64x4xi1, #blocked> -> tensor<1x64x4xi1, #blocked>
    %68 = tt.load %66, %67 : tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %69 = arith.extui %68 : tensor<1x64x4xi8, #blocked> to tensor<1x64x4xi16, #blocked>
    %70 = arith.shli %69, %cst : tensor<1x64x4xi16, #blocked>
    %71 = tt.bitcast %70 : tensor<1x64x4xi16, #blocked> -> tensor<1x64x4xbf16, #blocked>
    %72 = tt.fp_to_fp %27 : tensor<1x64x128xf8E4M3FN, #blocked9> -> tensor<1x64x128xbf16, #blocked9>
    %73 = tt.reshape %72 : tensor<1x64x128xbf16, #blocked9> -> tensor<1x64x4x32xbf16, #blocked1>
    %74 = tt.reshape %71 : tensor<1x64x4xbf16, #blocked> -> tensor<1x64x4x1xbf16, #blocked2>
    %75 = tt.reshape %68 : tensor<1x64x4xi8, #blocked> -> tensor<1x64x4x1xi8, #blocked2>
    %76 = tt.broadcast %74 : tensor<1x64x4x1xbf16, #blocked2> -> tensor<1x64x4x32xbf16, #blocked2>
    %77 = ttg.convert_layout %76 : tensor<1x64x4x32xbf16, #blocked2> -> tensor<1x64x4x32xbf16, #blocked1>
    %78 = arith.mulf %73, %77 : tensor<1x64x4x32xbf16, #blocked1>
    %79 = arith.extf %78 : tensor<1x64x4x32xbf16, #blocked1> to tensor<1x64x4x32xf32, #blocked1>
    %80 = tt.clampf %79, %cst_3, %cst_0, propagateNan = all : tensor<1x64x4x32xf32, #blocked1>
    %81 = arith.extui %75 : tensor<1x64x4x1xi8, #blocked2> to tensor<1x64x4x1xi32, #blocked2>
    %82 = arith.cmpi eq, %81, %cst_1 : tensor<1x64x4x1xi32, #blocked2>
    %83 = tt.broadcast %82 : tensor<1x64x4x1xi1, #blocked2> -> tensor<1x64x4x32xi1, #blocked2>
    %84 = ttg.convert_layout %83 : tensor<1x64x4x32xi1, #blocked2> -> tensor<1x64x4x32xi1, #blocked1>
    %85 = arith.select %84, %cst_2, %80 : tensor<1x64x4x32xi1, #blocked1>, tensor<1x64x4x32xf32, #blocked1>
    %86 = tt.reshape %85 : tensor<1x64x4x32xf32, #blocked1> -> tensor<1x64x128xf32, #blocked9>
    %87 = arith.truncf %86 : tensor<1x64x128xf32, #blocked9> to tensor<1x64x128xbf16, #blocked9>
    %88 = ttg.convert_layout %87 : tensor<1x64x128xbf16, #blocked9> -> tensor<1x64x128xbf16, #blocked13>
    tt.descriptor_store %arg0[%23, %24, %25], %88 : !tt.tensordesc<1x64x128xbf16>, tensor<1x64x128xbf16, #blocked13>
    tt.return
  }
}


// -----

// Regressions derived from exact, unmodified production TTIR at the first

// NVIDIA layout-assignment boundary.

// Preserve a single masked loop copy without introducing row/column conversions.
// Exact original TTIR SHA-256: 4269fc93a7d5ae0a3ffcf863b846b2bebd5395a76980bc743535891caf72989e.
//
// BASELINE-LABEL: @production_single_masked_copy_layout
// BASELINE: scf.for
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.store
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_single_masked_copy_layout
// OPTIMIZED: scf.for
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @production_single_masked_copy_layout(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c24_i32 = arith.constant 24 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %arg7 : i32
    %2 = arith.remsi %0, %arg7 : i32
    %3 = arith.muli %2, %c24_i32 : i32
    %4 = arith.addi %arg6, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %3, %c24_i32 : i32
    %7 = arith.minsi %5, %6 : i32
    %8 = arith.cmpi eq, %1, %c0_i32 : i32
    cf.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %9 = arith.subi %1, %c1_i32 : i32
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %11 = arith.extsi %10 : tensor<128xi32, #blocked1> to tensor<128xi64, #blocked1>
    %12 = tt.splat %arg5 : i32 -> tensor<128xi32, #blocked1>
    %13 = arith.cmpi slt, %10, %12 : tensor<128xi32, #blocked1>
    scf.for %arg14 = %3 to %7 step %c1_i32  : i32 {
      %14 = arith.muli %arg14, %c128_i32 : i32
      %15 = tt.splat %14 : i32 -> tensor<128xi32, #blocked1>
      %16 = arith.addi %15, %10 : tensor<128xi32, #blocked1>
      %17 = arith.extsi %16 : tensor<128xi32, #blocked1> to tensor<128xi64, #blocked1>
      %18 = ttg.convert_layout %13 : tensor<128xi1, #blocked1> -> tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi1, #blocked2>
      %20 = ttg.convert_layout %19 : tensor<128x1xi1, #blocked2> -> tensor<128x1xi1, #blocked3>
      %21 = tt.broadcast %20 : tensor<128x1xi1, #blocked3> -> tensor<128x128xi1, #blocked3>
      %22 = ttg.convert_layout %21 : tensor<128x128xi1, #blocked3> -> tensor<128x128xi1, #blocked>
      %23 = arith.muli %9, %arg3 : i32
      %24 = arith.extsi %23 : i32 to i64
      %25 = tt.splat %24 : i64 -> tensor<128xi64, #blocked1>
      %26 = arith.addi %25, %11 : tensor<128xi64, #blocked1>
      %27 = ttg.convert_layout %26 : tensor<128xi64, #blocked1> -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %28 = tt.expand_dims %27 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi64, #blocked2>
      %29 = ttg.convert_layout %28 : tensor<128x1xi64, #blocked2> -> tensor<128x1xi64, #blocked3>
      %30 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked3>
      %31 = tt.addptr %30, %29 : tensor<128x1x!tt.ptr<f16>, #blocked3>, tensor<128x1xi64, #blocked3>
      %32 = ttg.convert_layout %17 : tensor<128xi64, #blocked1> -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi64, #blocked4>
      %34 = ttg.convert_layout %33 : tensor<1x128xi64, #blocked4> -> tensor<1x128xi64, #blocked>
      %35 = tt.broadcast %31 : tensor<128x1x!tt.ptr<f16>, #blocked3> -> tensor<128x128x!tt.ptr<f16>, #blocked3>
      %36 = ttg.convert_layout %35 : tensor<128x128x!tt.ptr<f16>, #blocked3> -> tensor<128x128x!tt.ptr<f16>, #blocked>
      %37 = tt.broadcast %34 : tensor<1x128xi64, #blocked> -> tensor<128x128xi64, #blocked>
      %38 = tt.addptr %36, %37 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi64, #blocked>
      %39 = ttg.convert_layout %38 : tensor<128x128x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
      %40 = ttg.convert_layout %22 : tensor<128x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %41 = ttg.convert_layout %cst : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
      %42 = tt.load %39, %40, %41 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %43 = arith.muli %9, %arg1 : i32
      %44 = arith.extsi %43 : i32 to i64
      %45 = tt.splat %44 : i64 -> tensor<128xi64, #blocked1>
      %46 = arith.addi %45, %11 : tensor<128xi64, #blocked1>
      %47 = ttg.convert_layout %46 : tensor<128xi64, #blocked1> -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi64, #blocked2>
      %49 = ttg.convert_layout %48 : tensor<128x1xi64, #blocked2> -> tensor<128x1xi64, #blocked3>
      %50 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked3>
      %51 = tt.addptr %50, %49 : tensor<128x1x!tt.ptr<f16>, #blocked3>, tensor<128x1xi64, #blocked3>
      %52 = tt.broadcast %51 : tensor<128x1x!tt.ptr<f16>, #blocked3> -> tensor<128x128x!tt.ptr<f16>, #blocked3>
      %53 = ttg.convert_layout %52 : tensor<128x128x!tt.ptr<f16>, #blocked3> -> tensor<128x128x!tt.ptr<f16>, #blocked>
      %54 = tt.addptr %53, %37 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi64, #blocked>
      %55 = ttg.convert_layout %54 : tensor<128x128x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
      %56 = ttg.convert_layout %42 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
      %57 = ttg.convert_layout %22 : tensor<128x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
      tt.store %55, %56, %57 : tensor<128x128x!tt.ptr<f16>, #blocked>
    }
    tt.return
  }
}

// -----

// Do not bootstrap independent copy loops: the global assignment is already zero-copy.
// Exact original TTIR SHA-256: 38892714ffa54813580ee76fbec92c6fa11f945e4846ed39fc9b1404df45ca47.
//
// BASELINE-LABEL: @production_independent_masked_copy_loops_layout
// BASELINE-COUNT-8: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_independent_masked_copy_loops_layout
// OPTIMIZED: scf.for
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return
// OPTIMIZED: scf.for
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @production_independent_masked_copy_loops_layout(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c24_i32 = arith.constant 24 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg7, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.minsi %2, %c24_i32 : i32
    %4 = tt.addptr %arg10, %c2_i32 : !tt.ptr<i32>, i32
    %5 = tt.load %4 : !tt.ptr<i32>
    %6 = arith.cmpi eq, %0, %c0_i32 : i32
    cf.cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %7 = arith.extsi %5 : i32 to i64
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %9 = arith.extsi %8 : tensor<64xi32, #blocked1> to tensor<64xi64, #blocked1>
    %10 = tt.splat %7 : i64 -> tensor<64xi64, #blocked1>
    %11 = arith.addi %10, %9 : tensor<64xi64, #blocked1>
    %12 = arith.extsi %arg6 : i32 to i64
    %13 = tt.splat %12 : i64 -> tensor<64xi64, #blocked1>
    %14 = arith.cmpi slt, %11, %13 : tensor<64xi64, #blocked1>
    scf.for %arg19 = %c0_i32 to %3 step %c1_i32  : i32 {
      %25 = arith.muli %arg19, %c128_i32 : i32
      %26 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
      %27 = tt.splat %25 : i32 -> tensor<128xi32, #blocked1>
      %28 = arith.addi %27, %26 : tensor<128xi32, #blocked1>
      %29 = arith.extsi %28 : tensor<128xi32, #blocked1> to tensor<128xi64, #blocked1>
      %30 = arith.extsi %arg7 : i32 to i64
      %31 = tt.splat %30 : i64 -> tensor<128xi64, #blocked1>
      %32 = arith.cmpi slt, %29, %31 : tensor<128xi64, #blocked1>
      %33 = ttg.convert_layout %14 : tensor<64xi1, #blocked1> -> tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %34 = tt.expand_dims %33 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi1, #blocked2>
      %35 = ttg.convert_layout %34 : tensor<64x1xi1, #blocked2> -> tensor<64x1xi1, #blocked3>
      %36 = ttg.convert_layout %32 : tensor<128xi1, #blocked1> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %37 = tt.expand_dims %36 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi1, #blocked4>
      %38 = ttg.convert_layout %37 : tensor<1x128xi1, #blocked4> -> tensor<1x128xi1, #blocked>
      %39 = tt.broadcast %35 : tensor<64x1xi1, #blocked3> -> tensor<64x128xi1, #blocked3>
      %40 = ttg.convert_layout %39 : tensor<64x128xi1, #blocked3> -> tensor<64x128xi1, #blocked>
      %41 = tt.broadcast %38 : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %42 = arith.andi %40, %41 : tensor<64x128xi1, #blocked>
      %43 = arith.extsi %arg2 : i32 to i64
      %44 = tt.splat %43 : i64 -> tensor<64xi64, #blocked1>
      %45 = arith.muli %11, %44 : tensor<64xi64, #blocked1>
      %46 = ttg.convert_layout %45 : tensor<64xi64, #blocked1> -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %47 = tt.expand_dims %46 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi64, #blocked2>
      %48 = ttg.convert_layout %47 : tensor<64x1xi64, #blocked2> -> tensor<64x1xi64, #blocked3>
      %49 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked3>
      %50 = tt.addptr %49, %48 : tensor<64x1x!tt.ptr<f16>, #blocked3>, tensor<64x1xi64, #blocked3>
      %51 = ttg.convert_layout %29 : tensor<128xi64, #blocked1> -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi64, #blocked4>
      %53 = ttg.convert_layout %52 : tensor<1x128xi64, #blocked4> -> tensor<1x128xi64, #blocked>
      %54 = tt.broadcast %50 : tensor<64x1x!tt.ptr<f16>, #blocked3> -> tensor<64x128x!tt.ptr<f16>, #blocked3>
      %55 = ttg.convert_layout %54 : tensor<64x128x!tt.ptr<f16>, #blocked3> -> tensor<64x128x!tt.ptr<f16>, #blocked>
      %56 = tt.broadcast %53 : tensor<1x128xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %57 = tt.addptr %55, %56 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %58 = ttg.convert_layout %57 : tensor<64x128x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
      %59 = ttg.convert_layout %cst : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #blocked>
      %60 = ttg.convert_layout %42 : tensor<64x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      tt.store %58, %59, %60 : tensor<64x128x!tt.ptr<f16>, #blocked>
    }
    tt.return
  ^bb2:  // pred: ^bb0
    %15 = arith.subi %0, %c1_i32 : i32
    %16 = arith.remsi %15, %arg8 : i32
    %17 = arith.divsi %15, %arg8 : i32
    %18 = arith.muli %16, %c64_i32 : i32
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %20 = tt.splat %18 : i32 -> tensor<64xi32, #blocked1>
    %21 = arith.addi %20, %19 : tensor<64xi32, #blocked1>
    %22 = arith.extsi %21 : tensor<64xi32, #blocked1> to tensor<64xi64, #blocked1>
    %23 = tt.splat %arg6 : i32 -> tensor<64xi32, #blocked1>
    %24 = arith.cmpi slt, %21, %23 : tensor<64xi32, #blocked1>
    scf.for %arg19 = %c0_i32 to %3 step %c1_i32  : i32 {
      %25 = arith.muli %arg19, %c128_i32 : i32
      %26 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
      %27 = tt.splat %25 : i32 -> tensor<128xi32, #blocked1>
      %28 = arith.addi %27, %26 : tensor<128xi32, #blocked1>
      %29 = arith.extsi %28 : tensor<128xi32, #blocked1> to tensor<128xi64, #blocked1>
      %30 = arith.extsi %arg7 : i32 to i64
      %31 = tt.splat %30 : i64 -> tensor<128xi64, #blocked1>
      %32 = arith.cmpi slt, %29, %31 : tensor<128xi64, #blocked1>
      %33 = ttg.convert_layout %24 : tensor<64xi1, #blocked1> -> tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %34 = tt.expand_dims %33 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi1, #blocked2>
      %35 = ttg.convert_layout %34 : tensor<64x1xi1, #blocked2> -> tensor<64x1xi1, #blocked3>
      %36 = ttg.convert_layout %32 : tensor<128xi1, #blocked1> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %37 = tt.expand_dims %36 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi1, #blocked4>
      %38 = ttg.convert_layout %37 : tensor<1x128xi1, #blocked4> -> tensor<1x128xi1, #blocked>
      %39 = tt.broadcast %35 : tensor<64x1xi1, #blocked3> -> tensor<64x128xi1, #blocked3>
      %40 = ttg.convert_layout %39 : tensor<64x128xi1, #blocked3> -> tensor<64x128xi1, #blocked>
      %41 = tt.broadcast %38 : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %42 = arith.andi %40, %41 : tensor<64x128xi1, #blocked>
      %43 = arith.muli %17, %arg4 : i32
      %44 = arith.extsi %arg5 : i32 to i64
      %45 = tt.splat %44 : i64 -> tensor<64xi64, #blocked1>
      %46 = arith.muli %22, %45 : tensor<64xi64, #blocked1>
      %47 = arith.extsi %43 : i32 to i64
      %48 = tt.splat %47 : i64 -> tensor<64xi64, #blocked1>
      %49 = arith.addi %48, %46 : tensor<64xi64, #blocked1>
      %50 = ttg.convert_layout %49 : tensor<64xi64, #blocked1> -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %51 = tt.expand_dims %50 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi64, #blocked2>
      %52 = ttg.convert_layout %51 : tensor<64x1xi64, #blocked2> -> tensor<64x1xi64, #blocked3>
      %53 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked3>
      %54 = tt.addptr %53, %52 : tensor<64x1x!tt.ptr<f16>, #blocked3>, tensor<64x1xi64, #blocked3>
      %55 = ttg.convert_layout %29 : tensor<128xi64, #blocked1> -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
      %56 = tt.expand_dims %55 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi64, #blocked4>
      %57 = ttg.convert_layout %56 : tensor<1x128xi64, #blocked4> -> tensor<1x128xi64, #blocked>
      %58 = tt.broadcast %54 : tensor<64x1x!tt.ptr<f16>, #blocked3> -> tensor<64x128x!tt.ptr<f16>, #blocked3>
      %59 = ttg.convert_layout %58 : tensor<64x128x!tt.ptr<f16>, #blocked3> -> tensor<64x128x!tt.ptr<f16>, #blocked>
      %60 = tt.broadcast %57 : tensor<1x128xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %61 = tt.addptr %59, %60 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %62 = ttg.convert_layout %61 : tensor<64x128x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
      %63 = ttg.convert_layout %42 : tensor<64x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %64 = ttg.convert_layout %cst : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #blocked>
      %65 = tt.load %62, %63, %64 : tensor<64x128x!tt.ptr<f16>, #blocked>
      %66 = arith.muli %17, %arg1 : i32
      %67 = arith.extsi %arg2 : i32 to i64
      %68 = tt.splat %67 : i64 -> tensor<64xi64, #blocked1>
      %69 = arith.muli %22, %68 : tensor<64xi64, #blocked1>
      %70 = arith.extsi %66 : i32 to i64
      %71 = tt.splat %70 : i64 -> tensor<64xi64, #blocked1>
      %72 = arith.addi %71, %69 : tensor<64xi64, #blocked1>
      %73 = ttg.convert_layout %72 : tensor<64xi64, #blocked1> -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %74 = tt.expand_dims %73 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi64, #blocked2>
      %75 = ttg.convert_layout %74 : tensor<64x1xi64, #blocked2> -> tensor<64x1xi64, #blocked3>
      %76 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked3>
      %77 = tt.addptr %76, %75 : tensor<64x1x!tt.ptr<f16>, #blocked3>, tensor<64x1xi64, #blocked3>
      %78 = tt.broadcast %77 : tensor<64x1x!tt.ptr<f16>, #blocked3> -> tensor<64x128x!tt.ptr<f16>, #blocked3>
      %79 = ttg.convert_layout %78 : tensor<64x128x!tt.ptr<f16>, #blocked3> -> tensor<64x128x!tt.ptr<f16>, #blocked>
      %80 = tt.addptr %79, %60 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %81 = ttg.convert_layout %80 : tensor<64x128x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
      %82 = ttg.convert_layout %65 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #blocked>
      %83 = ttg.convert_layout %42 : tensor<64x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      tt.store %81, %82, %83 : tensor<64x128x!tt.ptr<f16>, #blocked>
    }
    tt.return
  }
}

// -----

// Keep a two-load packed MX assembly at the incumbent two conversions.
// Exact original TTIR SHA-256: 25ce16dfe7a62511f3e9d7bef9444d4011f85b86cf167ff9267343fa1a4e88c0.
//
// BASELINE-LABEL: @production_two_load_packed_mx_layout
// BASELINE-COUNT-2: tt.load
// BASELINE: tt.join
// BASELINE-COUNT-2: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_two_load_packed_mx_layout
// OPTIMIZED-COUNT-2: tt.load
// OPTIMIZED: tt.join
// OPTIMIZED-COUNT-2: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 8, 4, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [0, 1, 2]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 1, 2], order = [0, 1, 2]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 1, 2], order = [2, 1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked12 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked13 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked14 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked15 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [1, 2, 2, 1], order = [3, 2, 1, 0]}>
#blocked16 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 16, 2], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked17 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#blocked18 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked19 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [0, 1, 2]}>
#blocked20 = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 4], order = [1, 2, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @production_two_load_packed_mx_layout(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<7> : tensor<1x64x4xi16, #blocked>
    %cst_0 = arith.constant dense<3.38953139E+38> : tensor<1x64x4x32xf32, #blocked1>
    %cst_1 = arith.constant dense<255> : tensor<1x64x4x1xi32, #blocked2>
    %cst_2 = arith.constant dense<0x7FC00000> : tensor<1x64x4x32xf32, #blocked1>
    %cst_3 = arith.constant dense<-3.38953139E+38> : tensor<1x64x4x32xf32, #blocked1>
    %cst_4 = arith.constant dense<16> : tensor<1x64x64xi32, #blocked3>
    %cst_5 = arith.constant dense<65535> : tensor<1x64x64xi32, #blocked3>
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i64 = arith.constant 4 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_num_programs x : i32
    %3 = arith.cmpi ne, %arg14, %2 : i32
    %4:3 = scf.if %3 -> (i64, i64, i64) {
      %165 = arith.muli %arg13, %arg14 : i32
      %166 = arith.extsi %165 : i32 to i64
      %167 = arith.divsi %1, %166 : i64
      %168 = arith.remsi %1, %166 : i64
      %169 = arith.extsi %arg14 : i32 to i64
      %170 = arith.divsi %168, %169 : i64
      %171 = arith.remsi %1, %169 : i64
      scf.yield %171, %167, %170 : i64, i64, i64
    } else {
      %165 = tt.get_program_id y : i32
      %166 = arith.extsi %165 : i32 to i64
      %167 = tt.get_program_id z : i32
      %168 = arith.extsi %167 : i32 to i64
      scf.yield %1, %168, %166 : i64, i64, i64
    }
    %5 = arith.muli %4#2, %c64_i64 : i64
    %6 = arith.muli %4#0, %c128_i64 : i64
    %7 = arith.muli %4#0, %c64_i64 : i64
    %8 = arith.muli %4#0, %c4_i64 : i64
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked4>
    %10 = ttg.convert_layout %9 : tensor<64xi32, #blocked4> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x64xi32, #blocked5>
    %12 = ttg.convert_layout %11 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #blocked6>
    %13 = ttg.convert_layout %12 : tensor<1x64xi32, #blocked6> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked7}>>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked7}>> -> tensor<1x64x1xi32, #blocked7>
    %15 = ttg.convert_layout %14 : tensor<1x64x1xi32, #blocked7> -> tensor<1x64x1xi32, #blocked8>
    %16 = arith.extsi %15 : tensor<1x64x1xi32, #blocked8> to tensor<1x64x1xi64, #blocked8>
    %17 = arith.cmpi slt, %4#1, %c1_i64 : i64
    %18 = tt.splat %5 : i64 -> tensor<1x64x1xi64, #blocked8>
    %19 = arith.addi %18, %16 : tensor<1x64x1xi64, #blocked8>
    %20 = arith.extsi %arg9 : i32 to i64
    %21 = tt.splat %20 : i64 -> tensor<1x64x1xi64, #blocked8>
    %22 = arith.cmpi slt, %19, %21 : tensor<1x64x1xi64, #blocked8>
    %23 = ttg.convert_layout %12 : tensor<1x64xi32, #blocked6> -> tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked9}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked9}>> -> tensor<1x1x64xi32, #blocked9>
    %25 = ttg.convert_layout %24 : tensor<1x1x64xi32, #blocked9> -> tensor<1x1x64xi32, #blocked10>
    %26 = arith.extsi %25 : tensor<1x1x64xi32, #blocked10> to tensor<1x1x64xi64, #blocked10>
    %27 = arith.extsi %arg4 : i32 to i64
    %28 = arith.muli %4#1, %27 : i64
    %29 = arith.extsi %arg5 : i32 to i64
    %30 = arith.muli %5, %29 : i64
    %31 = arith.addi %30, %7 : i64
    %32 = arith.addi %28, %31 : i64
    %33 = tt.addptr %arg3, %32 : !tt.ptr<i8>, i64
    %34 = tt.splat %7 : i64 -> tensor<1x1x64xi64, #blocked10>
    %35 = arith.addi %34, %26 : tensor<1x1x64xi64, #blocked10>
    %36 = arith.divsi %arg10, %c2_i32 : i32
    %37 = arith.extsi %36 : i32 to i64
    %38 = tt.splat %37 : i64 -> tensor<1x1x64xi64, #blocked10>
    %39 = arith.cmpi slt, %35, %38 : tensor<1x1x64xi64, #blocked10>
    %40 = tt.splat %17 : i1 -> tensor<1x64x1xi1, #blocked8>
    %41 = arith.andi %40, %22 : tensor<1x64x1xi1, #blocked8>
    %42 = tt.broadcast %41 : tensor<1x64x1xi1, #blocked8> -> tensor<1x64x64xi1, #blocked8>
    %43 = ttg.convert_layout %42 : tensor<1x64x64xi1, #blocked8> -> tensor<1x64x64xi1, #blocked3>
    %44 = tt.broadcast %39 : tensor<1x1x64xi1, #blocked10> -> tensor<1x64x64xi1, #blocked10>
    %45 = ttg.convert_layout %44 : tensor<1x64x64xi1, #blocked10> -> tensor<1x64x64xi1, #blocked3>
    %46 = arith.andi %43, %45 : tensor<1x64x64xi1, #blocked3>
    %47 = tt.splat %29 : i64 -> tensor<1x64x1xi64, #blocked8>
    %48 = arith.muli %16, %47 : tensor<1x64x1xi64, #blocked8>
    %49 = tt.broadcast %48 : tensor<1x64x1xi64, #blocked8> -> tensor<1x64x64xi64, #blocked8>
    %50 = ttg.convert_layout %49 : tensor<1x64x64xi64, #blocked8> -> tensor<1x64x64xi64, #blocked3>
    %51 = tt.broadcast %26 : tensor<1x1x64xi64, #blocked10> -> tensor<1x64x64xi64, #blocked10>
    %52 = ttg.convert_layout %51 : tensor<1x64x64xi64, #blocked10> -> tensor<1x64x64xi64, #blocked3>
    %53 = arith.addi %50, %52 : tensor<1x64x64xi64, #blocked3>
    %54 = tt.splat %33 : !tt.ptr<i8> -> tensor<1x64x64x!tt.ptr<i8>, #blocked3>
    %55 = tt.addptr %54, %53 : tensor<1x64x64x!tt.ptr<i8>, #blocked3>, tensor<1x64x64xi64, #blocked3>
    %56 = ttg.convert_layout %55 : tensor<1x64x64x!tt.ptr<i8>, #blocked3> -> tensor<1x64x64x!tt.ptr<i8>, #blocked11>
    %57 = ttg.convert_layout %46 : tensor<1x64x64xi1, #blocked3> -> tensor<1x64x64xi1, #blocked11>
    %58 = tt.load %56, %57 : tensor<1x64x64x!tt.ptr<i8>, #blocked11>
    %59 = ttg.convert_layout %58 : tensor<1x64x64xi8, #blocked11> -> tensor<1x64x64xi8, #blocked3>
    %60 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked4>
    %61 = ttg.convert_layout %60 : tensor<4xi32, #blocked4> -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %62 = tt.expand_dims %61 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x4xi32, #blocked5>
    %63 = ttg.convert_layout %62 : tensor<1x4xi32, #blocked5> -> tensor<1x4xi32, #blocked12>
    %64 = ttg.convert_layout %63 : tensor<1x4xi32, #blocked12> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked13}>>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked13}>> -> tensor<1x1x4xi32, #blocked13>
    %66 = ttg.convert_layout %65 : tensor<1x1x4xi32, #blocked13> -> tensor<1x1x4xi32, #blocked14>
    %67 = arith.extsi %66 : tensor<1x1x4xi32, #blocked14> to tensor<1x1x4xi64, #blocked14>
    %68 = arith.extsi %arg7 : i32 to i64
    %69 = arith.muli %4#1, %68 : i64
    %70 = arith.extsi %arg8 : i32 to i64
    %71 = arith.muli %5, %70 : i64
    %72 = arith.addi %71, %8 : i64
    %73 = arith.addi %69, %72 : i64
    %74 = tt.addptr %arg6, %73 : !tt.ptr<i8>, i64
    %75 = tt.splat %8 : i64 -> tensor<1x1x4xi64, #blocked14>
    %76 = arith.addi %75, %67 : tensor<1x1x4xi64, #blocked14>
    %77 = arith.addi %arg10, %c31_i32 : i32
    %78 = arith.divsi %77, %c32_i32 : i32
    %79 = arith.extsi %78 : i32 to i64
    %80 = tt.splat %79 : i64 -> tensor<1x1x4xi64, #blocked14>
    %81 = arith.cmpi slt, %76, %80 : tensor<1x1x4xi64, #blocked14>
    %82 = tt.broadcast %41 : tensor<1x64x1xi1, #blocked8> -> tensor<1x64x4xi1, #blocked8>
    %83 = ttg.convert_layout %82 : tensor<1x64x4xi1, #blocked8> -> tensor<1x64x4xi1, #blocked>
    %84 = tt.broadcast %81 : tensor<1x1x4xi1, #blocked14> -> tensor<1x64x4xi1, #blocked14>
    %85 = ttg.convert_layout %84 : tensor<1x64x4xi1, #blocked14> -> tensor<1x64x4xi1, #blocked>
    %86 = arith.andi %83, %85 : tensor<1x64x4xi1, #blocked>
    %87 = tt.splat %70 : i64 -> tensor<1x64x1xi64, #blocked8>
    %88 = arith.muli %16, %87 : tensor<1x64x1xi64, #blocked8>
    %89 = tt.broadcast %88 : tensor<1x64x1xi64, #blocked8> -> tensor<1x64x4xi64, #blocked8>
    %90 = ttg.convert_layout %89 : tensor<1x64x4xi64, #blocked8> -> tensor<1x64x4xi64, #blocked>
    %91 = tt.broadcast %67 : tensor<1x1x4xi64, #blocked14> -> tensor<1x64x4xi64, #blocked14>
    %92 = ttg.convert_layout %91 : tensor<1x64x4xi64, #blocked14> -> tensor<1x64x4xi64, #blocked>
    %93 = arith.addi %90, %92 : tensor<1x64x4xi64, #blocked>
    %94 = tt.splat %74 : !tt.ptr<i8> -> tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %95 = tt.addptr %94, %93 : tensor<1x64x4x!tt.ptr<i8>, #blocked>, tensor<1x64x4xi64, #blocked>
    %96 = ttg.convert_layout %95 : tensor<1x64x4x!tt.ptr<i8>, #blocked> -> tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %97 = ttg.convert_layout %86 : tensor<1x64x4xi1, #blocked> -> tensor<1x64x4xi1, #blocked>
    %98 = tt.load %96, %97 : tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %99 = arith.extui %98 : tensor<1x64x4xi8, #blocked> to tensor<1x64x4xi16, #blocked>
    %100 = arith.shli %99, %cst : tensor<1x64x4xi16, #blocked>
    %101 = tt.bitcast %100 : tensor<1x64x4xi16, #blocked> -> tensor<1x64x4xbf16, #blocked>
    %102 = tt.elementwise_inline_asm "\0A            {\0A            .reg .b8 in_8;\0A            .reg .f16x2 out;\0A            cvt.u8.u32 in_8, $1;\0A            cvt.rn.f16x2.e2m1x2 out, in_8;\0A            mov.b32 $0, out;\0A            }\0A            " {constraints = "=r,r", packed_element = 1 : i32, pure = true} %59 : tensor<1x64x64xi8, #blocked3> -> tensor<1x64x64xi32, #blocked3>
    %103 = arith.andi %102, %cst_5 : tensor<1x64x64xi32, #blocked3>
    %104 = arith.trunci %103 : tensor<1x64x64xi32, #blocked3> to tensor<1x64x64xi16, #blocked3>
    %105 = arith.shrui %102, %cst_4 : tensor<1x64x64xi32, #blocked3>
    %106 = arith.trunci %105 : tensor<1x64x64xi32, #blocked3> to tensor<1x64x64xi16, #blocked3>
    %107 = tt.bitcast %104 : tensor<1x64x64xi16, #blocked3> -> tensor<1x64x64xf16, #blocked3>
    %108 = tt.bitcast %106 : tensor<1x64x64xi16, #blocked3> -> tensor<1x64x64xf16, #blocked3>
    %109 = arith.extf %107 : tensor<1x64x64xf16, #blocked3> to tensor<1x64x64xf32, #blocked3>
    %110 = arith.truncf %109 : tensor<1x64x64xf32, #blocked3> to tensor<1x64x64xbf16, #blocked3>
    %111 = arith.extf %108 : tensor<1x64x64xf16, #blocked3> to tensor<1x64x64xf32, #blocked3>
    %112 = arith.truncf %111 : tensor<1x64x64xf32, #blocked3> to tensor<1x64x64xbf16, #blocked3>
    %113 = tt.join %110, %112 : tensor<1x64x64xbf16, #blocked3> -> tensor<1x64x64x2xbf16, #blocked15>
    %114 = ttg.convert_layout %113 : tensor<1x64x64x2xbf16, #blocked15> -> tensor<1x64x64x2xbf16, #blocked16>
    %115 = tt.reshape %114 : tensor<1x64x64x2xbf16, #blocked16> -> tensor<1x64x4x32xbf16, #blocked1>
    %116 = tt.reshape %101 : tensor<1x64x4xbf16, #blocked> -> tensor<1x64x4x1xbf16, #blocked2>
    %117 = tt.reshape %98 : tensor<1x64x4xi8, #blocked> -> tensor<1x64x4x1xi8, #blocked2>
    %118 = tt.broadcast %116 : tensor<1x64x4x1xbf16, #blocked2> -> tensor<1x64x4x32xbf16, #blocked2>
    %119 = ttg.convert_layout %118 : tensor<1x64x4x32xbf16, #blocked2> -> tensor<1x64x4x32xbf16, #blocked1>
    %120 = arith.mulf %115, %119 : tensor<1x64x4x32xbf16, #blocked1>
    %121 = arith.extf %120 : tensor<1x64x4x32xbf16, #blocked1> to tensor<1x64x4x32xf32, #blocked1>
    %122 = tt.clampf %121, %cst_3, %cst_0, propagateNan = all : tensor<1x64x4x32xf32, #blocked1>
    %123 = arith.extui %117 : tensor<1x64x4x1xi8, #blocked2> to tensor<1x64x4x1xi32, #blocked2>
    %124 = arith.cmpi eq, %123, %cst_1 : tensor<1x64x4x1xi32, #blocked2>
    %125 = tt.broadcast %124 : tensor<1x64x4x1xi1, #blocked2> -> tensor<1x64x4x32xi1, #blocked2>
    %126 = ttg.convert_layout %125 : tensor<1x64x4x32xi1, #blocked2> -> tensor<1x64x4x32xi1, #blocked1>
    %127 = arith.select %126, %cst_2, %122 : tensor<1x64x4x32xi1, #blocked1>, tensor<1x64x4x32xf32, #blocked1>
    %128 = tt.reshape %127 : tensor<1x64x4x32xf32, #blocked1> -> tensor<1x64x128xf32, #blocked17>
    %129 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked4>
    %130 = ttg.convert_layout %129 : tensor<128xi32, #blocked4> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %131 = tt.expand_dims %130 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x128xi32, #blocked5>
    %132 = ttg.convert_layout %131 : tensor<1x128xi32, #blocked5> -> tensor<1x128xi32, #blocked18>
    %133 = ttg.convert_layout %132 : tensor<1x128xi32, #blocked18> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked19}>>
    %134 = tt.expand_dims %133 {axis = 1 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked19}>> -> tensor<1x1x128xi32, #blocked19>
    %135 = ttg.convert_layout %134 : tensor<1x1x128xi32, #blocked19> -> tensor<1x1x128xi32, #blocked17>
    %136 = arith.extsi %135 : tensor<1x1x128xi32, #blocked17> to tensor<1x1x128xi64, #blocked17>
    %137 = arith.extsi %arg1 : i32 to i64
    %138 = arith.muli %4#1, %137 : i64
    %139 = arith.extsi %arg2 : i32 to i64
    %140 = arith.muli %6, %139 : i64
    %141 = arith.addi %5, %140 : i64
    %142 = arith.addi %138, %141 : i64
    %143 = tt.addptr %arg0, %142 : !tt.ptr<bf16>, i64
    %144 = tt.splat %6 : i64 -> tensor<1x1x128xi64, #blocked17>
    %145 = arith.addi %144, %136 : tensor<1x1x128xi64, #blocked17>
    %146 = arith.extsi %arg10 : i32 to i64
    %147 = tt.splat %146 : i64 -> tensor<1x1x128xi64, #blocked17>
    %148 = arith.cmpi slt, %145, %147 : tensor<1x1x128xi64, #blocked17>
    %149 = tt.broadcast %41 : tensor<1x64x1xi1, #blocked8> -> tensor<1x64x128xi1, #blocked8>
    %150 = ttg.convert_layout %149 : tensor<1x64x128xi1, #blocked8> -> tensor<1x64x128xi1, #blocked17>
    %151 = tt.broadcast %148 : tensor<1x1x128xi1, #blocked17> -> tensor<1x64x128xi1, #blocked17>
    %152 = arith.andi %150, %151 : tensor<1x64x128xi1, #blocked17>
    %153 = tt.splat %139 : i64 -> tensor<1x1x128xi64, #blocked17>
    %154 = arith.muli %136, %153 : tensor<1x1x128xi64, #blocked17>
    %155 = tt.broadcast %16 : tensor<1x64x1xi64, #blocked8> -> tensor<1x64x128xi64, #blocked8>
    %156 = ttg.convert_layout %155 : tensor<1x64x128xi64, #blocked8> -> tensor<1x64x128xi64, #blocked17>
    %157 = tt.broadcast %154 : tensor<1x1x128xi64, #blocked17> -> tensor<1x64x128xi64, #blocked17>
    %158 = arith.addi %156, %157 : tensor<1x64x128xi64, #blocked17>
    %159 = tt.splat %143 : !tt.ptr<bf16> -> tensor<1x64x128x!tt.ptr<bf16>, #blocked17>
    %160 = tt.addptr %159, %158 : tensor<1x64x128x!tt.ptr<bf16>, #blocked17>, tensor<1x64x128xi64, #blocked17>
    %161 = arith.truncf %128 : tensor<1x64x128xf32, #blocked17> to tensor<1x64x128xbf16, #blocked17>
    %162 = ttg.convert_layout %160 : tensor<1x64x128x!tt.ptr<bf16>, #blocked17> -> tensor<1x64x128x!tt.ptr<bf16>, #blocked20>
    %163 = ttg.convert_layout %161 : tensor<1x64x128xbf16, #blocked17> -> tensor<1x64x128xbf16, #blocked20>
    %164 = ttg.convert_layout %152 : tensor<1x64x128xi1, #blocked17> -> tensor<1x64x128xi1, #blocked20>
    tt.store %162, %163, %164 : tensor<1x64x128x!tt.ptr<bf16>, #blocked20>
    tt.return
  }
}

// -----

// A repeated global layout assignment must use a stable constant order so
// equivalent matmul and reduction layouts do not continually rename aliases.
//
// BASELINE-LABEL: @production_stable_layout_constant_order
// BASELINE: arith.constant dense<3> : tensor<32xi32
// BASELINE: arith.constant dense<2> : tensor<32xi32
// BASELINE: arith.constant dense<1> : tensor<32xi32
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_stable_layout_constant_order
// OPTIMIZED: arith.constant dense<1> : tensor<32xi32
// OPTIMIZED-NEXT: {{.*}}arith.constant dense<2> : tensor<32xi32
// OPTIMIZED-NEXT: {{.*}}arith.constant dense<3> : tensor<32xi32
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @production_stable_layout_constant_order(%out: !tt.ptr<i32>) {
    %three = arith.constant 3 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %ones = tt.splat %one : i32 -> tensor<32xi32, #blocked>
    %twos = tt.splat %two : i32 -> tensor<32xi32, #blocked>
    %threes = tt.splat %three : i32 -> tensor<32xi32, #blocked>
    %offset1 = arith.addi %range, %ones : tensor<32xi32, #blocked>
    %offset2 = arith.addi %range, %twos : tensor<32xi32, #blocked>
    %offset3 = arith.addi %range, %threes : tensor<32xi32, #blocked>
    %sum0 = arith.addi %offset1, %offset2 : tensor<32xi32, #blocked>
    %sum = arith.addi %sum0, %offset3 : tensor<32xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %base, %range : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %ptrs, %sum : tensor<32x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// The third original private helper combines a rank-two pairwise bitmap, three
// masked FP8 loads, and one scalar reduction. Protect their complete shared
// layout without duplicating any FP8 conversions.
// Exact original TTIR SHA-256: dca1244813e62642847f242ed73ac5644cab98f7fa74439913caab9c17675331.
//
// BASELINE-LABEL: @production_pairwise_fp8_reduction_memory_protocol
// BASELINE-NOT: ttg.convert_layout
// BASELINE: "tt.reduce"
// BASELINE-NOT: ttg.convert_layout
// BASELINE-COUNT-3: tt.load {{.*}}f8E5M2
// BASELINE-NOT: ttg.convert_layout
// BASELINE: "tt.reduce"
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.store
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_pairwise_fp8_reduction_memory_protocol
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: "tt.reduce"
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-3: tt.load {{.*}}f8E5M2
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: "tt.reduce"
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @production_pairwise_fp8_reduction_memory_protocol(%arg0: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg1: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg2: !tt.ptr<f8E5M2> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg3: i64 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg4: i64 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg5: !tt.ptr<f8E5M2> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg6: i64 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg7: !tt.ptr<i1> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg8: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg9: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg10: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg11: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg12: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg13: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg14: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg15: i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg16: !tt.ptr<f32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg17: !tt.ptr<f32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg18: !tt.ptr<f32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg19: !tt.ptr<i16> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}, %arg20: !tt.ptr<i32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}) attributes {noinline = true} {
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %c2139095040_i32 = arith.constant 2139095040 : i32
    %cst_0 = arith.constant 1.000000e-30 : f32
    %c932333861_i32 = arith.constant 932333861 : i32
    %cst_1 = arith.constant dense<-5.734400e+04> : tensor<32x128xf32, #blocked1>
    %cst_2 = arith.constant dense<5.734400e+04> : tensor<32x128xf32, #blocked1>
    %c31_i32 = arith.constant 31 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf8E5M2, #blocked1>
    %cst_4 = arith.constant dense<1> : tensor<32xi32, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %cst_5 = arith.constant dense<0> : tensor<32x2xi8, #blocked2>
    %cst_6 = arith.constant dense<0> : tensor<32x2xi32, #blocked2>
    %cst_7 = arith.constant dense<2> : tensor<2xi32, #blocked>
    %cst_8 = arith.constant dense<0> : tensor<32xi32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.muli %arg0, %c32_i32 : i32
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %2 = tt.splat %0 : i32 -> tensor<32xi32, #blocked>
    %3 = arith.addi %2, %1 : tensor<32xi32, #blocked>
    %4 = tt.splat %arg13 : i32 -> tensor<32xi32, #blocked>
    %5 = arith.cmpi slt, %3, %4 : tensor<32xi32, #blocked>
    %6 = tt.load %arg16 : !tt.ptr<f32>
    %7 = tt.splat %arg20 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
    %8 = tt.addptr %7, %3 : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
    %9 = ttg.convert_layout %8 : tensor<32x!tt.ptr<i32>, #blocked> -> tensor<32x!tt.ptr<i32>, #blocked>
    %10 = ttg.convert_layout %5 : tensor<32xi1, #blocked> -> tensor<32xi1, #blocked>
    %11 = ttg.convert_layout %cst_8 : tensor<32xi32, #blocked> -> tensor<32xi32, #blocked>
    %12 = tt.load %9, %10, %11 : tensor<32x!tt.ptr<i32>, #blocked>
    %13 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #blocked>
    %14 = ttg.convert_layout %13 : tensor<2xi32, #blocked> -> tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %15 = tt.expand_dims %14 {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi32, #blocked3>
    %16 = ttg.convert_layout %15 : tensor<1x2xi32, #blocked3> -> tensor<1x2xi32, #blocked2>
    %17 = ttg.convert_layout %12 : tensor<32xi32, #blocked> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %18 = tt.expand_dims %17 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %19 = ttg.convert_layout %18 : tensor<32x1xi32, #blocked4> -> tensor<32x1xi32, #blocked5>
    %20 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #blocked5>
    %21 = arith.muli %19, %20 : tensor<32x1xi32, #blocked5>
    %22 = tt.splat %arg7 : !tt.ptr<i1> -> tensor<1x2x!tt.ptr<i1>, #blocked2>
    %23 = tt.addptr %22, %16 : tensor<1x2x!tt.ptr<i1>, #blocked2>, tensor<1x2xi32, #blocked2>
    %24 = tt.broadcast %23 : tensor<1x2x!tt.ptr<i1>, #blocked2> -> tensor<32x2x!tt.ptr<i1>, #blocked2>
    %25 = tt.broadcast %21 : tensor<32x1xi32, #blocked5> -> tensor<32x2xi32, #blocked5>
    %26 = ttg.convert_layout %25 : tensor<32x2xi32, #blocked5> -> tensor<32x2xi32, #blocked2>
    %27 = tt.addptr %24, %26 : tensor<32x2x!tt.ptr<i1>, #blocked2>, tensor<32x2xi32, #blocked2>
    %28 = arith.cmpi slt, %13, %cst_7 : tensor<2xi32, #blocked>
    %29 = ttg.convert_layout %5 : tensor<32xi1, #blocked> -> tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %30 = tt.expand_dims %29 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi1, #blocked4>
    %31 = ttg.convert_layout %30 : tensor<32x1xi1, #blocked4> -> tensor<32x1xi1, #blocked5>
    %32 = ttg.convert_layout %28 : tensor<2xi1, #blocked> -> tensor<2xi1, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<2xi1, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi1, #blocked3>
    %34 = ttg.convert_layout %33 : tensor<1x2xi1, #blocked3> -> tensor<1x2xi1, #blocked2>
    %35 = tt.broadcast %31 : tensor<32x1xi1, #blocked5> -> tensor<32x2xi1, #blocked5>
    %36 = ttg.convert_layout %35 : tensor<32x2xi1, #blocked5> -> tensor<32x2xi1, #blocked2>
    %37 = tt.broadcast %34 : tensor<1x2xi1, #blocked2> -> tensor<32x2xi1, #blocked2>
    %38 = arith.andi %36, %37 : tensor<32x2xi1, #blocked2>
    %39 = tt.bitcast %27 : tensor<32x2x!tt.ptr<i1>, #blocked2> -> tensor<32x2x!tt.ptr<i8>, #blocked2>
    %40 = ttg.convert_layout %39 : tensor<32x2x!tt.ptr<i8>, #blocked2> -> tensor<32x2x!tt.ptr<i8>, #blocked2>
    %41 = ttg.convert_layout %38 : tensor<32x2xi1, #blocked2> -> tensor<32x2xi1, #blocked2>
    %42 = ttg.convert_layout %cst_5 : tensor<32x2xi8, #blocked2> -> tensor<32x2xi8, #blocked2>
    %43 = tt.load %40, %41, %42 : tensor<32x2x!tt.ptr<i8>, #blocked2>
    %44 = arith.cmpi ne, %43, %cst_5 : tensor<32x2xi8, #blocked2>
    %45 = arith.extui %44 : tensor<32x2xi1, #blocked2> to tensor<32x2xi32, #blocked2>
    %46 = arith.cmpi ne, %45, %cst_6 : tensor<32x2xi32, #blocked2>
    %47 = arith.extui %46 : tensor<32x2xi1, #blocked2> to tensor<32x2xi32, #blocked2>
    %48 = tt.broadcast %16 : tensor<1x2xi32, #blocked2> -> tensor<32x2xi32, #blocked2>
    %49 = arith.shli %47, %48 : tensor<32x2xi32, #blocked2>
    %50 = "tt.reduce"(%49) <{axis = 1 : i32}> ({
    ^bb0(%arg21: i32, %arg22: i32):
      %204 = arith.addi %arg21, %arg22 : i32
      tt.reduce.return %204 : i32
    }) : (tensor<32x2xi32, #blocked2>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %51 = ttg.convert_layout %50 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32xi32, #blocked>
    %52 = arith.muli %arg1, %c128_i32 : i32
    %53 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %54 = tt.splat %52 : i32 -> tensor<128xi32, #blocked>
    %55 = arith.addi %54, %53 : tensor<128xi32, #blocked>
    %56 = tt.splat %arg14 : i32 -> tensor<128xi32, #blocked>
    %57 = arith.cmpi slt, %55, %56 : tensor<128xi32, #blocked>
    %58 = ttg.convert_layout %57 : tensor<128xi1, #blocked> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %59 = tt.expand_dims %58 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi1, #blocked3>
    %60 = ttg.convert_layout %59 : tensor<1x128xi1, #blocked3> -> tensor<1x128xi1, #blocked1>
    %61 = tt.broadcast %31 : tensor<32x1xi1, #blocked5> -> tensor<32x128xi1, #blocked5>
    %62 = ttg.convert_layout %61 : tensor<32x128xi1, #blocked5> -> tensor<32x128xi1, #blocked1>
    %63 = tt.broadcast %60 : tensor<1x128xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
    %64 = arith.andi %62, %63 : tensor<32x128xi1, #blocked1>
    %65 = tt.extern_elementwise %51 {libname = "", libpath = "", pure = true, symbol = "__nv_ffs"} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    %66 = arith.cmpi ne, %51, %cst_8 : tensor<32xi32, #blocked>
    %67 = arith.extui %66 : tensor<32xi1, #blocked> to tensor<32xi32, #blocked>
    %68 = arith.subi %51, %67 : tensor<32xi32, #blocked>
    %69 = arith.andi %51, %68 : tensor<32xi32, #blocked>
    %70 = arith.cmpi ne, %65, %cst_8 : tensor<32xi32, #blocked>
    %71 = ttg.convert_layout %70 : tensor<32xi1, #blocked> -> tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %72 = tt.expand_dims %71 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi1, #blocked4>
    %73 = ttg.convert_layout %72 : tensor<32x1xi1, #blocked4> -> tensor<32x1xi1, #blocked5>
    %74 = tt.broadcast %73 : tensor<32x1xi1, #blocked5> -> tensor<32x128xi1, #blocked5>
    %75 = ttg.convert_layout %74 : tensor<32x128xi1, #blocked5> -> tensor<32x128xi1, #blocked1>
    %76 = arith.andi %64, %75 : tensor<32x128xi1, #blocked1>
    %77 = arith.subi %65, %cst_4 : tensor<32xi32, #blocked>
    %78 = arith.maxsi %77, %cst_8 : tensor<32xi32, #blocked>
    %79 = ttg.convert_layout %78 : tensor<32xi32, #blocked> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %80 = tt.expand_dims %79 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %81 = ttg.convert_layout %80 : tensor<32x1xi32, #blocked4> -> tensor<32x1xi32, #blocked5>
    %82 = arith.extsi %81 : tensor<32x1xi32, #blocked5> to tensor<32x1xi64, #blocked5>
    %83 = tt.splat %arg3 : i64 -> tensor<32x1xi64, #blocked5>
    %84 = arith.muli %82, %83 : tensor<32x1xi64, #blocked5>
    %85 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<32x1x!tt.ptr<f8E5M2>, #blocked5>
    %86 = arith.extsi %19 : tensor<32x1xi32, #blocked5> to tensor<32x1xi64, #blocked5>
    %87 = tt.splat %arg4 : i64 -> tensor<32x1xi64, #blocked5>
    %88 = arith.muli %86, %87 : tensor<32x1xi64, #blocked5>
    %89 = arith.addi %84, %88 : tensor<32x1xi64, #blocked5>
    %90 = tt.addptr %85, %89 : tensor<32x1x!tt.ptr<f8E5M2>, #blocked5>, tensor<32x1xi64, #blocked5>
    %91 = ttg.convert_layout %55 : tensor<128xi32, #blocked> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %92 = tt.expand_dims %91 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi32, #blocked3>
    %93 = ttg.convert_layout %92 : tensor<1x128xi32, #blocked3> -> tensor<1x128xi32, #blocked1>
    %94 = tt.broadcast %90 : tensor<32x1x!tt.ptr<f8E5M2>, #blocked5> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked5>
    %95 = ttg.convert_layout %94 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked5> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked1>
    %96 = tt.broadcast %93 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
    %97 = tt.addptr %95, %96 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<32x128xi32, #blocked1>
    %98 = ttg.convert_layout %97 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked1> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked6>
    %99 = ttg.convert_layout %76 : tensor<32x128xi1, #blocked1> -> tensor<32x128xi1, #blocked6>
    %100 = ttg.convert_layout %cst_3 : tensor<32x128xf8E5M2, #blocked1> -> tensor<32x128xf8E5M2, #blocked6>
    %101 = tt.load %98, %99, %100 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked6>
    %102 = ttg.convert_layout %101 : tensor<32x128xf8E5M2, #blocked6> -> tensor<32x128xf8E5M2, #blocked1>
    %103 = tt.fp_to_fp %102 : tensor<32x128xf8E5M2, #blocked1> -> tensor<32x128xf32, #blocked1>
    %104 = tt.splat %6 : f32 -> tensor<32x128xf32, #blocked1>
    %105 = arith.mulf %103, %104 : tensor<32x128xf32, #blocked1>
    %106 = tt.extern_elementwise %69 {libname = "", libpath = "", pure = true, symbol = "__nv_ffs"} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    %107 = arith.cmpi ne, %69, %cst_8 : tensor<32xi32, #blocked>
    %108 = arith.extui %107 : tensor<32xi1, #blocked> to tensor<32xi32, #blocked>
    %109 = arith.subi %69, %108 : tensor<32xi32, #blocked>
    %110 = arith.andi %69, %109 : tensor<32xi32, #blocked>
    %111 = arith.cmpi ne, %106, %cst_8 : tensor<32xi32, #blocked>
    %112 = ttg.convert_layout %111 : tensor<32xi1, #blocked> -> tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %113 = tt.expand_dims %112 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi1, #blocked4>
    %114 = ttg.convert_layout %113 : tensor<32x1xi1, #blocked4> -> tensor<32x1xi1, #blocked5>
    %115 = tt.broadcast %114 : tensor<32x1xi1, #blocked5> -> tensor<32x128xi1, #blocked5>
    %116 = ttg.convert_layout %115 : tensor<32x128xi1, #blocked5> -> tensor<32x128xi1, #blocked1>
    %117 = arith.andi %64, %116 : tensor<32x128xi1, #blocked1>
    %118 = arith.subi %106, %cst_4 : tensor<32xi32, #blocked>
    %119 = arith.maxsi %118, %cst_8 : tensor<32xi32, #blocked>
    %120 = ttg.convert_layout %119 : tensor<32xi32, #blocked> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %121 = tt.expand_dims %120 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %122 = ttg.convert_layout %121 : tensor<32x1xi32, #blocked4> -> tensor<32x1xi32, #blocked5>
    %123 = arith.extsi %122 : tensor<32x1xi32, #blocked5> to tensor<32x1xi64, #blocked5>
    %124 = arith.muli %123, %83 : tensor<32x1xi64, #blocked5>
    %125 = arith.addi %124, %88 : tensor<32x1xi64, #blocked5>
    %126 = tt.addptr %85, %125 : tensor<32x1x!tt.ptr<f8E5M2>, #blocked5>, tensor<32x1xi64, #blocked5>
    %127 = tt.broadcast %126 : tensor<32x1x!tt.ptr<f8E5M2>, #blocked5> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked5>
    %128 = ttg.convert_layout %127 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked5> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked1>
    %129 = tt.addptr %128, %96 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<32x128xi32, #blocked1>
    %130 = ttg.convert_layout %129 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked1> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked6>
    %131 = ttg.convert_layout %117 : tensor<32x128xi1, #blocked1> -> tensor<32x128xi1, #blocked6>
    %132 = ttg.convert_layout %cst_3 : tensor<32x128xf8E5M2, #blocked1> -> tensor<32x128xf8E5M2, #blocked6>
    %133 = tt.load %130, %131, %132 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked6>
    %134 = ttg.convert_layout %133 : tensor<32x128xf8E5M2, #blocked6> -> tensor<32x128xf8E5M2, #blocked1>
    %135 = tt.fp_to_fp %134 : tensor<32x128xf8E5M2, #blocked1> -> tensor<32x128xf32, #blocked1>
    %136 = arith.mulf %135, %104 : tensor<32x128xf32, #blocked1>
    %137 = arith.addf %105, %136 : tensor<32x128xf32, #blocked1>
    %138 = tt.extern_elementwise %110 {libname = "", libpath = "", pure = true, symbol = "__nv_ffs"} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    %139 = arith.cmpi ne, %138, %cst_8 : tensor<32xi32, #blocked>
    %140 = ttg.convert_layout %139 : tensor<32xi1, #blocked> -> tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %141 = tt.expand_dims %140 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi1, #blocked4>
    %142 = ttg.convert_layout %141 : tensor<32x1xi1, #blocked4> -> tensor<32x1xi1, #blocked5>
    %143 = tt.broadcast %142 : tensor<32x1xi1, #blocked5> -> tensor<32x128xi1, #blocked5>
    %144 = ttg.convert_layout %143 : tensor<32x128xi1, #blocked5> -> tensor<32x128xi1, #blocked1>
    %145 = arith.andi %64, %144 : tensor<32x128xi1, #blocked1>
    %146 = arith.subi %138, %cst_4 : tensor<32xi32, #blocked>
    %147 = arith.maxsi %146, %cst_8 : tensor<32xi32, #blocked>
    %148 = ttg.convert_layout %147 : tensor<32xi32, #blocked> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %149 = tt.expand_dims %148 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<32x1xi32, #blocked4>
    %150 = ttg.convert_layout %149 : tensor<32x1xi32, #blocked4> -> tensor<32x1xi32, #blocked5>
    %151 = arith.extsi %150 : tensor<32x1xi32, #blocked5> to tensor<32x1xi64, #blocked5>
    %152 = arith.muli %151, %83 : tensor<32x1xi64, #blocked5>
    %153 = arith.addi %152, %88 : tensor<32x1xi64, #blocked5>
    %154 = tt.addptr %85, %153 : tensor<32x1x!tt.ptr<f8E5M2>, #blocked5>, tensor<32x1xi64, #blocked5>
    %155 = tt.broadcast %154 : tensor<32x1x!tt.ptr<f8E5M2>, #blocked5> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked5>
    %156 = ttg.convert_layout %155 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked5> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked1>
    %157 = tt.addptr %156, %96 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<32x128xi32, #blocked1>
    %158 = ttg.convert_layout %157 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked1> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked6>
    %159 = ttg.convert_layout %145 : tensor<32x128xi1, #blocked1> -> tensor<32x128xi1, #blocked6>
    %160 = ttg.convert_layout %cst_3 : tensor<32x128xf8E5M2, #blocked1> -> tensor<32x128xf8E5M2, #blocked6>
    %161 = tt.load %158, %159, %160 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked6>
    %162 = ttg.convert_layout %161 : tensor<32x128xf8E5M2, #blocked6> -> tensor<32x128xf8E5M2, #blocked1>
    %163 = tt.fp_to_fp %162 : tensor<32x128xf8E5M2, #blocked1> -> tensor<32x128xf32, #blocked1>
    %164 = arith.mulf %163, %104 : tensor<32x128xf32, #blocked1>
    %165 = arith.addf %137, %164 : tensor<32x128xf32, #blocked1>
    %166 = tt.splat %arg15 : i32 -> tensor<128xi32, #blocked>
    %167 = arith.cmpi slt, %55, %166 : tensor<128xi32, #blocked>
    %168 = tt.load %arg17 : !tt.ptr<f32>
    %169 = arith.divf %cst, %168 : f32
    %170 = tt.reshape %165 allow_reorder : tensor<32x128xf32, #blocked1> -> tensor<4096xf32, #blocked>
    %171 = "tt.reduce"(%170) <{axis = 0 : i32}> ({
    ^bb0(%arg21: f32, %arg22: f32):
      %204 = tt.elementwise_inline_asm "{\0A    max.NaN.xorsign.abs.f32 $0, $1, $2;\0A    }" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %arg21, %arg22 : f32, f32 -> f32
      tt.reduce.return %204 : f32
    }) : (tensor<4096xf32, #blocked>) -> f32
    %172 = tt.bitcast %171 : f32 -> i32
    %173 = arith.andi %172, %c2147483647_i32 : i32
    %174 = arith.minui %173, %c2139095040_i32 : i32
    %175 = tt.bitcast %174 : i32 -> f32
    %176 = tt.bitcast %c932333861_i32 : i32 -> f32
    %177 = math.fma %175, %176, %cst_0 : f32
    %178 = tt.bitcast %177 : f32 -> i32
    %179 = tt.bitcast %arg18 : !tt.ptr<f32> -> !tt.ptr<i32>
    %180 = arith.shrui %178, %c31_i32 : i32
    %181 = arith.cmpi ne, %180, %c0_i32 : i32
    %182 = arith.cmpi eq, %180, %c0_i32 : i32
    %183 = tt.atomic_rmw max, relaxed, gpu, %179, %178, %182 : (!tt.ptr<i32>, i32, i1) -> i32
    %184 = tt.atomic_rmw umin, relaxed, gpu, %179, %178, %181 : (!tt.ptr<i32>, i32, i1) -> i32
    %185 = tt.splat %169 : f32 -> tensor<32x128xf32, #blocked1>
    %186 = arith.mulf %165, %185 : tensor<32x128xf32, #blocked1>
    %187 = tt.clampf %186, %cst_1, %cst_2, propagateNan = none : tensor<32x128xf32, #blocked1>
    %188 = tt.splat %arg6 : i64 -> tensor<32x1xi64, #blocked5>
    %189 = arith.muli %86, %188 : tensor<32x1xi64, #blocked5>
    %190 = tt.splat %arg5 : !tt.ptr<f8E5M2> -> tensor<32x1x!tt.ptr<f8E5M2>, #blocked5>
    %191 = tt.addptr %190, %189 : tensor<32x1x!tt.ptr<f8E5M2>, #blocked5>, tensor<32x1xi64, #blocked5>
    %192 = tt.broadcast %191 : tensor<32x1x!tt.ptr<f8E5M2>, #blocked5> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked5>
    %193 = ttg.convert_layout %192 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked5> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked1>
    %194 = tt.addptr %193, %96 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<32x128xi32, #blocked1>
    %195 = ttg.convert_layout %167 : tensor<128xi1, #blocked> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %196 = tt.expand_dims %195 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi1, #blocked3>
    %197 = ttg.convert_layout %196 : tensor<1x128xi1, #blocked3> -> tensor<1x128xi1, #blocked1>
    %198 = tt.broadcast %197 : tensor<1x128xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
    %199 = arith.andi %62, %198 : tensor<32x128xi1, #blocked1>
    %200 = tt.fp_to_fp %187, rounding = rtne : tensor<32x128xf32, #blocked1> -> tensor<32x128xf8E5M2, #blocked1>
    %201 = ttg.convert_layout %194 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked1> -> tensor<32x128x!tt.ptr<f8E5M2>, #blocked6>
    %202 = ttg.convert_layout %200 : tensor<32x128xf8E5M2, #blocked1> -> tensor<32x128xf8E5M2, #blocked6>
    %203 = ttg.convert_layout %199 : tensor<32x128xi1, #blocked1> -> tensor<32x128xi1, #blocked6>
    tt.store %201, %202, %203 : tensor<32x128x!tt.ptr<f8E5M2>, #blocked6>
    tt.return
  }
}

// -----


// Preserve the complete three-load FP8, tensor-scale, and packed-assembly
// protocol without converting either load or introducing an extra conversion.
// Exact original TTIR SHA-256: 522164867ac21eb2dc910c8566ef60fb0fa36b9e5a09363b03d8bec0916c6520.
//
// BASELINE-LABEL: @production_three_load_packed_mx_layout
// BASELINE-COUNT-3: tt.load
// BASELINE: tt.join
// BASELINE-COUNT-2: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_three_load_packed_mx_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-NOT: tt.load
// OPTIMIZED: tt.join
// OPTIMIZED-COUNT-2: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 2, 16], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [0, 1, 2]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 1, 2], order = [0, 1, 2]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 1, 2], order = [2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked12 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked13 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked14 = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 16, 2], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked15 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [1, 2, 0]}>
#blocked16 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [1, 2, 2, 1], order = [3, 2, 1, 0]}>
#blocked17 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 16, 2], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked18 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 4, 8, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blocked19 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#blocked20 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked21 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [0, 1, 2]}>
#blocked22 = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 4], order = [1, 2, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @production_three_load_packed_mx_layout(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<3.38953139E+38> : tensor<1x64x8x16xf32, #blocked>
    %cst_0 = arith.constant dense<-3.38953139E+38> : tensor<1x64x8x16xf32, #blocked>
    %cst_1 = arith.constant dense<16> : tensor<1x64x64xi32, #blocked1>
    %cst_2 = arith.constant dense<65535> : tensor<1x64x64xi32, #blocked1>
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i64 = arith.constant 8 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_num_programs x : i32
    %3 = arith.cmpi ne, %arg14, %2 : i32
    %4:3 = scf.if %3 -> (i64, i64, i64) {
      %171 = arith.muli %arg13, %arg14 : i32
      %172 = arith.extsi %171 : i32 to i64
      %173 = arith.divsi %1, %172 : i64
      %174 = arith.remsi %1, %172 : i64
      %175 = arith.extsi %arg14 : i32 to i64
      %176 = arith.divsi %174, %175 : i64
      %177 = arith.remsi %1, %175 : i64
      scf.yield %177, %173, %176 : i64, i64, i64
    } else {
      %171 = tt.get_program_id y : i32
      %172 = arith.extsi %171 : i32 to i64
      %173 = tt.get_program_id z : i32
      %174 = arith.extsi %173 : i32 to i64
      scf.yield %1, %174, %172 : i64, i64, i64
    }
    %5 = arith.muli %4#2, %c64_i64 : i64
    %6 = arith.muli %4#0, %c128_i64 : i64
    %7 = arith.muli %4#0, %c64_i64 : i64
    %8 = arith.muli %4#0, %c8_i64 : i64
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked2>
    %10 = ttg.convert_layout %9 : tensor<64xi32, #blocked2> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
    %12 = ttg.convert_layout %11 : tensor<1x64xi32, #blocked3> -> tensor<1x64xi32, #blocked4>
    %13 = ttg.convert_layout %12 : tensor<1x64xi32, #blocked4> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked5}>>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked5}>> -> tensor<1x64x1xi32, #blocked5>
    %15 = ttg.convert_layout %14 : tensor<1x64x1xi32, #blocked5> -> tensor<1x64x1xi32, #blocked6>
    %16 = arith.extsi %15 : tensor<1x64x1xi32, #blocked6> to tensor<1x64x1xi64, #blocked6>
    %17 = arith.cmpi slt, %4#1, %c1_i64 : i64
    %18 = tt.splat %5 : i64 -> tensor<1x64x1xi64, #blocked6>
    %19 = arith.addi %18, %16 : tensor<1x64x1xi64, #blocked6>
    %20 = arith.extsi %arg9 : i32 to i64
    %21 = tt.splat %20 : i64 -> tensor<1x64x1xi64, #blocked6>
    %22 = arith.cmpi slt, %19, %21 : tensor<1x64x1xi64, #blocked6>
    %23 = ttg.convert_layout %12 : tensor<1x64xi32, #blocked4> -> tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked7}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked7}>> -> tensor<1x1x64xi32, #blocked7>
    %25 = ttg.convert_layout %24 : tensor<1x1x64xi32, #blocked7> -> tensor<1x1x64xi32, #blocked8>
    %26 = arith.extsi %25 : tensor<1x1x64xi32, #blocked8> to tensor<1x1x64xi64, #blocked8>
    %27 = arith.extsi %arg4 : i32 to i64
    %28 = arith.muli %4#1, %27 : i64
    %29 = arith.extsi %arg5 : i32 to i64
    %30 = arith.muli %5, %29 : i64
    %31 = arith.addi %30, %7 : i64
    %32 = arith.addi %28, %31 : i64
    %33 = tt.addptr %arg3, %32 : !tt.ptr<i8>, i64
    %34 = tt.splat %7 : i64 -> tensor<1x1x64xi64, #blocked8>
    %35 = arith.addi %34, %26 : tensor<1x1x64xi64, #blocked8>
    %36 = arith.divsi %arg10, %c2_i32 : i32
    %37 = arith.extsi %36 : i32 to i64
    %38 = tt.splat %37 : i64 -> tensor<1x1x64xi64, #blocked8>
    %39 = arith.cmpi slt, %35, %38 : tensor<1x1x64xi64, #blocked8>
    %40 = tt.splat %17 : i1 -> tensor<1x64x1xi1, #blocked6>
    %41 = arith.andi %40, %22 : tensor<1x64x1xi1, #blocked6>
    %42 = tt.broadcast %41 : tensor<1x64x1xi1, #blocked6> -> tensor<1x64x64xi1, #blocked6>
    %43 = ttg.convert_layout %42 : tensor<1x64x64xi1, #blocked6> -> tensor<1x64x64xi1, #blocked1>
    %44 = tt.broadcast %39 : tensor<1x1x64xi1, #blocked8> -> tensor<1x64x64xi1, #blocked8>
    %45 = ttg.convert_layout %44 : tensor<1x64x64xi1, #blocked8> -> tensor<1x64x64xi1, #blocked1>
    %46 = arith.andi %43, %45 : tensor<1x64x64xi1, #blocked1>
    %47 = tt.splat %29 : i64 -> tensor<1x64x1xi64, #blocked6>
    %48 = arith.muli %16, %47 : tensor<1x64x1xi64, #blocked6>
    %49 = tt.broadcast %48 : tensor<1x64x1xi64, #blocked6> -> tensor<1x64x64xi64, #blocked6>
    %50 = ttg.convert_layout %49 : tensor<1x64x64xi64, #blocked6> -> tensor<1x64x64xi64, #blocked1>
    %51 = tt.broadcast %26 : tensor<1x1x64xi64, #blocked8> -> tensor<1x64x64xi64, #blocked8>
    %52 = ttg.convert_layout %51 : tensor<1x64x64xi64, #blocked8> -> tensor<1x64x64xi64, #blocked1>
    %53 = arith.addi %50, %52 : tensor<1x64x64xi64, #blocked1>
    %54 = tt.splat %33 : !tt.ptr<i8> -> tensor<1x64x64x!tt.ptr<i8>, #blocked1>
    %55 = tt.addptr %54, %53 : tensor<1x64x64x!tt.ptr<i8>, #blocked1>, tensor<1x64x64xi64, #blocked1>
    %56 = ttg.convert_layout %55 : tensor<1x64x64x!tt.ptr<i8>, #blocked1> -> tensor<1x64x64x!tt.ptr<i8>, #blocked9>
    %57 = ttg.convert_layout %46 : tensor<1x64x64xi1, #blocked1> -> tensor<1x64x64xi1, #blocked9>
    %58 = tt.load %56, %57 : tensor<1x64x64x!tt.ptr<i8>, #blocked9>
    %59 = ttg.convert_layout %58 : tensor<1x64x64xi8, #blocked9> -> tensor<1x64x64xi8, #blocked1>
    %60 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked2>
    %61 = ttg.convert_layout %60 : tensor<8xi32, #blocked2> -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %62 = tt.expand_dims %61 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x8xi32, #blocked3>
    %63 = ttg.convert_layout %62 : tensor<1x8xi32, #blocked3> -> tensor<1x8xi32, #blocked10>
    %64 = ttg.convert_layout %63 : tensor<1x8xi32, #blocked10> -> tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #blocked11}>>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #blocked11}>> -> tensor<1x1x8xi32, #blocked11>
    %66 = ttg.convert_layout %65 : tensor<1x1x8xi32, #blocked11> -> tensor<1x1x8xi32, #blocked12>
    %67 = arith.extsi %66 : tensor<1x1x8xi32, #blocked12> to tensor<1x1x8xi64, #blocked12>
    %68 = arith.extsi %arg7 : i32 to i64
    %69 = arith.muli %4#1, %68 : i64
    %70 = arith.extsi %arg8 : i32 to i64
    %71 = arith.muli %5, %70 : i64
    %72 = arith.addi %71, %8 : i64
    %73 = arith.addi %69, %72 : i64
    %74 = tt.addptr %arg6, %73 : !tt.ptr<f8E4M3FN>, i64
    %75 = tt.splat %8 : i64 -> tensor<1x1x8xi64, #blocked12>
    %76 = arith.addi %75, %67 : tensor<1x1x8xi64, #blocked12>
    %77 = arith.addi %arg10, %c15_i32 : i32
    %78 = arith.divsi %77, %c16_i32 : i32
    %79 = arith.extsi %78 : i32 to i64
    %80 = tt.splat %79 : i64 -> tensor<1x1x8xi64, #blocked12>
    %81 = arith.cmpi slt, %76, %80 : tensor<1x1x8xi64, #blocked12>
    %82 = tt.broadcast %41 : tensor<1x64x1xi1, #blocked6> -> tensor<1x64x8xi1, #blocked6>
    %83 = ttg.convert_layout %82 : tensor<1x64x8xi1, #blocked6> -> tensor<1x64x8xi1, #blocked13>
    %84 = tt.broadcast %81 : tensor<1x1x8xi1, #blocked12> -> tensor<1x64x8xi1, #blocked12>
    %85 = ttg.convert_layout %84 : tensor<1x64x8xi1, #blocked12> -> tensor<1x64x8xi1, #blocked13>
    %86 = arith.andi %83, %85 : tensor<1x64x8xi1, #blocked13>
    %87 = tt.splat %70 : i64 -> tensor<1x64x1xi64, #blocked6>
    %88 = arith.muli %16, %87 : tensor<1x64x1xi64, #blocked6>
    %89 = tt.broadcast %88 : tensor<1x64x1xi64, #blocked6> -> tensor<1x64x8xi64, #blocked6>
    %90 = ttg.convert_layout %89 : tensor<1x64x8xi64, #blocked6> -> tensor<1x64x8xi64, #blocked13>
    %91 = tt.broadcast %67 : tensor<1x1x8xi64, #blocked12> -> tensor<1x64x8xi64, #blocked12>
    %92 = ttg.convert_layout %91 : tensor<1x64x8xi64, #blocked12> -> tensor<1x64x8xi64, #blocked13>
    %93 = arith.addi %90, %92 : tensor<1x64x8xi64, #blocked13>
    %94 = tt.splat %74 : !tt.ptr<f8E4M3FN> -> tensor<1x64x8x!tt.ptr<f8E4M3FN>, #blocked13>
    %95 = tt.addptr %94, %93 : tensor<1x64x8x!tt.ptr<f8E4M3FN>, #blocked13>, tensor<1x64x8xi64, #blocked13>
    %96 = ttg.convert_layout %95 : tensor<1x64x8x!tt.ptr<f8E4M3FN>, #blocked13> -> tensor<1x64x8x!tt.ptr<f8E4M3FN>, #blocked14>
    %97 = ttg.convert_layout %86 : tensor<1x64x8xi1, #blocked13> -> tensor<1x64x8xi1, #blocked14>
    %98 = tt.load %96, %97 : tensor<1x64x8x!tt.ptr<f8E4M3FN>, #blocked14>
    %99 = ttg.convert_layout %98 : tensor<1x64x8xf8E4M3FN, #blocked14> -> tensor<1x64x8xf8E4M3FN, #blocked13>
    %100 = arith.extsi %arg12 : i32 to i64
    %101 = arith.muli %4#1, %100 : i64
    %102 = tt.addptr %arg11, %101 : !tt.ptr<f32>, i64
    %103 = tt.splat %102 : !tt.ptr<f32> -> tensor<1x64x1x!tt.ptr<f32>, #blocked6>
    %104 = tt.addptr %103, %19 : tensor<1x64x1x!tt.ptr<f32>, #blocked6>, tensor<1x64x1xi64, #blocked6>
    %105 = ttg.convert_layout %104 : tensor<1x64x1x!tt.ptr<f32>, #blocked6> -> tensor<1x64x1x!tt.ptr<f32>, #blocked15>
    %106 = ttg.convert_layout %41 : tensor<1x64x1xi1, #blocked6> -> tensor<1x64x1xi1, #blocked15>
    %107 = tt.load %105, %106 : tensor<1x64x1x!tt.ptr<f32>, #blocked15>
    %108 = ttg.convert_layout %107 : tensor<1x64x1xf32, #blocked15> -> tensor<1x64x1xf32, #blocked6>
    %109 = tt.fp_to_fp %99 : tensor<1x64x8xf8E4M3FN, #blocked13> -> tensor<1x64x8xf32, #blocked13>
    %110 = tt.broadcast %108 : tensor<1x64x1xf32, #blocked6> -> tensor<1x64x8xf32, #blocked6>
    %111 = ttg.convert_layout %110 : tensor<1x64x8xf32, #blocked6> -> tensor<1x64x8xf32, #blocked13>
    %112 = arith.mulf %109, %111 : tensor<1x64x8xf32, #blocked13>
    %113 = arith.truncf %112 : tensor<1x64x8xf32, #blocked13> to tensor<1x64x8xbf16, #blocked13>
    %114 = tt.elementwise_inline_asm "\0A            {\0A            .reg .b8 in_8;\0A            .reg .f16x2 out;\0A            cvt.u8.u32 in_8, $1;\0A            cvt.rn.f16x2.e2m1x2 out, in_8;\0A            mov.b32 $0, out;\0A            }\0A            " {constraints = "=r,r", packed_element = 1 : i32, pure = true} %59 : tensor<1x64x64xi8, #blocked1> -> tensor<1x64x64xi32, #blocked1>
    %115 = arith.andi %114, %cst_2 : tensor<1x64x64xi32, #blocked1>
    %116 = arith.trunci %115 : tensor<1x64x64xi32, #blocked1> to tensor<1x64x64xi16, #blocked1>
    %117 = arith.shrui %114, %cst_1 : tensor<1x64x64xi32, #blocked1>
    %118 = arith.trunci %117 : tensor<1x64x64xi32, #blocked1> to tensor<1x64x64xi16, #blocked1>
    %119 = tt.bitcast %116 : tensor<1x64x64xi16, #blocked1> -> tensor<1x64x64xf16, #blocked1>
    %120 = tt.bitcast %118 : tensor<1x64x64xi16, #blocked1> -> tensor<1x64x64xf16, #blocked1>
    %121 = arith.extf %119 : tensor<1x64x64xf16, #blocked1> to tensor<1x64x64xf32, #blocked1>
    %122 = arith.truncf %121 : tensor<1x64x64xf32, #blocked1> to tensor<1x64x64xbf16, #blocked1>
    %123 = arith.extf %120 : tensor<1x64x64xf16, #blocked1> to tensor<1x64x64xf32, #blocked1>
    %124 = arith.truncf %123 : tensor<1x64x64xf32, #blocked1> to tensor<1x64x64xbf16, #blocked1>
    %125 = tt.join %122, %124 : tensor<1x64x64xbf16, #blocked1> -> tensor<1x64x64x2xbf16, #blocked16>
    %126 = ttg.convert_layout %125 : tensor<1x64x64x2xbf16, #blocked16> -> tensor<1x64x64x2xbf16, #blocked17>
    %127 = tt.reshape %126 : tensor<1x64x64x2xbf16, #blocked17> -> tensor<1x64x8x16xbf16, #blocked>
    %128 = tt.reshape %113 : tensor<1x64x8xbf16, #blocked13> -> tensor<1x64x8x1xbf16, #blocked18>
    %129 = tt.broadcast %128 : tensor<1x64x8x1xbf16, #blocked18> -> tensor<1x64x8x16xbf16, #blocked18>
    %130 = ttg.convert_layout %129 : tensor<1x64x8x16xbf16, #blocked18> -> tensor<1x64x8x16xbf16, #blocked>
    %131 = arith.mulf %127, %130 : tensor<1x64x8x16xbf16, #blocked>
    %132 = arith.extf %131 : tensor<1x64x8x16xbf16, #blocked> to tensor<1x64x8x16xf32, #blocked>
    %133 = tt.clampf %132, %cst_0, %cst, propagateNan = all : tensor<1x64x8x16xf32, #blocked>
    %134 = tt.reshape %133 : tensor<1x64x8x16xf32, #blocked> -> tensor<1x64x128xf32, #blocked19>
    %135 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %136 = ttg.convert_layout %135 : tensor<128xi32, #blocked2> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %137 = tt.expand_dims %136 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi32, #blocked3>
    %138 = ttg.convert_layout %137 : tensor<1x128xi32, #blocked3> -> tensor<1x128xi32, #blocked20>
    %139 = ttg.convert_layout %138 : tensor<1x128xi32, #blocked20> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked21}>>
    %140 = tt.expand_dims %139 {axis = 1 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked21}>> -> tensor<1x1x128xi32, #blocked21>
    %141 = ttg.convert_layout %140 : tensor<1x1x128xi32, #blocked21> -> tensor<1x1x128xi32, #blocked19>
    %142 = arith.extsi %141 : tensor<1x1x128xi32, #blocked19> to tensor<1x1x128xi64, #blocked19>
    %143 = arith.extsi %arg1 : i32 to i64
    %144 = arith.muli %4#1, %143 : i64
    %145 = arith.extsi %arg2 : i32 to i64
    %146 = arith.muli %6, %145 : i64
    %147 = arith.addi %5, %146 : i64
    %148 = arith.addi %144, %147 : i64
    %149 = tt.addptr %arg0, %148 : !tt.ptr<bf16>, i64
    %150 = tt.splat %6 : i64 -> tensor<1x1x128xi64, #blocked19>
    %151 = arith.addi %150, %142 : tensor<1x1x128xi64, #blocked19>
    %152 = arith.extsi %arg10 : i32 to i64
    %153 = tt.splat %152 : i64 -> tensor<1x1x128xi64, #blocked19>
    %154 = arith.cmpi slt, %151, %153 : tensor<1x1x128xi64, #blocked19>
    %155 = tt.broadcast %41 : tensor<1x64x1xi1, #blocked6> -> tensor<1x64x128xi1, #blocked6>
    %156 = ttg.convert_layout %155 : tensor<1x64x128xi1, #blocked6> -> tensor<1x64x128xi1, #blocked19>
    %157 = tt.broadcast %154 : tensor<1x1x128xi1, #blocked19> -> tensor<1x64x128xi1, #blocked19>
    %158 = arith.andi %156, %157 : tensor<1x64x128xi1, #blocked19>
    %159 = tt.splat %145 : i64 -> tensor<1x1x128xi64, #blocked19>
    %160 = arith.muli %142, %159 : tensor<1x1x128xi64, #blocked19>
    %161 = tt.broadcast %16 : tensor<1x64x1xi64, #blocked6> -> tensor<1x64x128xi64, #blocked6>
    %162 = ttg.convert_layout %161 : tensor<1x64x128xi64, #blocked6> -> tensor<1x64x128xi64, #blocked19>
    %163 = tt.broadcast %160 : tensor<1x1x128xi64, #blocked19> -> tensor<1x64x128xi64, #blocked19>
    %164 = arith.addi %162, %163 : tensor<1x64x128xi64, #blocked19>
    %165 = tt.splat %149 : !tt.ptr<bf16> -> tensor<1x64x128x!tt.ptr<bf16>, #blocked19>
    %166 = tt.addptr %165, %164 : tensor<1x64x128x!tt.ptr<bf16>, #blocked19>, tensor<1x64x128xi64, #blocked19>
    %167 = arith.truncf %134 : tensor<1x64x128xf32, #blocked19> to tensor<1x64x128xbf16, #blocked19>
    %168 = ttg.convert_layout %166 : tensor<1x64x128x!tt.ptr<bf16>, #blocked19> -> tensor<1x64x128x!tt.ptr<bf16>, #blocked22>
    %169 = ttg.convert_layout %167 : tensor<1x64x128xbf16, #blocked19> -> tensor<1x64x128xbf16, #blocked22>
    %170 = ttg.convert_layout %158 : tensor<1x64x128xi1, #blocked19> -> tensor<1x64x128xi1, #blocked22>
    tt.store %168, %169, %170 : tensor<1x64x128x!tt.ptr<bf16>, #blocked22>
    tt.return
  }
}


// -----

// The original FP8 tensor and i8 scale form a four-reshape packed component
// without a join; preserve its incumbent two-conversion layout.
// Exact original TTIR SHA-256: 0a0d8de1f9009e9d6543cd76773ac884486cf388b8f247b38015063b8421acb6.
//
// BASELINE-LABEL: @production_two_load_fp8_scale_layout
// BASELINE: tt.load
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.load
// BASELINE-COUNT-2: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_two_load_fp8_scale_layout
// OPTIMIZED: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.load
// OPTIMIZED-COUNT-2: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 8, 4, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [0, 1, 2]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [0, 1, 2]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked12 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked13 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked14 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked15 = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 4], order = [1, 2, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @production_two_load_fp8_scale_layout(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<7> : tensor<1x64x4xi16, #blocked>
    %cst_0 = arith.constant dense<3.38953139E+38> : tensor<1x64x4x32xf32, #blocked1>
    %cst_1 = arith.constant dense<255> : tensor<1x64x4x1xi32, #blocked2>
    %cst_2 = arith.constant dense<0x7FC00000> : tensor<1x64x4x32xf32, #blocked1>
    %cst_3 = arith.constant dense<-3.38953139E+38> : tensor<1x64x4x32xf32, #blocked1>
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4_i64 = arith.constant 4 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_num_programs x : i32
    %3 = arith.cmpi ne, %arg14, %2 : i32
    %4:3 = scf.if %3 -> (i64, i64, i64) {
      %136 = arith.muli %arg13, %arg14 : i32
      %137 = arith.extsi %136 : i32 to i64
      %138 = arith.divsi %1, %137 : i64
      %139 = arith.remsi %1, %137 : i64
      %140 = arith.extsi %arg14 : i32 to i64
      %141 = arith.divsi %139, %140 : i64
      %142 = arith.remsi %1, %140 : i64
      scf.yield %142, %138, %141 : i64, i64, i64
    } else {
      %136 = tt.get_program_id y : i32
      %137 = arith.extsi %136 : i32 to i64
      %138 = tt.get_program_id z : i32
      %139 = arith.extsi %138 : i32 to i64
      scf.yield %1, %139, %137 : i64, i64, i64
    }
    %5 = arith.muli %4#2, %c64_i64 : i64
    %6 = arith.muli %4#0, %c128_i64 : i64
    %7 = arith.muli %4#0, %c4_i64 : i64
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %9 = ttg.convert_layout %8 : tensor<64xi32, #blocked3> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi32, #blocked4>
    %11 = ttg.convert_layout %10 : tensor<1x64xi32, #blocked4> -> tensor<1x64xi32, #blocked5>
    %12 = ttg.convert_layout %11 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked6}>>
    %13 = tt.expand_dims %12 {axis = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked6}>> -> tensor<1x64x1xi32, #blocked6>
    %14 = ttg.convert_layout %13 : tensor<1x64x1xi32, #blocked6> -> tensor<1x64x1xi32, #blocked7>
    %15 = arith.extsi %14 : tensor<1x64x1xi32, #blocked7> to tensor<1x64x1xi64, #blocked7>
    %16 = arith.cmpi slt, %4#1, %c1_i64 : i64
    %17 = tt.splat %5 : i64 -> tensor<1x64x1xi64, #blocked7>
    %18 = arith.addi %17, %15 : tensor<1x64x1xi64, #blocked7>
    %19 = arith.extsi %arg9 : i32 to i64
    %20 = tt.splat %19 : i64 -> tensor<1x64x1xi64, #blocked7>
    %21 = arith.cmpi slt, %18, %20 : tensor<1x64x1xi64, #blocked7>
    %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3>
    %23 = ttg.convert_layout %22 : tensor<128xi32, #blocked3> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi32, #blocked4>
    %25 = ttg.convert_layout %24 : tensor<1x128xi32, #blocked4> -> tensor<1x128xi32, #blocked8>
    %26 = ttg.convert_layout %25 : tensor<1x128xi32, #blocked8> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked9}>>
    %27 = tt.expand_dims %26 {axis = 1 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked9}>> -> tensor<1x1x128xi32, #blocked9>
    %28 = ttg.convert_layout %27 : tensor<1x1x128xi32, #blocked9> -> tensor<1x1x128xi32, #blocked10>
    %29 = arith.extsi %28 : tensor<1x1x128xi32, #blocked10> to tensor<1x1x128xi64, #blocked10>
    %30 = arith.extsi %arg4 : i32 to i64
    %31 = arith.muli %4#1, %30 : i64
    %32 = arith.extsi %arg5 : i32 to i64
    %33 = arith.muli %5, %32 : i64
    %34 = arith.addi %33, %6 : i64
    %35 = arith.addi %31, %34 : i64
    %36 = tt.addptr %arg3, %35 : !tt.ptr<f8E4M3FN>, i64
    %37 = tt.splat %6 : i64 -> tensor<1x1x128xi64, #blocked10>
    %38 = arith.addi %37, %29 : tensor<1x1x128xi64, #blocked10>
    %39 = arith.extsi %arg10 : i32 to i64
    %40 = tt.splat %39 : i64 -> tensor<1x1x128xi64, #blocked10>
    %41 = arith.cmpi slt, %38, %40 : tensor<1x1x128xi64, #blocked10>
    %42 = tt.splat %16 : i1 -> tensor<1x64x1xi1, #blocked7>
    %43 = arith.andi %42, %21 : tensor<1x64x1xi1, #blocked7>
    %44 = tt.broadcast %43 : tensor<1x64x1xi1, #blocked7> -> tensor<1x64x128xi1, #blocked7>
    %45 = ttg.convert_layout %44 : tensor<1x64x128xi1, #blocked7> -> tensor<1x64x128xi1, #blocked10>
    %46 = tt.broadcast %41 : tensor<1x1x128xi1, #blocked10> -> tensor<1x64x128xi1, #blocked10>
    %47 = arith.andi %45, %46 : tensor<1x64x128xi1, #blocked10>
    %48 = tt.splat %32 : i64 -> tensor<1x64x1xi64, #blocked7>
    %49 = arith.muli %15, %48 : tensor<1x64x1xi64, #blocked7>
    %50 = tt.broadcast %49 : tensor<1x64x1xi64, #blocked7> -> tensor<1x64x128xi64, #blocked7>
    %51 = ttg.convert_layout %50 : tensor<1x64x128xi64, #blocked7> -> tensor<1x64x128xi64, #blocked10>
    %52 = tt.broadcast %29 : tensor<1x1x128xi64, #blocked10> -> tensor<1x64x128xi64, #blocked10>
    %53 = arith.addi %51, %52 : tensor<1x64x128xi64, #blocked10>
    %54 = tt.splat %36 : !tt.ptr<f8E4M3FN> -> tensor<1x64x128x!tt.ptr<f8E4M3FN>, #blocked10>
    %55 = tt.addptr %54, %53 : tensor<1x64x128x!tt.ptr<f8E4M3FN>, #blocked10>, tensor<1x64x128xi64, #blocked10>
    %56 = ttg.convert_layout %55 : tensor<1x64x128x!tt.ptr<f8E4M3FN>, #blocked10> -> tensor<1x64x128x!tt.ptr<f8E4M3FN>, #blocked11>
    %57 = ttg.convert_layout %47 : tensor<1x64x128xi1, #blocked10> -> tensor<1x64x128xi1, #blocked11>
    %58 = tt.load %56, %57 : tensor<1x64x128x!tt.ptr<f8E4M3FN>, #blocked11>
    %59 = ttg.convert_layout %58 : tensor<1x64x128xf8E4M3FN, #blocked11> -> tensor<1x64x128xf8E4M3FN, #blocked10>
    %60 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked3>
    %61 = ttg.convert_layout %60 : tensor<4xi32, #blocked3> -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %62 = tt.expand_dims %61 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x4xi32, #blocked4>
    %63 = ttg.convert_layout %62 : tensor<1x4xi32, #blocked4> -> tensor<1x4xi32, #blocked12>
    %64 = ttg.convert_layout %63 : tensor<1x4xi32, #blocked12> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked13}>>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked13}>> -> tensor<1x1x4xi32, #blocked13>
    %66 = ttg.convert_layout %65 : tensor<1x1x4xi32, #blocked13> -> tensor<1x1x4xi32, #blocked14>
    %67 = arith.extsi %66 : tensor<1x1x4xi32, #blocked14> to tensor<1x1x4xi64, #blocked14>
    %68 = arith.extsi %arg7 : i32 to i64
    %69 = arith.muli %4#1, %68 : i64
    %70 = arith.extsi %arg8 : i32 to i64
    %71 = arith.muli %5, %70 : i64
    %72 = arith.addi %71, %7 : i64
    %73 = arith.addi %69, %72 : i64
    %74 = tt.addptr %arg6, %73 : !tt.ptr<i8>, i64
    %75 = tt.splat %7 : i64 -> tensor<1x1x4xi64, #blocked14>
    %76 = arith.addi %75, %67 : tensor<1x1x4xi64, #blocked14>
    %77 = arith.addi %arg10, %c31_i32 : i32
    %78 = arith.divsi %77, %c32_i32 : i32
    %79 = arith.extsi %78 : i32 to i64
    %80 = tt.splat %79 : i64 -> tensor<1x1x4xi64, #blocked14>
    %81 = arith.cmpi slt, %76, %80 : tensor<1x1x4xi64, #blocked14>
    %82 = tt.broadcast %43 : tensor<1x64x1xi1, #blocked7> -> tensor<1x64x4xi1, #blocked7>
    %83 = ttg.convert_layout %82 : tensor<1x64x4xi1, #blocked7> -> tensor<1x64x4xi1, #blocked>
    %84 = tt.broadcast %81 : tensor<1x1x4xi1, #blocked14> -> tensor<1x64x4xi1, #blocked14>
    %85 = ttg.convert_layout %84 : tensor<1x64x4xi1, #blocked14> -> tensor<1x64x4xi1, #blocked>
    %86 = arith.andi %83, %85 : tensor<1x64x4xi1, #blocked>
    %87 = tt.splat %70 : i64 -> tensor<1x64x1xi64, #blocked7>
    %88 = arith.muli %15, %87 : tensor<1x64x1xi64, #blocked7>
    %89 = tt.broadcast %88 : tensor<1x64x1xi64, #blocked7> -> tensor<1x64x4xi64, #blocked7>
    %90 = ttg.convert_layout %89 : tensor<1x64x4xi64, #blocked7> -> tensor<1x64x4xi64, #blocked>
    %91 = tt.broadcast %67 : tensor<1x1x4xi64, #blocked14> -> tensor<1x64x4xi64, #blocked14>
    %92 = ttg.convert_layout %91 : tensor<1x64x4xi64, #blocked14> -> tensor<1x64x4xi64, #blocked>
    %93 = arith.addi %90, %92 : tensor<1x64x4xi64, #blocked>
    %94 = tt.splat %74 : !tt.ptr<i8> -> tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %95 = tt.addptr %94, %93 : tensor<1x64x4x!tt.ptr<i8>, #blocked>, tensor<1x64x4xi64, #blocked>
    %96 = ttg.convert_layout %95 : tensor<1x64x4x!tt.ptr<i8>, #blocked> -> tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %97 = ttg.convert_layout %86 : tensor<1x64x4xi1, #blocked> -> tensor<1x64x4xi1, #blocked>
    %98 = tt.load %96, %97 : tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %99 = arith.extui %98 : tensor<1x64x4xi8, #blocked> to tensor<1x64x4xi16, #blocked>
    %100 = arith.shli %99, %cst : tensor<1x64x4xi16, #blocked>
    %101 = tt.bitcast %100 : tensor<1x64x4xi16, #blocked> -> tensor<1x64x4xbf16, #blocked>
    %102 = tt.fp_to_fp %59 : tensor<1x64x128xf8E4M3FN, #blocked10> -> tensor<1x64x128xbf16, #blocked10>
    %103 = tt.reshape %102 : tensor<1x64x128xbf16, #blocked10> -> tensor<1x64x4x32xbf16, #blocked1>
    %104 = tt.reshape %101 : tensor<1x64x4xbf16, #blocked> -> tensor<1x64x4x1xbf16, #blocked2>
    %105 = tt.reshape %98 : tensor<1x64x4xi8, #blocked> -> tensor<1x64x4x1xi8, #blocked2>
    %106 = tt.broadcast %104 : tensor<1x64x4x1xbf16, #blocked2> -> tensor<1x64x4x32xbf16, #blocked2>
    %107 = ttg.convert_layout %106 : tensor<1x64x4x32xbf16, #blocked2> -> tensor<1x64x4x32xbf16, #blocked1>
    %108 = arith.mulf %103, %107 : tensor<1x64x4x32xbf16, #blocked1>
    %109 = arith.extf %108 : tensor<1x64x4x32xbf16, #blocked1> to tensor<1x64x4x32xf32, #blocked1>
    %110 = tt.clampf %109, %cst_3, %cst_0, propagateNan = all : tensor<1x64x4x32xf32, #blocked1>
    %111 = arith.extui %105 : tensor<1x64x4x1xi8, #blocked2> to tensor<1x64x4x1xi32, #blocked2>
    %112 = arith.cmpi eq, %111, %cst_1 : tensor<1x64x4x1xi32, #blocked2>
    %113 = tt.broadcast %112 : tensor<1x64x4x1xi1, #blocked2> -> tensor<1x64x4x32xi1, #blocked2>
    %114 = ttg.convert_layout %113 : tensor<1x64x4x32xi1, #blocked2> -> tensor<1x64x4x32xi1, #blocked1>
    %115 = arith.select %114, %cst_2, %110 : tensor<1x64x4x32xi1, #blocked1>, tensor<1x64x4x32xf32, #blocked1>
    %116 = tt.reshape %115 : tensor<1x64x4x32xf32, #blocked1> -> tensor<1x64x128xf32, #blocked10>
    %117 = arith.extsi %arg1 : i32 to i64
    %118 = arith.muli %4#1, %117 : i64
    %119 = arith.extsi %arg2 : i32 to i64
    %120 = arith.muli %6, %119 : i64
    %121 = arith.addi %5, %120 : i64
    %122 = arith.addi %118, %121 : i64
    %123 = tt.addptr %arg0, %122 : !tt.ptr<bf16>, i64
    %124 = tt.splat %119 : i64 -> tensor<1x1x128xi64, #blocked10>
    %125 = arith.muli %29, %124 : tensor<1x1x128xi64, #blocked10>
    %126 = tt.broadcast %15 : tensor<1x64x1xi64, #blocked7> -> tensor<1x64x128xi64, #blocked7>
    %127 = ttg.convert_layout %126 : tensor<1x64x128xi64, #blocked7> -> tensor<1x64x128xi64, #blocked10>
    %128 = tt.broadcast %125 : tensor<1x1x128xi64, #blocked10> -> tensor<1x64x128xi64, #blocked10>
    %129 = arith.addi %127, %128 : tensor<1x64x128xi64, #blocked10>
    %130 = tt.splat %123 : !tt.ptr<bf16> -> tensor<1x64x128x!tt.ptr<bf16>, #blocked10>
    %131 = tt.addptr %130, %129 : tensor<1x64x128x!tt.ptr<bf16>, #blocked10>, tensor<1x64x128xi64, #blocked10>
    %132 = arith.truncf %116 : tensor<1x64x128xf32, #blocked10> to tensor<1x64x128xbf16, #blocked10>
    %133 = ttg.convert_layout %131 : tensor<1x64x128x!tt.ptr<bf16>, #blocked10> -> tensor<1x64x128x!tt.ptr<bf16>, #blocked15>
    %134 = ttg.convert_layout %132 : tensor<1x64x128xbf16, #blocked10> -> tensor<1x64x128xbf16, #blocked15>
    %135 = ttg.convert_layout %47 : tensor<1x64x128xi1, #blocked10> -> tensor<1x64x128xi1, #blocked15>
    tt.store %133, %134, %135 : tensor<1x64x128x!tt.ptr<bf16>, #blocked15>
    tt.return
  }
}

// -----

// Preserve the independent load/store address and mask slices in the exact
// original ragged FP4 MX upcast. The packed conversion must remain below the
// join; no conversion may be recreated in the ragged memory protocol.
// Exact original TTIR SHA-256:
// 6fe7b71610e45a93a2627c9199b684430280b2208f7e30b3cf199b8c133150ac.
//
// BASELINE-LABEL: @production_ragged_packed_mx_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE-COUNT-2: tt.load
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.join
// BASELINE-COUNT-2: ttg.convert_layout
// BASELINE-NOT: ttg.convert_layout
// BASELINE: tt.store
// BASELINE: tt.return
//
// OPTIMIZED-LABEL: @production_ragged_packed_mx_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED-COUNT-2: tt.load
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.join
// OPTIMIZED-COUNT-2: ttg.convert_layout
// OPTIMIZED-NOT: ttg.convert_layout
// OPTIMIZED: tt.store
// OPTIMIZED: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 8, 4, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [0, 1, 2]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 1, 2], order = [0, 1, 2]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 1, 2], order = [2, 1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked12 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked13 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked14 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [1, 2, 2, 1], order = [3, 2, 1, 0]}>
#blocked15 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 16, 2], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked16 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#blocked17 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked18 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [0, 1, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @production_ragged_packed_mx_layout(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32) attributes {noinline = false} {
    %cst = arith.constant dense<7> : tensor<1x64x4xi16, #blocked>
    %cst_0 = arith.constant dense<3.38953139E+38> : tensor<1x64x4x32xf32, #blocked1>
    %cst_1 = arith.constant dense<255> : tensor<1x64x4x1xi32, #blocked2>
    %cst_2 = arith.constant dense<0x7FC00000> : tensor<1x64x4x32xf32, #blocked1>
    %cst_3 = arith.constant dense<-3.38953139E+38> : tensor<1x64x4x32xf32, #blocked1>
    %cst_4 = arith.constant dense<16> : tensor<1x64x64xi32, #blocked3>
    %cst_5 = arith.constant dense<65535> : tensor<1x64x64xi32, #blocked3>
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i64 = arith.constant 0 : i64
    %c2_i32 = arith.constant 2 : i32
    %c4_i64 = arith.constant 4 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_num_programs x : i32
    %3 = arith.cmpi ne, %2, %c1_i32 : i32
    %4 = arith.select %3, %c0_i64, %1 : i64
    %5:2 = scf.if %3 -> (i64, i64) {
      %166 = arith.extsi %arg14 : i32 to i64
      %167 = arith.divsi %1, %166 : i64
      %168 = arith.remsi %1, %166 : i64
      scf.yield %167, %168 : i64, i64
    } else {
      %166 = tt.get_program_id y : i32
      %167 = arith.extsi %166 : i32 to i64
      %168 = tt.get_program_id z : i32
      %169 = arith.extsi %168 : i32 to i64
      scf.yield %169, %167 : i64, i64
    }
    %6 = arith.muli %5#1, %c64_i64 : i64
    %7 = arith.muli %4, %c128_i64 : i64
    %8 = arith.muli %4, %c64_i64 : i64
    %9 = arith.muli %4, %c4_i64 : i64
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked4>
    %11 = ttg.convert_layout %10 : tensor<64xi32, #blocked4> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %12 = tt.expand_dims %11 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x64xi32, #blocked5>
    %13 = ttg.convert_layout %12 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #blocked6>
    %14 = ttg.convert_layout %13 : tensor<1x64xi32, #blocked6> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked7}>>
    %15 = tt.expand_dims %14 {axis = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked7}>> -> tensor<1x64x1xi32, #blocked7>
    %16 = ttg.convert_layout %15 : tensor<1x64x1xi32, #blocked7> -> tensor<1x64x1xi32, #blocked8>
    %17 = arith.extsi %16 : tensor<1x64x1xi32, #blocked8> to tensor<1x64x1xi64, #blocked8>
    %18 = arith.extsi %arg9 : i32 to i64
    %19 = arith.cmpi slt, %5#0, %18 : i64
    %20 = tt.splat %6 : i64 -> tensor<1x64x1xi64, #blocked8>
    %21 = arith.addi %20, %17 : tensor<1x64x1xi64, #blocked8>
    %22 = arith.extsi %arg10 : i32 to i64
    %23 = tt.splat %22 : i64 -> tensor<1x64x1xi64, #blocked8>
    %24 = arith.cmpi slt, %21, %23 : tensor<1x64x1xi64, #blocked8>
    %25 = ttg.convert_layout %13 : tensor<1x64xi32, #blocked6> -> tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked9}>>
    %26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked9}>> -> tensor<1x1x64xi32, #blocked9>
    %27 = ttg.convert_layout %26 : tensor<1x1x64xi32, #blocked9> -> tensor<1x1x64xi32, #blocked10>
    %28 = arith.extsi %27 : tensor<1x1x64xi32, #blocked10> to tensor<1x1x64xi64, #blocked10>
    %29 = arith.extsi %arg4 : i32 to i64
    %30 = arith.muli %5#0, %29 : i64
    %31 = arith.extsi %arg5 : i32 to i64
    %32 = arith.muli %6, %31 : i64
    %33 = arith.addi %32, %8 : i64
    %34 = arith.addi %30, %33 : i64
    %35 = tt.addptr %arg3, %34 : !tt.ptr<i8>, i64
    %36 = tt.splat %8 : i64 -> tensor<1x1x64xi64, #blocked10>
    %37 = arith.addi %36, %28 : tensor<1x1x64xi64, #blocked10>
    %38 = arith.divsi %arg11, %c2_i32 : i32
    %39 = arith.extsi %38 : i32 to i64
    %40 = tt.splat %39 : i64 -> tensor<1x1x64xi64, #blocked10>
    %41 = arith.cmpi slt, %37, %40 : tensor<1x1x64xi64, #blocked10>
    %42 = tt.splat %19 : i1 -> tensor<1x64x1xi1, #blocked8>
    %43 = arith.andi %42, %24 : tensor<1x64x1xi1, #blocked8>
    %44 = tt.broadcast %43 : tensor<1x64x1xi1, #blocked8> -> tensor<1x64x64xi1, #blocked8>
    %45 = ttg.convert_layout %44 : tensor<1x64x64xi1, #blocked8> -> tensor<1x64x64xi1, #blocked3>
    %46 = tt.broadcast %41 : tensor<1x1x64xi1, #blocked10> -> tensor<1x64x64xi1, #blocked10>
    %47 = ttg.convert_layout %46 : tensor<1x64x64xi1, #blocked10> -> tensor<1x64x64xi1, #blocked3>
    %48 = arith.andi %45, %47 : tensor<1x64x64xi1, #blocked3>
    %49 = tt.splat %31 : i64 -> tensor<1x64x1xi64, #blocked8>
    %50 = arith.muli %17, %49 : tensor<1x64x1xi64, #blocked8>
    %51 = tt.broadcast %50 : tensor<1x64x1xi64, #blocked8> -> tensor<1x64x64xi64, #blocked8>
    %52 = ttg.convert_layout %51 : tensor<1x64x64xi64, #blocked8> -> tensor<1x64x64xi64, #blocked3>
    %53 = tt.broadcast %28 : tensor<1x1x64xi64, #blocked10> -> tensor<1x64x64xi64, #blocked10>
    %54 = ttg.convert_layout %53 : tensor<1x64x64xi64, #blocked10> -> tensor<1x64x64xi64, #blocked3>
    %55 = arith.addi %52, %54 : tensor<1x64x64xi64, #blocked3>
    %56 = tt.splat %35 : !tt.ptr<i8> -> tensor<1x64x64x!tt.ptr<i8>, #blocked3>
    %57 = tt.addptr %56, %55 : tensor<1x64x64x!tt.ptr<i8>, #blocked3>, tensor<1x64x64xi64, #blocked3>
    %58 = ttg.convert_layout %57 : tensor<1x64x64x!tt.ptr<i8>, #blocked3> -> tensor<1x64x64x!tt.ptr<i8>, #blocked3>
    %59 = ttg.convert_layout %48 : tensor<1x64x64xi1, #blocked3> -> tensor<1x64x64xi1, #blocked3>
    %60 = tt.load %58, %59 : tensor<1x64x64x!tt.ptr<i8>, #blocked3>
    %61 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked4>
    %62 = ttg.convert_layout %61 : tensor<4xi32, #blocked4> -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %63 = tt.expand_dims %62 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x4xi32, #blocked5>
    %64 = ttg.convert_layout %63 : tensor<1x4xi32, #blocked5> -> tensor<1x4xi32, #blocked11>
    %65 = ttg.convert_layout %64 : tensor<1x4xi32, #blocked11> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked12}>>
    %66 = tt.expand_dims %65 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked12}>> -> tensor<1x1x4xi32, #blocked12>
    %67 = ttg.convert_layout %66 : tensor<1x1x4xi32, #blocked12> -> tensor<1x1x4xi32, #blocked13>
    %68 = arith.extsi %67 : tensor<1x1x4xi32, #blocked13> to tensor<1x1x4xi64, #blocked13>
    %69 = arith.extsi %arg7 : i32 to i64
    %70 = arith.muli %5#0, %69 : i64
    %71 = arith.extsi %arg8 : i32 to i64
    %72 = arith.muli %6, %71 : i64
    %73 = arith.addi %72, %9 : i64
    %74 = arith.addi %70, %73 : i64
    %75 = tt.addptr %arg6, %74 : !tt.ptr<i8>, i64
    %76 = tt.splat %9 : i64 -> tensor<1x1x4xi64, #blocked13>
    %77 = arith.addi %76, %68 : tensor<1x1x4xi64, #blocked13>
    %78 = arith.addi %arg11, %c31_i32 : i32
    %79 = arith.divsi %78, %c32_i32 : i32
    %80 = arith.extsi %79 : i32 to i64
    %81 = tt.splat %80 : i64 -> tensor<1x1x4xi64, #blocked13>
    %82 = arith.cmpi slt, %77, %81 : tensor<1x1x4xi64, #blocked13>
    %83 = tt.broadcast %43 : tensor<1x64x1xi1, #blocked8> -> tensor<1x64x4xi1, #blocked8>
    %84 = ttg.convert_layout %83 : tensor<1x64x4xi1, #blocked8> -> tensor<1x64x4xi1, #blocked>
    %85 = tt.broadcast %82 : tensor<1x1x4xi1, #blocked13> -> tensor<1x64x4xi1, #blocked13>
    %86 = ttg.convert_layout %85 : tensor<1x64x4xi1, #blocked13> -> tensor<1x64x4xi1, #blocked>
    %87 = arith.andi %84, %86 : tensor<1x64x4xi1, #blocked>
    %88 = tt.splat %71 : i64 -> tensor<1x64x1xi64, #blocked8>
    %89 = arith.muli %17, %88 : tensor<1x64x1xi64, #blocked8>
    %90 = tt.broadcast %89 : tensor<1x64x1xi64, #blocked8> -> tensor<1x64x4xi64, #blocked8>
    %91 = ttg.convert_layout %90 : tensor<1x64x4xi64, #blocked8> -> tensor<1x64x4xi64, #blocked>
    %92 = tt.broadcast %68 : tensor<1x1x4xi64, #blocked13> -> tensor<1x64x4xi64, #blocked13>
    %93 = ttg.convert_layout %92 : tensor<1x64x4xi64, #blocked13> -> tensor<1x64x4xi64, #blocked>
    %94 = arith.addi %91, %93 : tensor<1x64x4xi64, #blocked>
    %95 = tt.splat %75 : !tt.ptr<i8> -> tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %96 = tt.addptr %95, %94 : tensor<1x64x4x!tt.ptr<i8>, #blocked>, tensor<1x64x4xi64, #blocked>
    %97 = ttg.convert_layout %96 : tensor<1x64x4x!tt.ptr<i8>, #blocked> -> tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %98 = ttg.convert_layout %87 : tensor<1x64x4xi1, #blocked> -> tensor<1x64x4xi1, #blocked>
    %99 = tt.load %97, %98 : tensor<1x64x4x!tt.ptr<i8>, #blocked>
    %100 = arith.extui %99 : tensor<1x64x4xi8, #blocked> to tensor<1x64x4xi16, #blocked>
    %101 = arith.shli %100, %cst : tensor<1x64x4xi16, #blocked>
    %102 = tt.bitcast %101 : tensor<1x64x4xi16, #blocked> -> tensor<1x64x4xbf16, #blocked>
    %103 = tt.elementwise_inline_asm "\0A            {\0A            .reg .b8 in_8;\0A            .reg .f16x2 out;\0A            cvt.u8.u32 in_8, $1;\0A            cvt.rn.f16x2.e2m1x2 out, in_8;\0A            mov.b32 $0, out;\0A            }\0A            " {constraints = "=r,r", packed_element = 1 : i32, pure = true} %60 : tensor<1x64x64xi8, #blocked3> -> tensor<1x64x64xi32, #blocked3>
    %104 = arith.andi %103, %cst_5 : tensor<1x64x64xi32, #blocked3>
    %105 = arith.trunci %104 : tensor<1x64x64xi32, #blocked3> to tensor<1x64x64xi16, #blocked3>
    %106 = arith.shrui %103, %cst_4 : tensor<1x64x64xi32, #blocked3>
    %107 = arith.trunci %106 : tensor<1x64x64xi32, #blocked3> to tensor<1x64x64xi16, #blocked3>
    %108 = tt.bitcast %105 : tensor<1x64x64xi16, #blocked3> -> tensor<1x64x64xf16, #blocked3>
    %109 = tt.bitcast %107 : tensor<1x64x64xi16, #blocked3> -> tensor<1x64x64xf16, #blocked3>
    %110 = arith.extf %108 : tensor<1x64x64xf16, #blocked3> to tensor<1x64x64xf32, #blocked3>
    %111 = arith.truncf %110 : tensor<1x64x64xf32, #blocked3> to tensor<1x64x64xbf16, #blocked3>
    %112 = arith.extf %109 : tensor<1x64x64xf16, #blocked3> to tensor<1x64x64xf32, #blocked3>
    %113 = arith.truncf %112 : tensor<1x64x64xf32, #blocked3> to tensor<1x64x64xbf16, #blocked3>
    %114 = tt.join %111, %113 : tensor<1x64x64xbf16, #blocked3> -> tensor<1x64x64x2xbf16, #blocked14>
    %115 = ttg.convert_layout %114 : tensor<1x64x64x2xbf16, #blocked14> -> tensor<1x64x64x2xbf16, #blocked15>
    %116 = tt.reshape %115 : tensor<1x64x64x2xbf16, #blocked15> -> tensor<1x64x4x32xbf16, #blocked1>
    %117 = tt.reshape %102 : tensor<1x64x4xbf16, #blocked> -> tensor<1x64x4x1xbf16, #blocked2>
    %118 = tt.reshape %99 : tensor<1x64x4xi8, #blocked> -> tensor<1x64x4x1xi8, #blocked2>
    %119 = tt.broadcast %117 : tensor<1x64x4x1xbf16, #blocked2> -> tensor<1x64x4x32xbf16, #blocked2>
    %120 = ttg.convert_layout %119 : tensor<1x64x4x32xbf16, #blocked2> -> tensor<1x64x4x32xbf16, #blocked1>
    %121 = arith.mulf %116, %120 : tensor<1x64x4x32xbf16, #blocked1>
    %122 = arith.extf %121 : tensor<1x64x4x32xbf16, #blocked1> to tensor<1x64x4x32xf32, #blocked1>
    %123 = tt.clampf %122, %cst_3, %cst_0, propagateNan = all : tensor<1x64x4x32xf32, #blocked1>
    %124 = arith.extui %118 : tensor<1x64x4x1xi8, #blocked2> to tensor<1x64x4x1xi32, #blocked2>
    %125 = arith.cmpi eq, %124, %cst_1 : tensor<1x64x4x1xi32, #blocked2>
    %126 = tt.broadcast %125 : tensor<1x64x4x1xi1, #blocked2> -> tensor<1x64x4x32xi1, #blocked2>
    %127 = ttg.convert_layout %126 : tensor<1x64x4x32xi1, #blocked2> -> tensor<1x64x4x32xi1, #blocked1>
    %128 = arith.select %127, %cst_2, %123 : tensor<1x64x4x32xi1, #blocked1>, tensor<1x64x4x32xf32, #blocked1>
    %129 = tt.reshape %128 : tensor<1x64x4x32xf32, #blocked1> -> tensor<1x64x128xf32, #blocked16>
    %130 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked4>
    %131 = ttg.convert_layout %130 : tensor<128xi32, #blocked4> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %132 = tt.expand_dims %131 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x128xi32, #blocked5>
    %133 = ttg.convert_layout %132 : tensor<1x128xi32, #blocked5> -> tensor<1x128xi32, #blocked17>
    %134 = ttg.convert_layout %133 : tensor<1x128xi32, #blocked17> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked18}>>
    %135 = tt.expand_dims %134 {axis = 1 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked18}>> -> tensor<1x1x128xi32, #blocked18>
    %136 = ttg.convert_layout %135 : tensor<1x1x128xi32, #blocked18> -> tensor<1x1x128xi32, #blocked16>
    %137 = arith.extsi %136 : tensor<1x1x128xi32, #blocked16> to tensor<1x1x128xi64, #blocked16>
    %138 = arith.extsi %arg1 : i32 to i64
    %139 = arith.muli %5#0, %138 : i64
    %140 = arith.extsi %arg2 : i32 to i64
    %141 = arith.muli %6, %140 : i64
    %142 = arith.addi %141, %7 : i64
    %143 = arith.addi %139, %142 : i64
    %144 = tt.addptr %arg0, %143 : !tt.ptr<bf16>, i64
    %145 = tt.splat %7 : i64 -> tensor<1x1x128xi64, #blocked16>
    %146 = arith.addi %145, %137 : tensor<1x1x128xi64, #blocked16>
    %147 = arith.extsi %arg11 : i32 to i64
    %148 = tt.splat %147 : i64 -> tensor<1x1x128xi64, #blocked16>
    %149 = arith.cmpi slt, %146, %148 : tensor<1x1x128xi64, #blocked16>
    %150 = tt.broadcast %43 : tensor<1x64x1xi1, #blocked8> -> tensor<1x64x128xi1, #blocked8>
    %151 = ttg.convert_layout %150 : tensor<1x64x128xi1, #blocked8> -> tensor<1x64x128xi1, #blocked16>
    %152 = tt.broadcast %149 : tensor<1x1x128xi1, #blocked16> -> tensor<1x64x128xi1, #blocked16>
    %153 = arith.andi %151, %152 : tensor<1x64x128xi1, #blocked16>
    %154 = tt.splat %140 : i64 -> tensor<1x64x1xi64, #blocked8>
    %155 = arith.muli %17, %154 : tensor<1x64x1xi64, #blocked8>
    %156 = tt.broadcast %155 : tensor<1x64x1xi64, #blocked8> -> tensor<1x64x128xi64, #blocked8>
    %157 = ttg.convert_layout %156 : tensor<1x64x128xi64, #blocked8> -> tensor<1x64x128xi64, #blocked16>
    %158 = tt.broadcast %137 : tensor<1x1x128xi64, #blocked16> -> tensor<1x64x128xi64, #blocked16>
    %159 = arith.addi %157, %158 : tensor<1x64x128xi64, #blocked16>
    %160 = tt.splat %144 : !tt.ptr<bf16> -> tensor<1x64x128x!tt.ptr<bf16>, #blocked16>
    %161 = tt.addptr %160, %159 : tensor<1x64x128x!tt.ptr<bf16>, #blocked16>, tensor<1x64x128xi64, #blocked16>
    %162 = arith.truncf %129 : tensor<1x64x128xf32, #blocked16> to tensor<1x64x128xbf16, #blocked16>
    %163 = ttg.convert_layout %161 : tensor<1x64x128x!tt.ptr<bf16>, #blocked16> -> tensor<1x64x128x!tt.ptr<bf16>, #blocked16>
    %164 = ttg.convert_layout %162 : tensor<1x64x128xbf16, #blocked16> -> tensor<1x64x128xbf16, #blocked16>
    %165 = ttg.convert_layout %153 : tensor<1x64x128xi1, #blocked16> -> tensor<1x64x128xi1, #blocked16>
    tt.store %163, %164, %165 : tensor<1x64x128x!tt.ptr<bf16>, #blocked16>
    tt.return
  }
}
