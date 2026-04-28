// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions 2>&1 | FileCheck %s -implicit-check-not="scf cleanup did not converge"

// Regression test for pytorch/pytorch#180908: long chains of scf.if results
// that the cleanup phase of -tritongpu-remove-layout-conversions cannot
// converge in MLIR's default greedy-rewriter iteration cap. Site 2b (the
// SCF cleanup site) is allowed to bail with a stderr breadcrumb instead of
// failing the pass; sites 5a/5c (ConvertLayoutOp cleanup) remain strict.
//
// This test is contingent on LLVM regression cf9b3bbb0986 still being present.
// If that regression is fixed upstream, scf.if cleanup will converge in ≤10
// iterations and the breadcrumb won't fire — this test will then fail. When
// that happens, delete it (the bug it guards against is gone).
//
// The implicit-check-not flag above asserts the breadcrumb appears at most
// once across the whole run; the count directive below pins it to exactly
// once. Together these turn the negative case (a second module also
// triggering the bail) from a silent miss into a hard FileCheck failure.

// CHECK-COUNT-1: tritongpu-remove-layout-conversions: scf cleanup did not converge
// CHECK-LABEL: @crash_kernel
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @crash_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<8192xi32, #blocked>
    %c8192_i32 = arith.constant 8192 : i32
    %c9_i32 = arith.constant 9 : i32
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32, #blocked>
    %2 = tt.addptr %arg2, %0 : !tt.ptr<i32>, i32
    %3 = tt.load %2 : !tt.ptr<i32>
    %4 = arith.cmpi sgt, %3, %c0_i32 : i32
    %5 = scf.if %4 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.load %31 : !tt.ptr<i32>
      %33 = tt.splat %32 : i32 -> tensor<8192xi32, #blocked>
      %34 = arith.cmpi eq, %1, %33 : tensor<8192xi32, #blocked>
      %35 = arith.extui %34 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      scf.yield %35 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %cst : tensor<8192xi32, #blocked>
    }
    %6 = arith.cmpi sgt, %3, %c1_i32 : i32
    %7 = scf.if %6 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c1_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %5, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %5 : tensor<8192xi32, #blocked>
    }
    %8 = arith.cmpi sgt, %3, %c2_i32 : i32
    %9 = scf.if %8 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c2_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %7, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %7 : tensor<8192xi32, #blocked>
    }
    %10 = arith.cmpi sgt, %3, %c3_i32 : i32
    %11 = scf.if %10 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c3_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %9, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %9 : tensor<8192xi32, #blocked>
    }
    %12 = arith.cmpi sgt, %3, %c4_i32 : i32
    %13 = scf.if %12 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c4_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %11, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %11 : tensor<8192xi32, #blocked>
    }
    %14 = arith.cmpi sgt, %3, %c5_i32 : i32
    %15 = scf.if %14 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c5_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %13, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %13 : tensor<8192xi32, #blocked>
    }
    %16 = arith.cmpi sgt, %3, %c6_i32 : i32
    %17 = scf.if %16 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c6_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %15, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %15 : tensor<8192xi32, #blocked>
    }
    %18 = arith.cmpi sgt, %3, %c7_i32 : i32
    %19 = scf.if %18 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c7_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %17, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %17 : tensor<8192xi32, #blocked>
    }
    %20 = arith.cmpi sgt, %3, %c8_i32 : i32
    %21 = scf.if %20 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c8_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %19, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %19 : tensor<8192xi32, #blocked>
    }
    %22 = arith.cmpi sgt, %3, %c9_i32 : i32
    %23 = scf.if %22 -> (tensor<8192xi32, #blocked>) {
      %30 = arith.muli %0, %c10_i32 : i32
      %31 = tt.addptr %arg1, %30 : !tt.ptr<i32>, i32
      %32 = tt.addptr %31, %c9_i32 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = tt.splat %33 : i32 -> tensor<8192xi32, #blocked>
      %35 = arith.cmpi eq, %1, %34 : tensor<8192xi32, #blocked>
      %36 = arith.extui %35 : tensor<8192xi1, #blocked> to tensor<8192xi32, #blocked>
      %37 = arith.addi %21, %36 : tensor<8192xi32, #blocked>
      scf.yield %37 : tensor<8192xi32, #blocked>
    } else {
      scf.yield %21 : tensor<8192xi32, #blocked>
    }
    %24 = arith.muli %0, %c8192_i32 : i32
    %25 = tt.addptr %arg0, %24 : !tt.ptr<i32>, i32
    %26 = tt.splat %25 : !tt.ptr<i32> -> tensor<8192x!tt.ptr<i32>, #blocked>
    %27 = tt.addptr %26, %1 : tensor<8192x!tt.ptr<i32>, #blocked>, tensor<8192xi32, #blocked>
    %28 = ttg.convert_layout %27 : tensor<8192x!tt.ptr<i32>, #blocked> -> tensor<8192x!tt.ptr<i32>, #blocked1>
    %29 = ttg.convert_layout %23 : tensor<8192xi32, #blocked> -> tensor<8192xi32, #blocked1>
    tt.store %28, %29 : tensor<8192x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}

// -----

// Negative control: a healthy kernel with no scf.if chain. Pass should
// converge silently (no breadcrumb).

// CHECK-LABEL: @simple_kernel
// CHECK-NOT: scf cleanup did not converge
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @simple_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<8192xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8192x!tt.ptr<i32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<8192x!tt.ptr<i32>, #blocked>, tensor<8192xi32, #blocked>
    %4 = ttg.convert_layout %3 : tensor<8192x!tt.ptr<i32>, #blocked> -> tensor<8192x!tt.ptr<i32>, #blocked1>
    %5 = ttg.convert_layout %cst : tensor<8192xi32, #blocked> -> tensor<8192xi32, #blocked1>
    tt.store %4, %5 : tensor<8192x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}
