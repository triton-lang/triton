// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm --verify-diagnostics

// a scan with its axis split across CTAs cant be lowered, partial results
// only get combined inside one CTA. the error shoudl say so

#layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [2, 1], order = [0, 1], CGALayout = [[1, 0]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 16 : i32} {
  tt.func private @scan_axis_split_across_ctas(%arg0: tensor<16x16xi32, #layout>) -> tensor<16x16xi32, #layout> {
    // expected-error@+2 {{scan axis is split across CTAs}}
    // expected-error@+1 {{failed to legalize operation 'tt.scan'}}
    %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg1: i32, %arg2: i32):
      %1 = arith.addi %arg1, %arg2 : i32
      tt.scan.return %1 : i32
    }) : (tensor<16x16xi32, #layout>) -> tensor<16x16xi32, #layout>
    tt.return %0 : tensor<16x16xi32, #layout>
  }
}

// -----

// control: same scan but the CTAs are split along a non-scan axis, this
// one has to lower jsut fine (num_ctas stays 2)

#layout_non_axis = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [2, 1], order = [0, 1], CGALayout = [[0, 1]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 16 : i32} {
  tt.func private @scan_non_axis_split(%arg0: tensor<16x16xi32, #layout_non_axis>) -> tensor<16x16xi32, #layout_non_axis> {
    %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg1: i32, %arg2: i32):
      %1 = arith.addi %arg1, %arg2 : i32
      tt.scan.return %1 : i32
    }) : (tensor<16x16xi32, #layout_non_axis>) -> tensor<16x16xi32, #layout_non_axis>
    tt.return %0 : tensor<16x16xi32, #layout_non_axis>
  }
}
