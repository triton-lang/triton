// RUN: triton-opt -split-input-file %s --allocate-shared-memory-nv='compute-capability=120' --convert-triton-gpu-to-llvm='compute-capability=120' -verify-diagnostics

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @convert_layout_forced_non_warp_local(
      %arg0: tensor<32x32xf32, #blocked0>) attributes {always_use_warp_shuffle} {
    // expected-error @+2 {{'always_use_warp_shuffle' requires a warp-local layout conversion}}
    // expected-error @+1 {{failed to legalize operation 'ttg.convert_layout' that was explicitly marked illegal}}
    %0 = ttg.convert_layout %arg0
        : tensor<32x32xf32, #blocked0> -> tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

// expected-error @below {{num_ctas > 1 is not supported on sm_120}}
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @sm120_multi_cta() {
    tt.return
  }
}
