// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @canonicalize_after_concat(
    %arg0: tensor<32x64xf32, #blocked>,
    %arg1: tensor<32x64xf32, #blocked>,
    %arg2: tensor<32x64xf32, #blocked>,
    %arg3: tensor<32x64xf32, #blocked>,
    %arg4: tensor<32x64xf32, #blocked>,
    %arg5: tensor<32x64xf32, #blocked>,
    %arg6: tensor<32x64xf32, #blocked>,
    %arg7: tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #blocked> {
    // CHECK-LABEL: tt.func @canonicalize_after_concat

    %1 = amdg.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7:
    tensor<32x64xf32, #blocked>,tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %2 = amdg.extract_slice %1 [32, 64] : tensor<128x128xf32, #blocked> to tensor<32x64xf32, #blocked>
    // CHECK: tt.return %arg3 : tensor<32x64xf32, #blocked>
    tt.return %2 : tensor<32x64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @canonicalize_singleton_concat(%arg0: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    // CHECK-LABEL: tt.func @canonicalize_singleton_concat

    %1 = amdg.concat %arg0: tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
    // CHECK: tt.return %arg0 : tensor<128x128xf32, #blocked>
    tt.return %1 : tensor<128x128xf32, #blocked>
  }
}

// -----

#src = #ttg.linear<{register = [[1], [128]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = []}>
#dst = #ttg.linear<{register = [[128], [1]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = []}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @do_not_canonicalize_singleton_concat_layout_change(%arg0: tensor<256xi32, #src>) -> tensor<256xi32, #dst> {
    // CHECK-LABEL: tt.func @do_not_canonicalize_singleton_concat_layout_change
    // CHECK: %[[CONCAT:.*]] = amdg.concat %arg0
    // CHECK: tt.return %[[CONCAT]]

    %concat = amdg.concat %arg0 : tensor<256xi32, #src> -> tensor<256xi32, #dst>
    tt.return %concat : tensor<256xi32, #dst>
  }
}
