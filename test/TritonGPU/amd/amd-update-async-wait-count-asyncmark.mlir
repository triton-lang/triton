// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=gfx-arch=gfx950 | FileCheck %s

// For CDNA3/CDNA4, ttg.async_wait is generated from 3 sources:
//   - By pipeliner, num computed within the pass.
//   - By block-pingpong, num computed within the pass.
//   - By gluon, num filled by user.
//
// On CDNA3/CDNA4, since PR #9883, UpdateAsyncWaitCount stays no-op for all 3
// cases, which is checked by this test.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: single_token_two_crossed
  // Wait on %1 with %3 and %5 between - derivation would yield num=2; sentinel
  // num=7 must survive.
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 7 : i32}
  tt.func public @single_token_two_crossed(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_wait %1 {num = 7 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tokenless_wait_preserved
  // Tokenless wait carries a producer-authored num that derivation cannot
  // recover from a def chain - num=3 stays put.
  // CHECK: ttg.async_wait {num = 3 : i32}
  tt.func public @tokenless_wait_preserved(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    ttg.async_wait {num = 3 : i32}
    tt.return
  }
}
