// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s
// Cross-warp tt.reduce: axis 0 spans 8 warps (256 rows over 8 warps x 32 lanes),
// so the reduce epilogue goes through shared memory. Each warp's partial is
// broadcast across all 32 lanes; the shared store must be predicated on the
// representative lane (regression #11614, restored on 3.7.1).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @reduce_cross_warp(%ptr: !tt.ptr<i32>, %arg0: tensor<256x4xi32, #blocked>) {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: i32, %arg2: i32):
      %1 = arith.addi %arg1, %arg2 : i32
      tt.reduce.return %1 : i32
    }) : (tensor<256x4xi32, #blocked>) -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    // CHECK: llvm.inline_asm{{.*}}st.shared::cta.v4.b32
    // CHECK: llvm.icmp "eq"
    %2 = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.store %2, %0 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return
  }
}