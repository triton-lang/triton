// RUN: triton-opt %s --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250" | FileCheck %s

// A barrier must be inserted between a convert_layout and a buffer_atomic_rmw
// when they share the same LDS scratch region.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @buffer_atomic_rmw
  // CHECK-COUNT-3: rocdl.s.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xi64>
  // CHECK: rocdl.s.barrier
  // CHECK: llvm.amdgcn.raw.ptr.buffer.atomic.add
  tt.func public @buffer_atomic_rmw(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %cst = arith.constant dense<1> : tensor<2x64xi64, #blocked1>
    %1 = "tt.reduce"(%cst) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i64, %arg3: i64):
      %4 = arith.addi %arg2, %arg3 : i64
      tt.reduce.return %4 : i64
    }) : (tensor<2x64xi64, #blocked1>) -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = ttg.convert_layout %1 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<64xi64, #blocked>
    %3 = amdg.buffer_atomic_rmw add, acq_rel, gpu, %2, %arg0[%0] : tensor<64xi64, #blocked>
    amdg.buffer_store %3, %arg1[%0] : tensor<64xi64, #blocked>
    tt.return
  }
}
