// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#non_disjoint = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[32, 0], [16, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @convert_layout_non_disjoint_bases(%arg0: tensor<64x32xf32, #blocked>) {
    // expected-error @+2 {{convert_layout through shared memory is not supported for swizzled layouts. Use explicit shared memory operations (local_alloc/local_load/local_store) instead.}}
    // expected-error @+1 {{failed to legalize operation 'ttg.convert_layout' that was explicitly marked illegal}}
    %0 = ttg.convert_layout %arg0 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #non_disjoint>
    tt.return
  }
}
