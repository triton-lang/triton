// RUN: not --crash triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 2>&1 | FileCheck %s

// Regression test for issue #11795:
// Calling toLinearLayout on an unencoded RankedTensorType previously caused a null
// pointer dereference in Attribute::getContext(). It now produces an explicit fatal error.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @unencoded_tensor_local_load() {
    %buf0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    // CHECK: LLVM ERROR: toLinearLayout: RankedTensorType must have a layout encoding, got none
    %res = ttg.local_load %buf0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    tt.return
  }
}
