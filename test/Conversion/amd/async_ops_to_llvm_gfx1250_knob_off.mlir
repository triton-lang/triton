// With staged async marks disabled, a commit group on gfx1250 emits no
// asyncmark (the previous behavior; waits go through s_wait_asynccnt).
// RUN: env TRITON_HIP_USE_STAGED_ASYNC_MARKS=0 triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: commit_knob_off
  tt.func public @commit_knob_off(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %tok = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x64x!tt.ptr<f32>, #blocked> -> <32x64xf32, #shared, #smem, mutable>
    // CHECK-NOT: rocdl.asyncmark
    %ctok = ttg.async_commit_group tokens %tok
    tt.return
  }
}
