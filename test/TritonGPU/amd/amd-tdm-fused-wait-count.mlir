// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=gfx-arch=gfx1250 | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_explicit_fused_waitcnt
  // CHECK: amdg.async_tdm_fused_copy_global_to_local
  // CHECK: amdg.async_tdm_intrinsic_wait {{.*}} {count = 0 : i32
  tt.func public @tdm_explicit_fused_waitcnt(
      %desc0: !tt.tensordesc<64x64xf16, #shared>,
      %desc1: !tt.tensordesc<64x64xf16, #shared>,
      %dst0: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
      %dst1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %0 = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %dst0, %dst1 {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %w = amdg.async_tdm_wait %0 {num = 0 : i32}
    tt.return
  }
}
