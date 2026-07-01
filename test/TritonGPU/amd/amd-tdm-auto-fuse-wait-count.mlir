// RUN: env TRITON_HIP_DISABLE_TDM_AUTO_FUSE=0 triton-opt %s -split-input-file --tritonamdgpu-auto-fuse-tdm-copy --tritonamdgpu-update-async-wait-count=gfx-arch=gfx1250 | FileCheck %s --check-prefix=FUSE
// RUN: env TRITON_HIP_DISABLE_TDM_AUTO_FUSE=1 triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=gfx-arch=gfx1250 | FileCheck %s --check-prefix=NOFUSE

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // FUSE-LABEL: tdm_auto_fuse_waitcnt
  // FUSE: amdg.async_tdm_fused_copy_global_to_local
  // FUSE-NOT: amdg.async_tdm_copy_global_to_local
  // FUSE: amdg.async_tdm_intrinsic_wait {{.*}} {count = 0 : i32

  // NOFUSE-LABEL: tdm_auto_fuse_waitcnt
  // NOFUSE: amdg.async_tdm_copy_global_to_local
  // NOFUSE: amdg.async_tdm_copy_global_to_local
  // NOFUSE: amdg.async_tdm_intrinsic_wait {{.*}} {count = 1 : i32
  tt.func public @tdm_auto_fuse_waitcnt(
      %desc: !tt.tensordesc<64x64xf16, #shared>,
      %dst0: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
      %dst1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %0 = amdg.async_tdm_copy_global_to_local %desc into %dst0 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %dst1 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // Auto-fusion rewrites both original copy tokens to the fused token, so the
    // tokenized wait no longer has the second copy outstanding.
    %w = amdg.async_tdm_wait %0 {num = 0 : i32}
    tt.return
  }
}
