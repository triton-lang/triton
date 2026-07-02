// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --verify-diagnostics

// The lowering of `ttg.global_scratch_alloc` reads the offset assigned by the
// `-tritongpu-global-scratch-memory-allocation` pass. That pass is not part of
// the AMD pipeline (the op is only produced by NVIDIA device-side TMA), so when
// such an op reaches lowering its offset attribute is absent. The conversion
// must fail to legalize cleanly rather than abort on an assertion. See #9078.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @global_scratch_alloc_missing_offset() {
    // expected-error@+1 {{failed to legalize operation 'ttg.global_scratch_alloc' that was explicitly marked illegal}}
    %0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : !tt.ptr<i8>
    "use"(%0) : (!tt.ptr<i8>) -> ()
    tt.return
  }
}
