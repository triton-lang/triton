// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --verify-diagnostics

// Only gfx1250 descriptors hold more than 32 bits of num_records.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @make_buffer_desc_i64_num_records(%arg0: !tt.ptr<f32>, %n: i64) {
    // expected-error @+1 {{a 64-bit num_records is only supported on gfx1250}}
    %0 = amdg.make_buffer_desc %arg0, %n : !tt.ptr<f32>, i64 -> !amdg.buffer_desc<f32>
    tt.return
  }
}
