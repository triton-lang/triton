// RUN: triton-opt --split-input-file %s --verify-diagnostics

#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @undersized_profile_record(%seg: !proton_gpu.segment<16, #smem, warp>, %clock: i32) {
    // expected-error @+1 {{profiling buffer segment must hold at least 8 bytes for a clock entry}}
    proton_gpu.circular_store start %seg, %clock {scopeId = 0 : i32} : !proton_gpu.segment<16, #smem, warp>, i32
    tt.return
  }
}
