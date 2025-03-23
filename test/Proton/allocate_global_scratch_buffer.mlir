// RUN: triton-opt --split-input-file -allocate-proton-global-scratch-buffer %s | FileCheck %s

// CHECK: module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 768 : i32} {
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<i8>) {
    // CHECK: %0 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i8>
    %0 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32} : !tt.ptr<i8>
    // CHECK: %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 384 : i32} : !tt.ptr<i8>
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32} : !tt.ptr<i8>
    tt.return
  }
}
