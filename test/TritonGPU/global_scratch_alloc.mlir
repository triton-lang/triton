// RUN: triton-opt %s -split-input-file --tritongpu-global-scratch-memory-allocation | FileCheck %s

// CHECK: module attributes {triton_gpu.global_scratch_memory_alignment = 128 : i32, triton_gpu.global_scratch_memory_size = 256 : i32{{.*}}}
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK: @test_alloc{{.*}}triton_gpu.global_scratch_memory_alignment = 128 : i32, triton_gpu.global_scratch_memory_size = 256 : i32
  tt.func public @test_alloc() -> (!tt.ptr<i8>, !tt.ptr<i8>) {
    // CHECK:  triton_gpu.global_scratch_memory_offset = 0
    %0 = triton_gpu.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : <i8>
    // CHECK:  triton_gpu.global_scratch_memory_offset = 128
    %1 = triton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : <i8>
    tt.return %0, %1 : !tt.ptr<i8>, !tt.ptr<i8>
  }
}

// -----

// CHECK: module attributes {triton_gpu.global_scratch_memory_alignment = 128 : i32, triton_gpu.global_scratch_memory_size = 256 : i32{{.*}}}
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK: @helper1{{.*}}triton_gpu.global_scratch_memory_alignment = 128 : i32, triton_gpu.global_scratch_memory_size = 128 : i32
  tt.func private @helper1() -> (!tt.ptr<i8>) {
    // CHECK:  triton_gpu.global_scratch_memory_offset = 0
    %0 = triton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : <i8>
    tt.return %0 : !tt.ptr<i8>
  }

// CHECK: @test_function{{.*}}triton_gpu.global_scratch_memory_alignment = 128 : i32, triton_gpu.global_scratch_memory_size = 256 : i32
  tt.func public @test_function() -> (!tt.ptr<i8>, !tt.ptr<i8>) {
    // CHECK:  triton_gpu.global_scratch_memory_offset = 0
    %0 = triton_gpu.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : <i8>
    // CHECK:  triton_gpu.global_scratch_memory_offset = 128
    %1 = tt.call @helper1() : () -> (!tt.ptr<i8>)
    tt.return %0, %1 : !tt.ptr<i8>, !tt.ptr<i8>
  }
}
