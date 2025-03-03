// RUN: triton-opt %s -split-input-file --add-proton-kernel-arg | FileCheck %s --dump-input-context 20

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK: tt.func @test_empty_kernel(%arg0: index, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>) {
  tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<i8>) {
    %0 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32} : !tt.ptr<i8>
    tt.return
  }
}
