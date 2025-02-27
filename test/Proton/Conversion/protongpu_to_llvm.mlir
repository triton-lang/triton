// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s --dump-input-context 20
//TODO: move this test into --convert-proton-gpu-to-llvm once it gets implemented
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.func @test_empty_kernel(%arg0: i64, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>)
  tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {
    // CHECK: %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %1 = llvm.getelementptr %arg3[%0] : (!llvm.ptr<1>, i32)
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32} : !tt.ptr<i32>
    // CHECK:  llvm.return
    tt.return
  }
} // end module
