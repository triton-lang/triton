// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=num-warps=8

// CHECK: llvm.func @test_empty_kernel(%arg0: i64, %arg1: !llvm.ptr<f16, 1>)
// CHECK: attributes {nvvm.maxntidx = 96 : i32}
func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  // CHECK:  llvm.return
  return
}
