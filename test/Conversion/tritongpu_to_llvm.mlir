// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s


module attributes {"triton_gpu.num-warps" = 4 : i32} {

// CHECK: llvm.func @test_empty_kernel(%arg0: i64, %arg1: !llvm.ptr<f16, 1>)
// Here the 128 comes from the 4 in module attribute multiples 32
// CHECK: attributes {nvvm.maxntid = 128 : i32} {{.*}}
func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  // CHECK:  llvm.return
  return
}

} // end module
