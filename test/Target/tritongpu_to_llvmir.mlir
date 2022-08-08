// RUN: triton-translate %s --target=llvmir | FileCheck %s

// == LLVM IR check begin ==
// CHECK-LABEL: ; ModuleID = 'LLVMDialectModule'
// CHECK: define void @test_empty_kernel
// CHECK: !nvvm.annotations
// CHECK: !{void (i64, half addrspace(1)*)* @test_empty_kernel, !"maxntidx", i32 128}

module attributes {"triton_gpu.num-warps" = 4 : i32} {

func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  return
}

}
