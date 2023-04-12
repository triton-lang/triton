// RUN: %PYTHON -m triton.tools.aot %s --target=llvm-ir --sm=80 | FileCheck %s

// == LLVM IR check begin ==
// CHECK-LABEL: ; ModuleID = 'LLVMDialectModule'
// CHECK: define void @test_empty_kernel
// CHECK: !nvvm.annotations
// CHECK: !{ptr @test_empty_kernel, !"maxntidx", i32 128}

module attributes {"triton_gpu.num-warps" = 4 : i32} {

tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  tt.return
}

}
