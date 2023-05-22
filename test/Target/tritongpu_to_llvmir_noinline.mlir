// RUN: %PYTHON -m triton.tools.aot %s --target=llvm-ir --gfx=90a | FileCheck %s

// == LLVM IR check begin ==
// CHECK-LABEL: ; ModuleID = 'LLVMDialectModule'
// CHECK: define void @test_func
// CHECK: define amdgpu_kernel void @test_kernel
// CHECK: tail call void @test_func

module attributes {"triton_gpu.num-warps" = 4 : i32} {

tt.func @test_func(%lb : index, %A : !tt.ptr<f16>) attributes { noinline = true } {
  %0 = arith.constant 1.0 : f16
  tt.store %A, %0 : f16
  tt.return
}

tt.func @test_kernel(%lb : index, %A : !tt.ptr<f16>) {
  tt.call @test_func(%lb, %A) : (index, !tt.ptr<f16>) -> ()
  tt.return
}

}
