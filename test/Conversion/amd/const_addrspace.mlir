// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s

// A `tl.const` pointer carries Triton's constant address space (4). On AMD it
// lowers to the AMDGPU constant address space so uniform loads select s_load; a
// plain pointer stays global (1).
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: llvm.func @const_ptr_arg
  // CHECK-SAME: %arg0: !llvm.ptr<4>
  // CHECK-SAME: %arg1: !llvm.ptr<1>
  tt.func @const_ptr_arg(%const: !tt.ptr<f32, "constant">, %global: !tt.ptr<f32>) {
    tt.return
  }
}
