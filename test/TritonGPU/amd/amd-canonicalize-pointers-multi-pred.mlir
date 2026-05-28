// RUN: triton-opt %s -allow-unregistered-dialect -split-input-file -tritonamdgpu-canonicalize-pointers="enable-large-tensor-ptr-canon=true" | FileCheck %s

// Scalar pointer passed as a block argument from two different predecessors:
//   ^bb0 -> (cf.cond_br, false branch) -> ^bb2(ptr=arg0)
//   ^bb1 -> (cf.br)                    -> ^bb2(ptr=arg1)

// CHECK-LABEL:   tt.func @scalar_ptr_block_arg_two_predecessors(
// CHECK:           cf.cond_br {{.*}}, ^bb1, ^bb2({{.*}} : !tt.ptr<f32>, i64)
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2({{.*}} : !tt.ptr<f32>, i64)
// CHECK:         ^bb2({{.*}}: !tt.ptr<f32>, {{.*}}: i64):

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @scalar_ptr_block_arg_two_predecessors(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i1) -> f32 {
    cf.cond_br %arg2, ^bb1, ^bb2(%arg0 : !tt.ptr<f32>)
  ^bb1:  // pred: ^bb0 (true branch of cond_br)
    cf.br ^bb2(%arg1 : !tt.ptr<f32>)
  ^bb2(%0: !tt.ptr<f32>):  // 2 preds: ^bb0 (false branch), ^bb1
    %1 = tt.load %0 : !tt.ptr<f32>
    tt.return %1 : f32
  }
}
