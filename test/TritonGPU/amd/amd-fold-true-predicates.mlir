// RUN: triton-opt %s -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @buffer_load_masks
  // CHECK: %[[FIRST:.*]] = amdg.buffer_load %arg0[%arg1] soffset = %arg4 : i32
  // CHECK: %[[SECOND:.*]] = amdg.buffer_load %arg0[%arg1], %arg2, %[[FIRST]] soffset = %arg4 : i32
  // CHECK: tt.return %[[SECOND]]
  tt.func @buffer_load_masks(%ptr: !tt.ptr<i32>, %offsets: tensor<64xi32, #blocked>, %mask: tensor<64xi1, #blocked>, %other: tensor<64xi32, #blocked>, %soffset: i32) -> tensor<64xi32, #blocked> {
    %true = arith.constant dense<true> : tensor<64xi1, #blocked>
    %first = amdg.buffer_load %ptr[%offsets], %true, %other soffset = %soffset : i32 : !tt.ptr<i32> -> tensor<64xi32, #blocked>
    %second = amdg.buffer_load %ptr[%offsets], %mask, %first soffset = %soffset : i32 : !tt.ptr<i32> -> tensor<64xi32, #blocked>
    tt.return %second : tensor<64xi32, #blocked>
  }

  // Omitting pred would inherit the input descriptor's predicate.
  // CHECK-LABEL: @set_descriptor_pred
  // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
  // CHECK: %[[DESC:.*]] = amdg.update_tensor_descriptor %arg0 pred = %[[ONE]]
  // CHECK: tt.return %[[DESC]]
  tt.func @set_descriptor_pred(%desc: !tt.tensordesc<64xi32>) -> !tt.tensordesc<64xi32> {
    %one = arith.constant 1 : i32
    %updated = amdg.update_tensor_descriptor %desc pred = %one : !tt.tensordesc<64xi32>
    tt.return %updated : !tt.tensordesc<64xi32>
  }
}
