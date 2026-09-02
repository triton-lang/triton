// SPDX-License-Identifier: Apache-2.0
// RUN: triton-opt %s -verify-diagnostics

module {
  tt.func @gaudi_add_f32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %pid = tt.get_program_id x : i32
    %c256 = arith.constant 256 : i32
    %base = arith.muli %pid, %c256 : i32
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %base_splat = tt.splat %base : i32 -> tensor<256xi32>
    %offsets = arith.addi %base_splat, %range : tensor<256xi32>
    %bound = tt.splat %arg3 : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<256xi32>

    %lhs_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %lhs_ptr = tt.addptr %lhs_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %rhs_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %rhs_ptr = tt.addptr %rhs_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

    %zero = arith.constant 0.000000e+00 : f32
    %other = tt.splat %zero : f32 -> tensor<256xf32>
    %lhs = tt.load %lhs_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %rhs = tt.load %rhs_ptr, %mask, %other : tensor<256x!tt.ptr<f32>>
    %sum = arith.addf %lhs, %rhs : tensor<256xf32>
    tt.store %out_ptr, %sum, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
