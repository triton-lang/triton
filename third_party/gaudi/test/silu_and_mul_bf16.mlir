// SPDX-License-Identifier: Apache-2.0
// RUN: triton-opt %s -verify-diagnostics

module {
  tt.func @gaudi_silu_and_mul_bf16(
      %arg0: !tt.ptr<bf16>,
      %arg1: !tt.ptr<bf16>) {
    %one = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %zero = arith.constant dense<0.000000e+00> : tensor<128xbf16>
    %input_stride = arith.constant 7168 : i32
    %n_cols_tensor = arith.constant dense<3584> : tensor<128xi32>
    %n_cols = arith.constant 3584 : i32
    %block_size = arith.constant 128 : i32
    %chunk = tt.get_program_id x : i32
    %row = tt.get_program_id y : i32
    %chunk_base = arith.muli %chunk, %block_size : i32
    %lanes = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %chunk_base_splat = tt.splat %chunk_base : i32 -> tensor<128xi32>
    %columns = arith.addi %chunk_base_splat, %lanes : tensor<128xi32>
    %mask = arith.cmpi slt, %columns, %n_cols_tensor : tensor<128xi32>
    %input_row = arith.muli %row, %input_stride : i32
    %output_row = arith.muli %row, %n_cols : i32
    %input_row_splat = tt.splat %input_row : i32 -> tensor<128xi32>
    %gate_offsets = arith.addi %input_row_splat, %columns : tensor<128xi32>
    %up_offsets = arith.addi %gate_offsets, %n_cols_tensor : tensor<128xi32>
    %output_row_splat = tt.splat %output_row : i32 -> tensor<128xi32>
    %output_offsets = arith.addi %output_row_splat, %columns : tensor<128xi32>

    %input_base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %gate_ptr = tt.addptr %input_base, %gate_offsets : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %gate_bf16 = tt.load %gate_ptr, %mask, %zero : tensor<128x!tt.ptr<bf16>>
    %gate = arith.extf %gate_bf16 : tensor<128xbf16> to tensor<128xf32>
    %up_ptr = tt.addptr %input_base, %up_offsets : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %up_bf16 = tt.load %up_ptr, %mask, %zero : tensor<128x!tt.ptr<bf16>>
    %up = arith.extf %up_bf16 : tensor<128xbf16> to tensor<128xf32>

    %negative_gate = arith.negf %gate : tensor<128xf32>
    %exponential = math.exp %negative_gate : tensor<128xf32>
    %denominator = arith.addf %exponential, %one : tensor<128xf32>
    %sigmoid = arith.divf %one, %denominator : tensor<128xf32>
    %silu = arith.mulf %gate, %sigmoid : tensor<128xf32>
    %result = arith.mulf %silu, %up : tensor<128xf32>

    %output_base = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %output_ptr = tt.addptr %output_base, %output_offsets : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %output_bf16 = arith.truncf %result : tensor<128xf32> to tensor<128xbf16>
    tt.store %output_ptr, %output_bf16, %mask : tensor<128x!tt.ptr<bf16>>
    tt.return
  }
}
