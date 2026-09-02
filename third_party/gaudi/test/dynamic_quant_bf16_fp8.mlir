// SPDX-License-Identifier: Apache-2.0
// RUN: triton-opt %s -verify-diagnostics

module {
  tt.func @gaudi_dynamic_quant_bf16_fp8(
      %arg0: !tt.ptr<bf16>,
      %arg1: !tt.ptr<f8E4M3FN>,
      %arg2: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<1024xbf16>
    %fp8_max = arith.constant 2.400000e+02 : f32
    %scale_epsilon = arith.constant 9.99999993E-9 : f32
    %n_cols_tensor = arith.constant dense<769> : tensor<1024xi32>
    %n_cols = arith.constant 769 : i32
    %row = tt.get_program_id x : i32
    %columns = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %mask = arith.cmpi slt, %columns, %n_cols_tensor : tensor<1024xi32>
    %row_base = arith.muli %row, %n_cols : i32
    %row_base_splat = tt.splat %row_base : i32 -> tensor<1024xi32>
    %offsets = arith.addi %row_base_splat, %columns : tensor<1024xi32>

    %input_base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %input_ptr = tt.addptr %input_base, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %input_bf16 = tt.load %input_ptr, %mask, %zero : tensor<1024x!tt.ptr<bf16>>
    %input_f32 = arith.extf %input_bf16 : tensor<1024xbf16> to tensor<1024xf32>
    %absolute = math.absf %input_f32 : tensor<1024xf32>
    %maximum = "tt.reduce"(%absolute) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<1024xf32>) -> f32
    %maximum_eps = arith.addf %maximum, %scale_epsilon : f32
    %scale = arith.divf %maximum_eps, %fp8_max : f32

    %scale_ptr = tt.addptr %arg2, %row : !tt.ptr<f32>, i32
    tt.store %scale_ptr, %scale : !tt.ptr<f32>
    %scale_splat = tt.splat %scale : f32 -> tensor<1024xf32>
    %quantized = arith.divf %input_f32, %scale_splat : tensor<1024xf32>
    %output_base = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<1024x!tt.ptr<f8E4M3FN>>
    %output_ptr = tt.addptr %output_base, %offsets : tensor<1024x!tt.ptr<f8E4M3FN>>, tensor<1024xi32>
    %output_fp8 = tt.fp_to_fp %quantized, rounding = rtne : tensor<1024xf32> -> tensor<1024xf8E4M3FN>
    tt.store %output_ptr, %output_fp8, %mask : tensor<1024x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}
