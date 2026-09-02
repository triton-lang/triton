// SPDX-License-Identifier: Apache-2.0
// RUN: triton-opt %s -verify-diagnostics

module {
  tt.func @gaudi_silu_and_mul_dynamic_quant_bf16_fp8(
      %arg0: !tt.ptr<bf16>,
      %arg1: !tt.ptr<f8E4M3FN>,
      %arg2: !tt.ptr<f32>) {
    %one = arith.constant dense<1.000000e+00> : tensor<4096xf32>
    %zero = arith.constant dense<0.000000e+00> : tensor<4096xbf16>
    %fp8_max = arith.constant 2.400000e+02 : f32
    %scale_epsilon = arith.constant 9.99999993E-9 : f32
    %input_stride = arith.constant 7168 : i32
    %n_cols_tensor = arith.constant dense<3584> : tensor<4096xi32>
    %n_cols = arith.constant 3584 : i32
    %row = tt.get_program_id x : i32
    %columns = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %mask = arith.cmpi slt, %columns, %n_cols_tensor : tensor<4096xi32>
    %input_row = arith.muli %row, %input_stride : i32
    %output_row = arith.muli %row, %n_cols : i32

    %gate_base = tt.addptr %arg0, %input_row : !tt.ptr<bf16>, i32
    %gate_splat = tt.splat %gate_base : !tt.ptr<bf16> -> tensor<4096x!tt.ptr<bf16>>
    %gate_ptr = tt.addptr %gate_splat, %columns : tensor<4096x!tt.ptr<bf16>>, tensor<4096xi32>
    %gate_bf16 = tt.load %gate_ptr, %mask, %zero : tensor<4096x!tt.ptr<bf16>>
    %gate = arith.extf %gate_bf16 : tensor<4096xbf16> to tensor<4096xf32>
    %up_base = tt.addptr %gate_base, %n_cols : !tt.ptr<bf16>, i32
    %up_splat = tt.splat %up_base : !tt.ptr<bf16> -> tensor<4096x!tt.ptr<bf16>>
    %up_ptr = tt.addptr %up_splat, %columns : tensor<4096x!tt.ptr<bf16>>, tensor<4096xi32>
    %up_bf16 = tt.load %up_ptr, %mask, %zero : tensor<4096x!tt.ptr<bf16>>
    %up = arith.extf %up_bf16 : tensor<4096xbf16> to tensor<4096xf32>

    %negative_gate = arith.negf %gate : tensor<4096xf32>
    %exponential = math.exp %negative_gate : tensor<4096xf32>
    %denominator = arith.addf %exponential, %one : tensor<4096xf32>
    %sigmoid = arith.divf %one, %denominator : tensor<4096xf32>
    %silu = arith.mulf %gate, %sigmoid : tensor<4096xf32>
    %result = arith.mulf %silu, %up : tensor<4096xf32>
    %rounded = arith.truncf %result : tensor<4096xf32> to tensor<4096xbf16>
    %rounded_f32 = arith.extf %rounded : tensor<4096xbf16> to tensor<4096xf32>

    %absolute = math.absf %rounded_f32 : tensor<4096xf32>
    %maximum = "tt.reduce"(%absolute) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<4096xf32>) -> f32
    %maximum_eps = arith.addf %maximum, %scale_epsilon : f32
    %scale = arith.divf %maximum_eps, %fp8_max : f32

    %scale_ptr = tt.addptr %arg2, %row : !tt.ptr<f32>, i32
    tt.store %scale_ptr, %scale : !tt.ptr<f32>
    %scale_splat = tt.splat %scale : f32 -> tensor<4096xf32>
    %quantized = arith.divf %rounded_f32, %scale_splat : tensor<4096xf32>
    %output_base = tt.addptr %arg1, %output_row : !tt.ptr<f8E4M3FN>, i32
    %output_splat = tt.splat %output_base : !tt.ptr<f8E4M3FN> -> tensor<4096x!tt.ptr<f8E4M3FN>>
    %output_ptr = tt.addptr %output_splat, %columns : tensor<4096x!tt.ptr<f8E4M3FN>>, tensor<4096xi32>
    %output_fp8 = tt.fp_to_fp %quantized, rounding = rtne : tensor<4096xf32> -> tensor<4096xf8E4M3FN>
    tt.store %output_ptr, %output_fp8, %mask : tensor<4096x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}
