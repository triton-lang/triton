// SPDX-License-Identifier: Apache-2.0
// RUN: triton-opt %s -verify-diagnostics

module {
  tt.func @gaudi_fused_add_rms_norm_bf16(
      %arg0: !tt.ptr<bf16>,
      %arg1: !tt.ptr<bf16>,
      %arg2: !tt.ptr<bf16>,
      %arg3: !tt.ptr<bf16>,
      %arg4: !tt.ptr<bf16>,
      %arg5: f32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<1024xbf16>
    %n_cols_f32 = arith.constant 7.690000e+02 : f32
    %n_cols_tensor = arith.constant dense<769> : tensor<1024xi32>
    %n_cols_i32 = arith.constant 769 : i32
    %row = tt.get_program_id x : i32
    %columns = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %mask = arith.cmpi slt, %columns, %n_cols_tensor : tensor<1024xi32>
    %row_base = arith.muli %row, %n_cols_i32 : i32
    %row_base_splat = tt.splat %row_base : i32 -> tensor<1024xi32>
    %offsets = arith.addi %row_base_splat, %columns : tensor<1024xi32>

    %hidden_base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %hidden_ptr = tt.addptr %hidden_base, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %hidden_bf16 = tt.load %hidden_ptr, %mask, %zero : tensor<1024x!tt.ptr<bf16>>
    %hidden = arith.extf %hidden_bf16 : tensor<1024xbf16> to tensor<1024xf32>

    %residual_base = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %residual_ptr = tt.addptr %residual_base, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %residual_bf16 = tt.load %residual_ptr, %mask, %zero : tensor<1024x!tt.ptr<bf16>>
    %residual = arith.extf %residual_bf16 : tensor<1024xbf16> to tensor<1024xf32>

    %summed = arith.addf %hidden, %residual : tensor<1024xf32>
    %summed_bf16 = arith.truncf %summed : tensor<1024xf32> to tensor<1024xbf16>
    %residual_output_base = tt.splat %arg4 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %residual_output_ptr = tt.addptr %residual_output_base, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    tt.store %residual_output_ptr, %summed_bf16, %mask : tensor<1024x!tt.ptr<bf16>>

    %summed_f32 = arith.extf %summed_bf16 : tensor<1024xbf16> to tensor<1024xf32>
    %squared = arith.mulf %summed_f32, %summed_f32 : tensor<1024xf32>
    %sum = "tt.reduce"(%squared) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.addf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<1024xf32>) -> f32
    %variance = arith.divf %sum, %n_cols_f32 : f32
    %variance_eps = arith.addf %variance, %arg5 : f32
    %rrms = math.rsqrt %variance_eps : f32

    %weight_base = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %weight_ptr = tt.addptr %weight_base, %columns : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %weight_bf16 = tt.load %weight_ptr, %mask, %zero : tensor<1024x!tt.ptr<bf16>>
    %weight = arith.extf %weight_bf16 : tensor<1024xbf16> to tensor<1024xf32>
    %rrms_splat = tt.splat %rrms : f32 -> tensor<1024xf32>
    %normalized = arith.mulf %summed_f32, %rrms_splat : tensor<1024xf32>
    %weighted = arith.mulf %normalized, %weight : tensor<1024xf32>

    %output_base = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %output_ptr = tt.addptr %output_base, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %output_bf16 = arith.truncf %weighted : tensor<1024xf32> to tensor<1024xbf16>
    tt.store %output_ptr, %output_bf16, %mask : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}
