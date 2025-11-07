// RUN: triton-opt %s -split-input-file --gluon-infer-efficient-encodings -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @divisibility_conflict(
    %in_ptr : !tt.ptr<f32> {tt.divisibility = 64 : i32},
    %out_ptr : !tt.ptr<f32> {tt.divisibility = 8 : i32}) {
    %mask = arith.constant dense<1> : tensor<128x256xi1, #gluon.auto_encoding>
    %cst = arith.constant 4 : i32

    %x_range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #gluon.auto_encoding> 
    %in_offsets = tt.expand_dims %x_range {axis = 1 : i32} : tensor<128xi32, #gluon.auto_encoding> -> tensor<128x1xi32, #gluon.auto_encoding> 
    %in_offsets_8 = tt.splat %cst : i32 -> tensor<128x1xi32, #gluon.auto_encoding> 
    %x_offsets = arith.muli %in_offsets_8, %in_offsets : tensor<128x1xi32, #gluon.auto_encoding> 

    %y_range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #gluon.auto_encoding> 
    %y_offsets = tt.expand_dims %y_range {axis = 0 : i32} : tensor<256xi32, #gluon.auto_encoding> -> tensor<1x256xi32, #gluon.auto_encoding> 

    %in_offsets_11 = tt.broadcast %x_offsets : tensor<128x1xi32, #gluon.auto_encoding> -> tensor<128x256xi32, #gluon.auto_encoding> 
    %in_offsets_12 = tt.broadcast %y_offsets : tensor<1x256xi32, #gluon.auto_encoding> -> tensor<128x256xi32, #gluon.auto_encoding> 
    %ptr_offsets = arith.addi %in_offsets_11, %in_offsets_12 : tensor<128x256xi32, #gluon.auto_encoding> 

    %in_ptrs = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> 
    %in_ptrs_28 = tt.addptr %in_ptrs, %ptr_offsets : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>, tensor<128x256xi32, #gluon.auto_encoding> 

    // expected-error @+1 {{found conflicting encodings for value}}
    %in_ptrs_29 = gluon.set_auto_layout %in_ptrs_28 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.efficient_encoding> 
    %mask_in = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.efficient_encoding>
    %value = tt.load %in_ptrs_29, %mask_in : tensor<128x256x!tt.ptr<f32>, #gluon.efficient_encoding> 

    %out_ptrs = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> 
    %out_ptrs_34 = tt.addptr %out_ptrs, %ptr_offsets : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>, tensor<128x256xi32, #gluon.auto_encoding> 
    %out_ptrs_35 = gluon.set_auto_layout %out_ptrs_34 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.efficient_encoding>
    %mask_out = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.efficient_encoding>
    tt.store %out_ptrs_35, %value, %mask_out : tensor<128x256x!tt.ptr<f32>, #gluon.efficient_encoding>
    tt.return
}}

// -----