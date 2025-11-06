// RUN: triton-opt %s --gluon-infer-efficient-encodings -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @divisibility_conflict(
    %in_ptr : !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %out_ptr : !tt.ptr<f32> {tt.divisibility = 8 : i32}) {

    %mask = arith.constant dense<1> : tensor<128x256xi1, #gluon.auto_encoding>
    %xstride_in = arith.constant 2048 : i32
    %xstride_out = arith.constant 2048 : i32

    %pid_x = tt.get_program_id x : i32
    %pid_y = tt.get_program_id y : i32

    %indices_x = arith.constant 128 : i32
    %indices_x_0 = arith.muli %pid_x, %indices_x : i32
    %indices_x_1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #gluon.auto_encoding>
    %indices_x_2 = tt.splat %indices_x_0 : i32 -> tensor<128xi32, #gluon.auto_encoding>
    %indices_x_3 = arith.addi %indices_x_2, %indices_x_1 : tensor<128xi32, #gluon.auto_encoding>

    %indices_y = arith.constant 256 : i32
    %indices_y_4 = arith.muli %pid_y, %indices_y : i32
    %indices_y_5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #gluon.auto_encoding>
    %indices_y_6 = tt.splat %indices_y_4 : i32 -> tensor<256xi32, #gluon.auto_encoding>
    %indices_y_7 = arith.addi %indices_y_6, %indices_y_5 : tensor<256xi32, #gluon.auto_encoding>

    %in_offsets = tt.expand_dims %indices_x_3 {axis = 1 : i32} : tensor<128xi32, #gluon.auto_encoding> -> tensor<128x1xi32, #gluon.auto_encoding>
    %in_offsets_8 = tt.splat %xstride_in : i32 -> tensor<128x1xi32, #gluon.auto_encoding>
    %in_offsets_9 = arith.muli %in_offsets_8, %in_offsets : tensor<128x1xi32, #gluon.auto_encoding>
    %in_offsets_10 = tt.expand_dims %indices_y_7 {axis = 0 : i32} : tensor<256xi32, #gluon.auto_encoding> -> tensor<1x256xi32, #gluon.auto_encoding>
    %in_offsets_11 = tt.broadcast %in_offsets_9 : tensor<128x1xi32, #gluon.auto_encoding> -> tensor<128x256xi32, #gluon.auto_encoding>
    %in_offsets_12 = tt.broadcast %in_offsets_10 : tensor<1x256xi32, #gluon.auto_encoding> -> tensor<128x256xi32, #gluon.auto_encoding>
    %in_offsets_13 = arith.addi %in_offsets_11, %in_offsets_12 : tensor<128x256xi32, #gluon.auto_encoding>

    %out_offsets = tt.expand_dims %indices_x_3 {axis = 1 : i32} : tensor<128xi32, #gluon.auto_encoding> -> tensor<128x1xi32, #gluon.auto_encoding>
    %out_offsets_14 = tt.splat %xstride_out : i32 -> tensor<128x1xi32, #gluon.auto_encoding>
    %out_offsets_15 = arith.muli %out_offsets_14, %out_offsets : tensor<128x1xi32, #gluon.auto_encoding>
    %out_offsets_16 = tt.expand_dims %indices_y_7 {axis = 0 : i32} : tensor<256xi32, #gluon.auto_encoding> -> tensor<1x256xi32, #gluon.auto_encoding>
    %out_offsets_17 = tt.broadcast %out_offsets_15 : tensor<128x1xi32, #gluon.auto_encoding> -> tensor<128x256xi32, #gluon.auto_encoding>
    %out_offsets_18 = tt.broadcast %out_offsets_16 : tensor<1x256xi32, #gluon.auto_encoding> -> tensor<128x256xi32, #gluon.auto_encoding>
    %out_offsets_19 = arith.addi %out_offsets_17, %out_offsets_18 : tensor<128x256xi32, #gluon.auto_encoding>

    %in_ptrs = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %in_ptrs_28 = tt.addptr %in_ptrs, %in_offsets_13 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>, tensor<128x256xi32, #gluon.auto_encoding>

    // expected-error @+1 {{found conflicting encodings for value}}
    %in_ptrs_29 = gluon.set_auto_layout %in_ptrs_28 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.efficient_encoding>

    %value = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.efficient_encoding>
    %value_30 = tt.load %in_ptrs_29, %value : tensor<128x256x!tt.ptr<f32>, #gluon.efficient_encoding>

    %value_31 = math.sin %value_30 : tensor<128x256xf32, #gluon.efficient_encoding>
    %value_32 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #gluon.efficient_encoding>
    %value_33 = arith.maxnumf %value_31, %value_32 : tensor<128x256xf32, #gluon.efficient_encoding>
    %out_ptrs = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %out_ptrs_34 = tt.addptr %out_ptrs, %out_offsets_19 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>, tensor<128x256xi32, #gluon.auto_encoding>
    %out_ptrs_35 = gluon.set_auto_layout %out_ptrs_34 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.efficient_encoding>
    %out_mask_layouted = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.efficient_encoding>
    tt.store %out_ptrs_35, %value_33, %out_mask_layouted : tensor<128x256x!tt.ptr<f32>, #gluon.efficient_encoding>
    tt.return
}}
