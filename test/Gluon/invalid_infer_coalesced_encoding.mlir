// RUN: triton-opt %s -split-input-file --gluon-infer-coalesced-encodings -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @divisibility_conflict( %in_ptr : !tt.ptr<f32>, %out_ptr : !tt.ptr<f32>) {
    %mask = arith.constant dense<1> : tensor<128x256xi1, #gluon.auto_encoding>
    %offsets = arith.constant dense<0> : tensor<128x256xi32, #gluon.auto_encoding>

    %in_ptrs = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %in_ptrs_28 = tt.addptr %in_ptrs, %offsets {tt.contiguity = dense<[1, 256]> : tensor<2xi32>, tt.divisibility = dense<[4, 16]> : tensor<2xi32>} : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>, tensor<128x256xi32, #gluon.auto_encoding>
    // expected-error @+1 {{found conflicting encodings for value}}
    %in_ptrs_29 = gluon.set_auto_layout %in_ptrs_28 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %mask_in = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.coalesced_encoding>
    %value = tt.load %in_ptrs_29, %mask_in : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>

    %out_ptrs = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %out_ptrs_34 = tt.addptr %out_ptrs, %offsets {tt.contiguity = dense<[1, 256]> : tensor<2xi32>, tt.divisibility = dense<[4, 8]> : tensor<2xi32>} : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>, tensor<128x256xi32, #gluon.auto_encoding>
    %out_ptrs_35 = gluon.set_auto_layout %out_ptrs_34 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %mask_out = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.coalesced_encoding>
    tt.store %out_ptrs_35, %value, %mask_out : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    tt.return
}}


// -----
