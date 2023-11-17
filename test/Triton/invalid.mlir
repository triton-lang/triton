// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func public @reshape_different_num_elements(%arg0: tensor<32x128xf16>) {
    // expected-error @+1 {{number of src and dst elements of reshape must be the same}}
    %a = tt.reshape %arg0 : tensor<32x128xf16> -> tensor<64x32xf16>
    tt.return
}
