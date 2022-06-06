// RUN: triton-opt %s -tritongpu-verifier
func @test_dot(%a : tensor<128x32xf16>, %b : tensor<32x128xf16>, %c : tensor<128x128xf16>) {
  %d = tt.dot %a, %b, %c {allowTF32 = true} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf16>
  return
}