// RUN: triton-opt %s -convert-triton-to-tritongpu

func @ops() {
  %a = arith.constant dense<1.00e+00> : tensor<128x32xf16>
  %b = arith.constant dense<2.00e+00> : tensor<32x128xf16>
  %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
  %0 = tt.dot %a, %b, %c {allowTF32 = true} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
  return
}