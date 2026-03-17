// RUN: triton-opt %s -test-print-amd-alignment -split-input-file -verify-diagnostics=only-expected -o /dev/null

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>

tt.func public @kernel_contiguity_limiting_size(%arg0: tensor<256x64xf16, #mma> {tt.contiguity = dense<[256, 64]> : tensor<2xi32>, tt.divisibility = dense<[8, 8]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
  // expected-remark @below {{contiguity = [128, 32], divisibility = [8, 8], constancy = [1, 1], constant_value = <none>}}
  %0 = amdg.extract_slice %arg0 [0, 0] : tensor<256x64xf16, #mma> to tensor<128x32xf16, #mma>
  tt.return
}

// -----

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>

tt.func public @kernel_contiguity_limiting_offset(%arg0: tensor<256x64xf16, #mma> {tt.contiguity = dense<[256, 64]> : tensor<2xi32>, tt.divisibility = dense<[8, 8]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
  // expected-remark @below {{contiguity = [64, 16], divisibility = [8, 8], constancy = [1, 1], constant_value = <none>}}
  %0 = amdg.extract_slice %arg0 [64, 16] : tensor<256x64xf16, #mma> to tensor<128x32xf16, #mma>
  tt.return
}

// -----

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>

tt.func public @kernel_divisibility(%arg0: tensor<256x64xf16, #mma> {tt.contiguity = dense<[256, 64]> : tensor<2xi32>, tt.divisibility = dense<[1024, 1024]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
  // expected-remark @below {{contiguity = [32, 32], divisibility = [128, 32], constancy = [1, 1], constant_value = <none>}}
  %0 = amdg.extract_slice %arg0 [128, 32] : tensor<256x64xf16, #mma> to tensor<32x32xf16, #mma>
  tt.return
}

// -----

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>

tt.func public @kernel_constancy_limiting_size(%arg0: tensor<256x64xf16, #mma> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[1024, 1024]> : tensor<2xi32>, tt.constancy = dense<[256, 64]> : tensor<2xi32>}) {
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1024, 1024], constancy = [128, 32], constant_value = <none>}}
  %0 = amdg.extract_slice %arg0 [0, 0] : tensor<256x64xf16, #mma> to tensor<128x32xf16, #mma>
  tt.return
}

// -----

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>

tt.func public @kernel_constancy_limiting_offset(%arg0: tensor<256x64xf16, #mma> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[1024, 1024]> : tensor<2xi32>, tt.constancy = dense<[128, 32]> : tensor<2xi32>}) {
  // expected-remark @below {{contiguity = [1, 1], divisibility = [1024, 1024], constancy = [64, 16], constant_value = <none>}}
  %0 = amdg.extract_slice %arg0 [64, 16] : tensor<256x64xf16, #mma> to tensor<128x32xf16, #mma>
  tt.return
}
