// RUN: triton-opt %s -test-print-amd-alignment -split-input-file -verify-diagnostics=only-expected -o /dev/null

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>

tt.func public @kernel(%arg0: tensor<256x64xf16, #mma> {tt.contiguity=256 : i32, tt.divisibility=6: i32, tt.constancy=1: i32}) {
  // expeted-remark @below {{contiguity = [128, 32], divisibility = [6, 6], constancy = [1, 1], constant_value = <none>}}
  %0 = amdgpu.extract_slice %arg0 [128, 32] : tensor<256x64xf16, #mma> to tensor<128x32xf16, #mma>
  tt.return
}
