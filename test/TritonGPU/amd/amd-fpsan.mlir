// RUN: triton-opt %s -split-input-file -tritonamdgpu-fp-sanitizer | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @scaled_upcast_fp8
  tt.func public @scaled_upcast_fp8(%src: tensor<32x128xf8E4M3FN, #blocked>, %scale: tensor<32x128xbf16, #blocked>) -> tensor<32x128xbf16, #blocked> {
    // CHECK: tt.fp_to_fp
    // CHECK: arith.mulf
    // CHECK-NOT: amdg.scaled_upcast_fp8
    %0 = amdg.scaled_upcast_fp8 %src scale %scale : tensor<32x128xf8E4M3FN, #blocked>, tensor<32x128xbf16, #blocked> -> tensor<32x128xbf16, #blocked>
    tt.return %0 : tensor<32x128xbf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @scaled_upcast_fp4
  tt.func public @scaled_upcast_fp4(%src: tensor<16x32xi8, #blocked>, %scale: tensor<16x64xbf16, #blocked>) -> tensor<16x64xbf16, #blocked> {
    // CHECK: ttg.fp4_to_fp
    // CHECK: arith.mulf
    // CHECK-NOT: amdg.scaled_upcast_fp4
    %0 = amdg.scaled_upcast_fp4 %src scale %scale {axis = 1 : i32} : tensor<16x32xi8, #blocked>, tensor<16x64xbf16, #blocked> -> tensor<16x64xbf16, #blocked>
    tt.return %0 : tensor<16x64xbf16, #blocked>
  }
}
