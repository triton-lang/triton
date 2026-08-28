// RUN: triton-opt %s -split-input-file -tritoninstrument-prepare-fp-sanitizer -tritonamdgpu-fp-sanitizer | FileCheck %s

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

#packed = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // Preparation removes dead functions before AMD-specific rewrites inspect them.
  // CHECK-LABEL: @dead_scaled_upcast_kernel
  tt.func public @dead_scaled_upcast_kernel() {
    // CHECK: tt.return
    tt.return
  }
  // CHECK-NOT: @dead_compact_scaled_upcast
  tt.func private @dead_compact_scaled_upcast(%src: tensor<8x256xi8, #packed>, %scale: tensor<8x16xi8, #scale>) -> tensor<8x512xbf16, #unpacked> {
    %0 = amdg.scaled_upcast_fp4 %src scale %scale {axis = 1 : i32} : tensor<8x256xi8, #packed>, tensor<8x16xi8, #scale> -> tensor<8x512xbf16, #unpacked>
    tt.return %0 : tensor<8x512xbf16, #unpacked>
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
