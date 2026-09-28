// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250-strict" --canonicalize --cse | FileCheck %s
//
// gfx1250-strict has no block16-cvt-scale-insts, affects every cvt_scale_pk8 upcast intrinsic

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scaled_upcast_fp8_non_broadcast
  // CHECK-NOT: cvt.scale.pk8
  // CHECK: llvm.fmul
  // CHECK-NOT: cvt.scale.pk8
  tt.func public @scaled_upcast_fp8_non_broadcast(
      %input: tensor<32x128xf8E4M3FN, #blocked>,
      %scale: tensor<32x128xi8, #blocked>,
      %output: tensor<32x128x!tt.ptr<bf16>, #blocked>) {
    %upcast = amdg.scaled_upcast_fp8 %input scale %scale :
        tensor<32x128xf8E4M3FN, #blocked>, tensor<32x128xi8, #blocked> ->
        tensor<32x128xbf16, #blocked>
    tt.store %output, %upcast : tensor<32x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#packed = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#compact = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scaled_upcast_fp4_broadcast_block32
  // CHECK-NOT: cvt.scale.pk8
  // CHECK: llvm.fmul
  // CHECK-NOT: cvt.scale.pk8
  tt.func public @scaled_upcast_fp4_broadcast_block32(
      %output: tensor<1x1024x!tt.ptr<bf16>, #unpacked>,
      %input: tensor<1x512xi8, #packed>,
      %scale: tensor<1x32xi8, #compact>) {
    %upcast = amdg.scaled_upcast_fp4 %input scale %scale {axis = 1 : i32} :
        tensor<1x512xi8, #packed>, tensor<1x32xi8, #compact> ->
        tensor<1x1024xbf16, #unpacked>
    tt.store %output, %upcast : tensor<1x1024x!tt.ptr<bf16>, #unpacked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250-strict", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @fp8_upcast
  // CHECK-NOT: cvt.scale.pk8
  // CHECK: llvm.return
  tt.func public @fp8_upcast(%e4m3: tensor<1024xf8E4M3FN, #blocked>, %e5m2: tensor<1024xf8E5M2, #blocked>,
                             %f16: tensor<1024x!tt.ptr<f16>, #blocked>, %bf16: tensor<1024x!tt.ptr<bf16>, #blocked>,
                             %f32: tensor<1024x!tt.ptr<f32>, #blocked>) {
    %0 = tt.fp_to_fp %e4m3 : tensor<1024xf8E4M3FN, #blocked> -> tensor<1024xf16, #blocked>
    %1 = tt.fp_to_fp %e4m3 : tensor<1024xf8E4M3FN, #blocked> -> tensor<1024xbf16, #blocked>
    %2 = tt.fp_to_fp %e4m3 : tensor<1024xf8E4M3FN, #blocked> -> tensor<1024xf32, #blocked>
    %3 = tt.fp_to_fp %e5m2 : tensor<1024xf8E5M2, #blocked> -> tensor<1024xf16, #blocked>
    %4 = tt.fp_to_fp %e5m2 : tensor<1024xf8E5M2, #blocked> -> tensor<1024xbf16, #blocked>
    %5 = tt.fp_to_fp %e5m2 : tensor<1024xf8E5M2, #blocked> -> tensor<1024xf32, #blocked>
    tt.store %f16, %0 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.store %bf16, %1 : tensor<1024x!tt.ptr<bf16>, #blocked>
    tt.store %f32, %2 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.store %f16, %3 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.store %bf16, %4 : tensor<1024x!tt.ptr<bf16>, #blocked>
    tt.store %f32, %5 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
