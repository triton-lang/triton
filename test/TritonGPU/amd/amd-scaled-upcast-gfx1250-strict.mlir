// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250-strict" --canonicalize --cse | FileCheck %s
//
// gfx1250-strict has no block16-cvt-scale-insts, which gates every cvt_scale_pk8
// upcast intrinsic. Affects scaled upcasts.

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
