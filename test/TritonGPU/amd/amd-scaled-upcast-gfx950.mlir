// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse | FileCheck %s

#packed = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scalef32_bf16_fp4_compact
  // CHECK: rocdl.cvt
  tt.func public @cvt_scalef32_bf16_fp4_compact(%output: tensor<8x256x!tt.ptr<bf16>, #unpacked>, %x: tensor<8x128xi8, #packed>, %scale: tensor<8x8xi8, #scale>) {
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 1 : i32} : tensor<8x128xi8, #packed>, tensor<8x8xi8, #scale> -> tensor<8x256xbf16, #unpacked>
    tt.store %output, %up : tensor<8x256x!tt.ptr<bf16>, #unpacked>
    tt.return
  }
}
