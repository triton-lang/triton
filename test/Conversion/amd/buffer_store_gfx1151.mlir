// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1151 | FileCheck %s --check-prefix=GFX1151

#blocked0 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // GFX1151-LABEL: masked_buffer_store_vec2_f32
  // GFX1151-COUNT-2: rocdl.raw.ptr.buffer.store {{.*}} : f32
  // GFX1151-NOT: rocdl.raw.ptr.buffer.store {{.*}} : vector<2xf32>
  tt.func @masked_buffer_store_vec2_f32(%value: f32, %base: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %pred: i1, %stride: i32) {
    %vals = tt.splat %value : f32 -> tensor<64xf32, #blocked0>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
    %mask = tt.splat %pred : i1 -> tensor<64xi1, #blocked0>
    amdg.buffer_store %vals, %base[%offs], %mask stride = %stride {contiguity = 2 : i32} : !tt.ptr<f32> -> tensor<64xf32, #blocked0>
    tt.return
  }

  // GFX1151-LABEL: unmasked_buffer_store_vec2_f32
  // GFX1151: rocdl.raw.ptr.buffer.store {{.*}} : vector<2xf32>
  tt.func @unmasked_buffer_store_vec2_f32(%value: f32, %base: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %stride: i32) {
    %vals = tt.splat %value : f32 -> tensor<64xf32, #blocked0>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
    amdg.buffer_store %vals, %base[%offs] stride = %stride {contiguity = 2 : i32} : !tt.ptr<f32> -> tensor<64xf32, #blocked0>
    tt.return
  }
}
