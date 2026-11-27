// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck %s --check-prefixes=CHECK,RDNA3
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1150 | FileCheck %s --check-prefixes=CHECK,RDNA3
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1201 | FileCheck %s --check-prefixes=CHECK,GFX12
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s --check-prefixes=CHECK,GFX12

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_evict_first
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_evict_first(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // RDNA3: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 6
    // GFX12: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 1
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_first>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cg_evict_first
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cg_evict_first(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // RDNA3: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 7
    // GFX12: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 17
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_first>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_evict_last
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_evict_last(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // RDNA3: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 0
    // GFX12: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 2
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_last>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_evict_first
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_evict_first(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // RDNA3: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 6
    // GFX12: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 1
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_first>} : !tt.ptr<f32> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: global_load_store_evict_first
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @global_load_store_evict_first(%arg0: tensor<128x!tt.ptr<f32>, #blocked>, %arg1: tensor<128x!tt.ptr<f32>, #blocked>) {
    // CHECK: llvm.load {{.*}} <nontemporal>
    %val = tt.load %arg0 {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_first>} : tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: llvm.store {{.*}} <{{.*}}nontemporal>
    tt.store %arg1, %val {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_first>} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
