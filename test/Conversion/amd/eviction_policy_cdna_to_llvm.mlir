// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_evict_first_cdna
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_evict_first_cdna(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<256xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // CDNA3/4: aux = 2 (nt)
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 2
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_first>} : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cv_evict_first_cdna
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_cv_evict_first_cdna(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<256xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // CDNA3/4: .cv keeps sc0|sc1 (17), not clobbered by evict_first
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 17
    %ret = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_first>} : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_evict_first_cdna
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_store_evict_first_cdna(%value: tensor<256xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<256xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // CDNA3/4: aux = 2 (nt)
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 2
    amdg.buffer_store %value, %arg0[%offset] stride = %c256_i32 {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_first>} : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: global_load_store_evict_first_cdna
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @global_load_store_evict_first_cdna(%arg0: tensor<256x!tt.ptr<f32>, #blocked>, %arg1: tensor<256x!tt.ptr<f32>, #blocked>) {
    // CHECK: llvm.load {{.*}} <nontemporal>
    %val = tt.load %arg0 {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_first>} : tensor<256x!tt.ptr<f32>, #blocked>
    // CHECK: llvm.store {{.*}} <{{.*}}nontemporal>
    tt.store %arg1, %val {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_first>} : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
