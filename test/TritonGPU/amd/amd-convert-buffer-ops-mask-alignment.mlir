// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx942 analyze-small-tensor-ofst=true" | FileCheck %s --check-prefix=COMMON
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx950 analyze-small-tensor-ofst=true" | FileCheck %s --check-prefix=COMMON
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1151 analyze-small-tensor-ofst=true" | FileCheck %s --check-prefix=COMMON
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1250 analyze-small-tensor-ofst=true" | FileCheck %s --check-prefix=COMMON

#blocked0 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // COMMON-LABEL: unmasked_store_vec2
  // COMMON: amdg.buffer_store %{{.*}}, %arg0[%[[RANGE:.*]]] {contiguity = 2 : i32} : !tt.ptr<f32> -> tensor<64xf32, #{{.*}}>
  tt.func @unmasked_store_vec2(%base: !tt.ptr<f32> {tt.divisibility = 8 : i32, tt.pointer_range = 32 : i32}, %value: f32) {
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
    %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked0>
    %addrs = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<f32>, #blocked0>, tensor<64xi32, #blocked0>
    %vals = tt.splat %value : f32 -> tensor<64xf32, #blocked0>
    tt.store %addrs, %vals : tensor<64x!tt.ptr<f32>, #blocked0>
    tt.return
  }

  // COMMON-LABEL: masked_store_even_boundary
  // COMMON: amdg.buffer_store %{{.*}}, %arg0[%[[RANGE:.*]]], %[[MASK:.*]] {contiguity = 2 : i32} : !tt.ptr<f32> -> tensor<64xf32, #{{.*}}>
  tt.func @masked_store_even_boundary(%base: !tt.ptr<f32> {tt.divisibility = 8 : i32, tt.pointer_range = 32 : i32}, %value: f32) {
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
    %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked0>
    %addrs = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<f32>, #blocked0>, tensor<64xi32, #blocked0>
    %vals = tt.splat %value : f32 -> tensor<64xf32, #blocked0>
    %limit = arith.constant 62 : i32
    %limits = tt.splat %limit : i32 -> tensor<64xi32, #blocked0>
    %mask = arith.cmpi slt, %offs, %limits : tensor<64xi32, #blocked0>
    tt.store %addrs, %vals, %mask : tensor<64x!tt.ptr<f32>, #blocked0>
    tt.return
  }

  // COMMON-LABEL: masked_store_odd_boundary
  // COMMON: amdg.buffer_store %{{.*}}, %arg0[%[[RANGE:.*]]], %[[MASK:.*]] : !tt.ptr<f32> -> tensor<64xf32, #{{.*}}>
  // COMMON-NOT: amdg.buffer_store {{.*}} {contiguity = 2 : i32}
  tt.func @masked_store_odd_boundary(%base: !tt.ptr<f32> {tt.divisibility = 8 : i32, tt.pointer_range = 32 : i32}, %value: f32) {
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
    %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked0>
    %addrs = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<f32>, #blocked0>, tensor<64xi32, #blocked0>
    %vals = tt.splat %value : f32 -> tensor<64xf32, #blocked0>
    %limit = arith.constant 63 : i32
    %limits = tt.splat %limit : i32 -> tensor<64xi32, #blocked0>
    %mask = arith.cmpi slt, %offs, %limits : tensor<64xi32, #blocked0>
    tt.store %addrs, %vals, %mask : tensor<64x!tt.ptr<f32>, #blocked0>
    tt.return
  }

  // COMMON-LABEL: masked_store_runtime_boundary
  // COMMON: amdg.buffer_store %{{.*}}, %arg0[%[[RANGE:.*]]], %[[MASK:.*]] : !tt.ptr<f32> -> tensor<64xf32, #{{.*}}>
  // COMMON-NOT: amdg.buffer_store {{.*}} {contiguity = 2 : i32}
  tt.func @masked_store_runtime_boundary(%base: !tt.ptr<f32> {tt.divisibility = 8 : i32, tt.pointer_range = 32 : i32}, %value: f32, %limit: i32) {
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
    %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked0>
    %addrs = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<f32>, #blocked0>, tensor<64xi32, #blocked0>
    %vals = tt.splat %value : f32 -> tensor<64xf32, #blocked0>
    %limits = tt.splat %limit : i32 -> tensor<64xi32, #blocked0>
    %mask = arith.cmpi slt, %offs, %limits : tensor<64xi32, #blocked0>
    tt.store %addrs, %vals, %mask : tensor<64x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}
