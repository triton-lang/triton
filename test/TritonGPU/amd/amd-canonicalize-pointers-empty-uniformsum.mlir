// RUN: triton-opt %s -split-input-file -tritonamdgpu-canonicalize-pointers="enable-large-tensor-ptr-canon=false" | FileCheck %s

// Test case for empty uniformSum bug fix.
//
// This test reproduces the scenario where both fatPtrOffset and origOffset are constant tensors,
// causing uniformSum to be NULL in rewriteSmallTensorPtr().
//
// Before fix: Would crash with assertion "dyn_cast on a non-existent value"
// After fix: Handles gracefully by initializing uniformSum to 0 if NULL

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @test_empty_uniformsum
  tt.func @test_empty_uniformsum(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) {
    // Constant offset tensor (simulates fully unrolled loop index)
    %cst = arith.constant dense<1> : tensor<128xi32, #blocked>

    // Create pointer tensor from scalar pointer
    // After canonicalization: FatPtr(base=%arg0, offset=splat(0))
    // CHECK: tt.splat %arg0
    %ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>

    // Load with base pointer (iteration 0)
    // CHECK: tt.load
    %data0 = tt.load %ptr : tensor<128x!tt.ptr<f32>, #blocked>

    // BUG TRIGGER: addptr with constant offset
    // - fatPtrOffset = splat(0)  [constant, classified as splatTensor]
    // - origOffset = dense<1>     [constant, classified as splatTensor]
    // Result: uniforms=[], nonUniforms=[], splatTensors=[(splat(0),0), (dense<1>,1)]
    //         uniformSum stays NULL -> crash before fix
    // CHECK: tt.addptr
    %ptr_next = tt.addptr %ptr, %cst : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>

    // Load with updated pointer (iteration 1)
    // CHECK: tt.load
    %data1 = tt.load %ptr_next : tensor<128x!tt.ptr<f32>, #blocked>

    // Store results to prevent DCE (dead code elimination)
    %out_ptr = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: tt.store
    tt.store %out_ptr, %data0 : tensor<128x!tt.ptr<f32>, #blocked>

    %cst_128 = arith.constant dense<128> : tensor<128xi32, #blocked>
    %out_ptr_next = tt.addptr %out_ptr, %cst_128 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    // CHECK: tt.store
    tt.store %out_ptr_next, %data1 : tensor<128x!tt.ptr<f32>, #blocked>

    tt.return
  }
}
