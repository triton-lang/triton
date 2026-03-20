// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s

// Test that the amdg.lds domain alias scopes are correctly propagated to
// lowered LLVM ops when two non-overlapping LDS buffers exist.

// CHECK-DAG: [[$LDS_SCOPE_0:#.*]] = #llvm.alias_scope<id = "triton.lds.buffer.0"
// CHECK-DAG: [[$LDS_SCOPE_1:#.*]] = #llvm.alias_scope<id = "triton.lds.buffer.1"
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @two_lds_buffers_alias_scopes
  tt.func public @two_lds_buffers_alias_scopes(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_a = arith.constant dense<0.0> : tensor<64x1xf16, #blocked>
    %cst_b = arith.constant dense<0.0> : tensor<64x1xf16, #blocked>

    %alloc_a = ttg.local_alloc %cst_a : (tensor<64x1xf16, #blocked>) -> !ttg.memdesc<64x1xf16, #shared, #smem>
    %alloc_b = ttg.local_alloc %cst_b : (tensor<64x1xf16, #blocked>) -> !ttg.memdesc<64x1xf16, #shared, #smem>

    // Stores to LDS for buffer A get scope 0, noalias scope 1
    // CHECK: llvm.store {{.*}} {alias_scopes = {{.*}}[[$LDS_SCOPE_0]]{{.*}}, {{.*}}noalias_scopes = {{.*}}[[$LDS_SCOPE_1]]
    // Stores to LDS for buffer B get scope 1, noalias scope 0
    // CHECK: llvm.store {{.*}} {alias_scopes = {{.*}}[[$LDS_SCOPE_1]]{{.*}}, {{.*}}noalias_scopes = {{.*}}[[$LDS_SCOPE_0]]

    // Loads from buffer A get scope 0, noalias scope 1
    // CHECK: llvm.load {{.*}} {alias_scopes = {{.*}}[[$LDS_SCOPE_0]]{{.*}}, noalias_scopes = {{.*}}[[$LDS_SCOPE_1]]
    %load_a = ttg.local_load %alloc_a : !ttg.memdesc<64x1xf16, #shared, #smem> -> tensor<64x1xf16, #blocked>

    // Loads from buffer B get scope 1, noalias scope 0
    // CHECK: llvm.load {{.*}} {alias_scopes = {{.*}}[[$LDS_SCOPE_1]]{{.*}}, noalias_scopes = {{.*}}[[$LDS_SCOPE_0]]
    %load_b = ttg.local_load %alloc_b : !ttg.memdesc<64x1xf16, #shared, #smem> -> tensor<64x1xf16, #blocked>

    // Global stores should NOT have LDS alias scopes
    // CHECK: llvm.store {{.*}} !llvm.ptr<1>
    // CHECK-NOT: alias_scopes
    %ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    tt.store %ptr, %load_a : tensor<64x1x!tt.ptr<f16>, #blocked>
    tt.store %ptr, %load_b : tensor<64x1x!tt.ptr<f16>, #blocked>

    // CHECK: llvm.return
    tt.return
  }
}
