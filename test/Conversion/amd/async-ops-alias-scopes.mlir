// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s

// CHECK: [[ASYNC_COPY_SCOPE:#.*]] = #llvm.alias_scope<id = "AsyncCopies"
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @local_load_without_no_alias(%arg1: !ttg.memdesc<64x1xf16, #shared, #smem, mutable>, %arg3: tensor<64x1x!tt.ptr<f16>, #blocked>) {
    // CHECK: rocdl.global.load.lds {{.*}} {alias_scopes = [[[ASYNC_COPY_SCOPE]]]
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<64x1x!tt.ptr<f16>, #blocked> -> <64x1xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group %0

    %3 = ttg.async_wait %1 {num = 1 : i32}

    // Both LocalLoads should not have noAlias set to ASYNC_COPY_SCOPE because they do not use a token from ttg.async_wait
    // CHECK-NOT: [[[ASYNC_COPY_SCOPE]]]
    %4 = ttg.local_load %arg1: !ttg.memdesc<64x1xf16, #shared, #smem, mutable> -> tensor<64x1xf16, #blocked>
    %5 = ttg.local_load %arg1 token %0 : !ttg.memdesc<64x1xf16, #shared, #smem, mutable> -> tensor<64x1xf16, #blocked>

    // CHECK: llvm.return
    tt.return
  }
}

// -----

// CHECK: [[ASYNC_COPY_SCOPE:#.*]] = #llvm.alias_scope<id = "AsyncCopies"
// CHECK: [[LOCAL_LOAD_SCOPE:#.*]] = #llvm.alias_scope<id = "LocalLoads"
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @local_load_with_token_from_async_wait(%arg1: !ttg.memdesc<64x1xf16, #shared, #smem, mutable>, %arg3: tensor<64x1x!tt.ptr<f16>, #blocked>) {
    // CHECK: rocdl.global.load.lds {{.*}} {alias_scopes = [[[ASYNC_COPY_SCOPE]]]
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<64x1x!tt.ptr<f16>, #blocked> -> <64x1xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group %0

    %3 = ttg.async_wait %1 {num = 1 : i32}

    // CHECK: llvm.load {{.*}} {alias_scopes = [[[LOCAL_LOAD_SCOPE]]], {{.*}}, noalias_scopes = [[[ASYNC_COPY_SCOPE]]]
    %4 = ttg.local_load %arg1 token %3 : !ttg.memdesc<64x1xf16, #shared, #smem, mutable> -> tensor<64x1xf16, #blocked>

    // CHECK: llvm.return
    tt.return
  }
}
