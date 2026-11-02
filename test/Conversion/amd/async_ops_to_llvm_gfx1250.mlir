// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: staged_commit_wait
  tt.func public @staged_commit_wait(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %tok = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x64x!tt.ptr<f32>, #blocked> -> <32x64xf32, #shared, #smem, mutable>
    // The copy may lower to a plain or a cluster (multicast) async load, so the
    // commit conservatively marks both load stages.
    // CHECK: rocdl.asyncmark stages global_load_async_to_lds|global_load_async_to_lds_mcast
    %ctok = ttg.async_commit_group tokens %tok
    // CHECK: rocdl.wait.asyncmark 0 stages global_load_async_to_lds|global_load_async_to_lds_mcast
    ttg.async_wait %ctok {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: staged_wait_no_token
  tt.func public @staged_wait_no_token() {
    // Without tokens the stages cannot be proven, so the wait names every
    // stage (printed without a stages clause).
    // CHECK: rocdl.wait.asyncmark 0{{$}}
    ttg.async_wait {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: staged_commit_no_token
  tt.func public @staged_commit_no_token() {
    // An empty commit group marks every stage.
    // CHECK: rocdl.asyncmark{{$}}
    ttg.async_commit_group
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: staged_store_commit_wait
  tt.func public @staged_store_commit_wait(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %tok = amdg.async_copy_local_to_global %arg1, %1 : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    // CHECK: rocdl.asyncmark stages global_store_async_from_lds
    %ctok = ttg.async_commit_group tokens %tok
    // CHECK: rocdl.wait.asyncmark 0 stages global_store_async_from_lds
    ttg.async_wait %ctok {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: staged_tdm_commit_wait
  tt.func public @staged_tdm_commit_wait(%arg0: !tt.tensordesc<64x64xf16, #shared>,
                                %arg1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    // TDM copies are normally waited on with async_tdm_wait; feeding the token
    // through a commit group locks in the tensor stage classification.
    %tok = amdg.async_tdm_copy_global_to_local %arg0 into %arg1 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: rocdl.asyncmark stages tensor
    %ctok = ttg.async_commit_group tokens %tok
    // CHECK: rocdl.wait.asyncmark 0 stages tensor
    ttg.async_wait %ctok {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: staged_mixed_commit_wait
  tt.func public @staged_mixed_commit_wait(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>,
                                %arg2: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %tok1 = ttg.async_copy_global_to_local %1, %arg1 : tensor<32x64x!tt.ptr<f32>, #blocked> -> <32x64xf32, #shared, #smem, mutable>
    %tok2 = amdg.async_copy_local_to_global %arg2, %1 : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    // One commit group marks the union of its producers' stages.
    // CHECK: rocdl.asyncmark stages global_load_async_to_lds|global_load_async_to_lds_mcast|global_store_async_from_lds
    %ctok = ttg.async_commit_group tokens %tok1, %tok2
    // CHECK: rocdl.wait.asyncmark 0 stages global_load_async_to_lds|global_load_async_to_lds_mcast|global_store_async_from_lds
    ttg.async_wait %ctok {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: staged_wait_two_groups
  tt.func public @staged_wait_two_groups(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>,
                                %arg2: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %tok1 = ttg.async_copy_global_to_local %1, %arg1 : tensor<32x64x!tt.ptr<f32>, #blocked> -> <32x64xf32, #shared, #smem, mutable>
    %ctok1 = ttg.async_commit_group tokens %tok1
    %tok2 = amdg.async_copy_local_to_global %arg2, %1 : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %ctok2 = ttg.async_commit_group tokens %tok2
    // A wait over two groups names the union of both groups' stages and
    // passes its commit-group count through.
    // CHECK: rocdl.wait.asyncmark 1 stages global_load_async_to_lds|global_load_async_to_lds_mcast|global_store_async_from_lds
    ttg.async_wait %ctok1, %ctok2 {num = 1 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: staged_wait_chained
  tt.func public @staged_wait_chained(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %tok = ttg.async_copy_global_to_local %1, %arg1 : tensor<32x64x!tt.ptr<f32>, #blocked> -> <32x64xf32, #shared, #smem, mutable>
    %ctok = ttg.async_commit_group tokens %tok
    %w = ttg.async_wait %ctok {num = 0 : i32}
    // Stages propagate through an intervening wait token.
    // CHECK: rocdl.wait.asyncmark 2 stages global_load_async_to_lds|global_load_async_to_lds_mcast
    ttg.async_wait %w {num = 2 : i32}
    tt.return
  }
}
