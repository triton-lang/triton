// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 --verify-diagnostics

#blocked_small_vec = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_small_vec = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_small_vector_size(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x32xf16, #shared_small_vec, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked_small_vec>
    // This fails the vectoSize < 32 bits
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %2 = ttg.async_copy_global_to_local %1, %arg2 {contiguity = 1 : i32} : tensor<32x32x!tt.ptr<f16>, #blocked_small_vec> -> <32x32xf16, #shared_small_vec, #smem, mutable>
    tt.return
  }
}

// -----

#blocked_order_mismatch = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_order_mismatch = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_order_mismatch(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<64x32xf32, #shared_order_mismatch, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked_order_mismatch>
    // Order of blocked and shared mismatch resuls in non warp coalesced writes into LDS
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<64x32x!tt.ptr<f32>, #blocked_order_mismatch> -> <64x32xf32, #shared_order_mismatch, #smem, mutable>
    tt.return
  }
}

// -----

#blocked_strided = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_strided = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_strided_writes(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<64x32xf32, #shared_strided, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked_strided>
    // The blocked layout has sizePerThread=[2,1] with order=[0,1], but shared layout has order=[1,0]
    // This causes vectorization and contiguity to mismatch, resulting in strided warp writes into LDS
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<64x32x!tt.ptr<f32>, #blocked_strided> -> <64x32xf32, #shared_strided, #smem, mutable>
    tt.return
  }
}

// -----

#blocked_noncoalesced = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_noncoalesced = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_non_coalesced_layout(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<64x32xf32, #shared_noncoalesced, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked_noncoalesced>
    // The blocked layout does not exhaust the fastest dim, requiring strided warp writes into LDS
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<64x32x!tt.ptr<f32>, #blocked_noncoalesced> -> <64x32xf32, #shared_noncoalesced, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_into_invalid_subslice(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %2 = ttg.memdesc_subslice %arg2 [0, 0]  : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable, 32x64>
    // We slice in the fastest dim and one warp loads multiple rows, therefore we cannot write warp coalesced into LDS
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %3 = ttg.async_copy_global_to_local %1, %2 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable, 32x64>
    tt.return
  }
}

// -----

#blocked_subslice_slowest = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_subslice_slowest = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_subslice_too_small(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<64x32xf32, #shared_subslice_slowest, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked_subslice_slowest>
    // After slicing dim1 is 32 but threadsPerWarp is 64 which results in broadcasts for lanes > 32 which break warp coalescing
    %2 = ttg.memdesc_subslice %arg2 [32, 0]  : !ttg.memdesc<64x32xf32, #shared_subslice_slowest, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared_subslice_slowest, #smem, mutable, 64x32>
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %3 = ttg.async_copy_global_to_local %1, %2 : tensor<32x32x!tt.ptr<f32>, #blocked_subslice_slowest> -> <32x32xf32, #shared_subslice_slowest, #smem, mutable, 64x32>
    tt.return
  }
}
