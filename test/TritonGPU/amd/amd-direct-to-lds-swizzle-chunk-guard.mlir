//
// Regression test for triton-patches/patch-direct-to-lds-swizzle-chunk.patch.
//
// One f32 per lane makes each warp write 64 consecutive f32, half of a
// 128-wide row, and maxPhase = 16 moves elements into the other half. The
// lowering has to refuse such a copy instead of shuffling source pointers from
// lanes outside the warp.

// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32768 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_swizzle_leaves_chunk(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                  %arg1: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<64x128x!tt.ptr<f32>, #blocked>, tensor<64x128xi32, #blocked>
    // expected-error@+2 {{cannot lower 'ttg.async_copy_global_to_local' to a direct-to-LDS copy: the swizzle moves elements out of the chunk a warp writes with one load}}
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local'}}
    %5 = ttg.async_copy_global_to_local %4, %arg1 : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32768 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @buffer_load_to_local_swizzle_leaves_chunk(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                                            %arg1: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    // expected-error@+1 {{cannot lower 'amdg.buffer_load_to_local' to a direct-to-LDS copy: the swizzle moves elements out of the chunk a warp writes with one load}}
    %3 = amdg.buffer_load_to_local %arg0[%2] into %arg1 : !tt.ptr<f32>[tensor<64x128xi32, #blocked>] -> <64x128xf32, #shared, #smem, mutable>
    tt.return
  }
}
