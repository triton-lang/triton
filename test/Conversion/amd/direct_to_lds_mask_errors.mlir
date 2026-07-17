// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --verify-diagnostics

// A direct-to-LDS copy transfers each lane's whole vector in a single
// transaction, so the predication mask must be aligned to the vector width.
// When it is not (its true/false boundary is a runtime value, giving per-element
// alignment) the load cannot be lowered, and the diagnostic must blame the mask
// rather than the pointer alignment.

// ---- async_copy: pointer allows a 2xf16 (32-bit) vector, mask forces it to 1 ----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_unaligned_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                            %bound: i32,
                                            %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
    // mask boundary is a runtime value along the vectorized (fast) dim -> alignment 1
    %6 = tt.splat %bound : i32 -> tensor<32x64xi32, #blocked>
    %7 = arith.cmpi slt, %3, %6 : tensor<32x64xi32, #blocked>

    // expected-error@+2 {{The mask is the limiting factor}}
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %8 = ttg.async_copy_global_to_local %5, %arg2 mask %7 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// ---- buffer_load_to_local: same situation via a scalar base + i32 offsets ----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @buffer_load_to_local_unaligned_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                                      %bound: i32,
                                                      %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    // mask boundary is a runtime value along the vectorized (fast) dim -> alignment 1
    %6 = tt.splat %bound : i32 -> tensor<32x64xi32, #blocked>
    %7 = arith.cmpi slt, %3, %6 : tensor<32x64xi32, #blocked>

    // expected-error@+2 {{The mask is the limiting factor}}
    // expected-error@+1 {{failed to legalize operation 'amdg.buffer_load_to_local' that was explicitly marked illegal}}
    %8 = amdg.buffer_load_to_local %arg0[%3] mask = %7 into %arg2 : <f16>[tensor<32x64xi32, #blocked>] -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
