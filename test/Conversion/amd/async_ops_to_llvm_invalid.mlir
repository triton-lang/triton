// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 --verify-diagnostics
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_1_byte(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xi8, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x64x!tt.ptr<i8>, #blocked>
    // AsyncCopyGlobalToLocal is only supported for >= 4 bytes
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x64x!tt.ptr<i8>, #blocked> -> <32x64xi8, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_2_bytes(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    // AsyncCopyGlobalToLocal is only supported for >= 4 bytes
    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// Padding interval of 1 forces vec==1 which we cannot lower because it's less than 32bits per lane
#shared = #ttg.padded_shared<[1:+2] {order = [1, 0], shape = [32, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @async_copy_padded_invalid_vec(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                    %arg1: i32 {tt.divisibility = 16 : i32},
                                    %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // We need the index calculation so AxisAnalysis sees that we can vectorize the load
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>

    // expected-error@+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %6 = ttg.async_copy_global_to_local %5, %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
