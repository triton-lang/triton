// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 --convert-builtin-func-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_vectorized
  tt.func public @async_copy_vectorized(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // We need the index calculation so AxisAnalysis sees that we can vectorize the load
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>

    // Each thread needs to load 8 elements and we load 8 (sizePerThread) per global.load.lds
    // CHECK: rocdl.global.load.lds
    // CHECK-NOT: rocdl.global.load.lds
    %6 = ttg.async_copy_global_to_local %5, %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
