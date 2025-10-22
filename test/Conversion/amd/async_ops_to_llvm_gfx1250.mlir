
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy
  // GFX1250-LABEL: async_copy
  tt.func public @async_copy_with_swizzle(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    // Each thread needs to load 8 elements and we load 1 (sizePerThread) per global.load.lds
    // CHECK-COUNT-8: llvm.amdgcn.global.load.async.to.lds.b32
    // CHECK-NOT: llvm.amdgcn.global.load.async.to.lds
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_strided_into_lds
  tt.func public @async_load_strided_into_lds_with_swizzle(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // Each thread loads 256 contiguous bits so we split into 2 128bit loads. This was not possible on GFX9
    // CHECK-COUNT-2: llvm.amdgcn.global.load.async.to.lds.b128
    // CHECK-NOT: llvm.amdgcn.global.load.async.to.lds
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_wait
  tt.func public @async_wait() {
    // CHECK: [[WAIT_0:%.*]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: llvm.amdgcn.s.wait.asynccnt"([[WAIT_0]])
    ttg.async_wait {num = 0 : i32}
    // CHECK: [[WAIT_32:%.*]] = llvm.mlir.constant(32 : i16) : i16
    // CHECK: llvm.amdgcn.s.wait.asynccnt"([[WAIT_32]])
    ttg.async_wait {num = 32 : i32}
    tt.return
  }
}
