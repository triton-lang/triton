// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250-strict --convert-builtin-func-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250-strict --convert-builtin-func-to-llvm 2>&1 | FileCheck %s --check-prefix=REMARK

// REMARK-NOT: Multicast group contains
// REMARK-NOT: falling back to regular load

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [0, 0], [0, 0]]}>
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: load_broadcast_no_cluster_load
  tt.func public @load_broadcast_no_cluster_load(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>}) {
    // CHECK-NOT: rocdl.cluster.workgroup.id.x
    // CHECK-NOT: llvm.amdgcn.cluster.load
    // CHECK: llvm.return
    %6 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0], [0, 0]]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CGALayout = [[0, 0], [0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_broadcast_no_cluster_load
  tt.func public @async_load_broadcast_no_cluster_load(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: rocdl.cluster.load.async.to.lds
    // CHECK: rocdl.global.load.async.to.lds
    // CHECK-NOT: rocdl.cluster.load.async.to.lds
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1], [0, 2], [0, 0], [0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 16 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_load_broadcast_no_multicast_mask
  tt.func public @tdm_load_broadcast_no_multicast_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    // CHECK-NOT: llvm.mlir.constant(4369 : i32)
    // CHECK: "llvm.amdgcn.tensor.load.to.lds"
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %2 = amdg.async_tdm_copy_global_to_local %0 into %1 : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
