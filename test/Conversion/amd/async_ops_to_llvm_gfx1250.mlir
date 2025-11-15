// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_with_swizzle
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
  // CHECK-LABEL: async_load_strided_into_lds_with_swizzle
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

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_copy_with_swizzle
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

// Broadcast to all CTAs so we should just see 15 (0b1111) as the broadcast mask since we have 4 CTAs per CGA
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [2, 2], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [2, 2], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multicast_to_all_ctas
  tt.func public @async_load_multicast_to_all_ctas(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK: llvm.amdgcn.cluster.load.async.to.lds{{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[GROUP_MASK]]

    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// 8 CTAs, 2 multicast groups of 4 CTAs each. Each group is strided by 1 so the base mask should be 0b1010101 (85) and the non free mask is -7 (~0b110)
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [8, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [8, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multicast_to_half_ctas
  tt.func public @async_load_multicast_to_half_ctas(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // CHECK: llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[CTA_ID:.*]] = {{.*}}llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[NON_FREE_BITS:.*]] = llvm.mlir.constant(-7 : i32) : i32
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(85 : i32) : i32
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // CHECK: llvm.amdgcn.cluster.load.async.to.lds{{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[CTA_MASK]]
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// 16 CTAs, 8 multicast groups of 2 CTAs each, each group is strided by 8 so the base mask should be 0b100000001 (257) and the non free mask is -9 (~0b1000)
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 16], CTASplitNum = [1, 8], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 16], CTASplitNum = [1, 8], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 16 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multicast_group_of_2_strided_by_8
  tt.func public @async_load_multicast_group_of_2_strided_by_8(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // Skip the first cluster id because it's emitted for address calculation
    // CHECK: llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[CTA_ID:.*]] = {{.*}}llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[NON_FREE_BITS:.*]] = llvm.mlir.constant(-9 : i32) : i32
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(257 : i32) : i32
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // CHECK: llvm.amdgcn.cluster.load.async.to.lds{{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[CTA_MASK]]
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// 16 CTAs split into 16 multicast groups so we should not emit cluster load since we do not share any data
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 16], CTASplitNum = [1, 16], CTAOrder = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 16], CTASplitNum = [1, 16], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 16 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multi_cta_but_not_data_sharing
  tt.func public @async_load_multi_cta_but_not_data_sharing(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: llvm.amdgcn.cluster.load.async.to.lds
    // CHECK: llvm.amdgcn.global.load.async.to.lds.b64
    // CHECK-NOT: llvm.amdgcn.cluster.load.async.to.lds
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test with linear layout as src layout
// 16 CTAs, 8 multicast groups of 2 CTAs each, each group is strided by 8 so the base mask should be 0b100000001 (257) and the non free mask is -9 (~0b1000)
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [[0, 4], [0, 8], [0, 16], [0, 0]], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 16], CTASplitNum = [1, 8], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 16 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32} {
  // CHECK-LABEL: async_load_multi_cta_linear_layout
  tt.func public @async_load_multi_cta_linear_layout(%arg0: tensor<32x32x!tt.ptr<f32>, #linear> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
                                %arg1: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>) {
    // Skip the first cluster id because it's emitted for address calculation
    // CHECK: llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[CTA_ID:.*]] = {{.*}}llvm.amdgcn.cluster.workgroup.id.x
    // CHECK: %[[NON_FREE_BITS:.*]] = llvm.mlir.constant(-9 : i32) : i32
    // CHECK: %[[SHIFT_AMOUNT:.*]] = llvm.and %[[CTA_ID]], %[[NON_FREE_BITS]]
    // CHECK: %[[GROUP_MASK:.*]] = llvm.mlir.constant(257 : i32) : i32
    // CHECK: %[[CTA_MASK:.*]] = llvm.shl %[[GROUP_MASK]], %[[SHIFT_AMOUNT]]
    // CHECK: llvm.amdgcn.cluster.load.async.to.lds{{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[CTA_MASK]]
    %6 = ttg.async_copy_global_to_local %arg0, %arg1 : tensor<32x32x!tt.ptr<f32>, #linear> -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}
