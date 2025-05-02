// RUN: triton-opt %s -split-input-file -triton-amdgpu-refine-ops='arch=gfx942' -canonicalize | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#smem = #ttg.shared_memory


// CHECK-LABEL: @local_alloc_refinement
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @local_alloc_refinement(%arg0: tensor<64x16xf16, #blocked>) attributes {noinline = false} {

    // CHECK: [[OFFSET_12:%.*]] = arith.constant 12 : i32
    // CHECK: [[OFFSET_8:%.*]] = arith.constant 8 : i32
    // CHECK: [[OFFSET_4:%.*]] = arith.constant 4 : i32
    // CHECK: [[OFFSET_0:%.*]] = arith.constant 0 : i32
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>
    // CHECK: [[SUBVIEW_0:%.*]] = ttg.memdesc_subview [[ALLOC]][[[OFFSET_0]], [[OFFSET_0]], [[OFFSET_0]]] : !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x4xf16, #shared, #smem, mutable, 1x64x16>
    // CHECK: [[SLICE_0:%.*]] = amdgpu.extract_slice %arg0 [0, 0] : tensor<64x16xf16, #blocked> to tensor<64x4xf16, #blocked>
    // CHECK: ttg.local_store [[SLICE_0]], [[SUBVIEW_0]] : tensor<64x4xf16, #blocked> -> !ttg.memdesc<64x4xf16, #shared, #smem, mutable, 1x64x16>
    // CHECK: [[SUBVIEW_1:%.*]] = ttg.memdesc_subview [[ALLOC]][[[OFFSET_0]], [[OFFSET_0]], [[OFFSET_4]]] : !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x4xf16, #shared, #smem, mutable, 1x64x16>
    // CHECK: [[SLICE_1:%.*]] = amdgpu.extract_slice %arg0 [0, 4] : tensor<64x16xf16, #blocked> to tensor<64x4xf16, #blocked>
    // CHECK: ttg.local_store [[SLICE_1]], [[SUBVIEW_1]] : tensor<64x4xf16, #blocked> -> !ttg.memdesc<64x4xf16, #shared, #smem, mutable, 1x64x16>
    // CHECK: [[SUBVIEW_2:%.*]] = ttg.memdesc_subview [[ALLOC]][[[OFFSET_0]], [[OFFSET_0]], [[OFFSET_8]]] : !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x4xf16, #shared, #smem, mutable, 1x64x16>
    // CHECK: [[SLICE_2:%.*]] = amdgpu.extract_slice %arg0 [0, 8] : tensor<64x16xf16, #blocked> to tensor<64x4xf16, #blocked>
    // CHECK: ttg.local_store [[SLICE_2]], [[SUBVIEW_2]] : tensor<64x4xf16, #blocked> -> !ttg.memdesc<64x4xf16, #shared, #smem, mutable, 1x64x16>
    // CHECK: [[SUBVIEW_3:%.*]] = ttg.memdesc_subview [[ALLOC]][[[OFFSET_0]], [[OFFSET_0]], [[OFFSET_12]]] : !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x4xf16, #shared, #smem, mutable, 1x64x16>
    // CHECK: [[SLICE_3:%.*]] = amdgpu.extract_slice %arg0 [0, 12] : tensor<64x16xf16, #blocked> to tensor<64x4xf16, #blocked>
    // CHECK: ttg.local_store [[SLICE_3]], [[SUBVIEW_3]] : tensor<64x4xf16, #blocked> -> !ttg.memdesc<64x4xf16, #shared, #smem, mutable, 1x64x16>
    // CHECK: amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    // CHECK: ttg.local_dealloc [[ALLOC]] : !ttg.memdesc<1x64x16xf16, #shared, #smem, mutable>
    %0 = ttg.local_alloc %arg0 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared, #smem>
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    tt.return
  }
}
