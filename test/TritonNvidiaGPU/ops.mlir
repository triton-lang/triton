// RUN: triton-opt %s | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_f16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 2>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
#scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

  // CHECK-LABEL: @tcgen5
  //       CHECK:   ttng.tc_gen5_mma
  //       CHECK:   ttng.tc_gen5_mma
  tt.func @tcgen5(%a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                  %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>

    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred:
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @async_tma_gather
  // CHECK-SAME: [[DESC:%arg[0-9]+]]:
  // CHECK-SAME: [[X_OFFSETS:%arg[0-9]+]]:
  // CHECK-SAME: [[Y_OFFSET:%arg[0-9]+]]:
  // CHECK-SAME: [[BAR:%arg[0-9]+]]:
  // CHECK-SAME: [[RESULT:%arg[0-9]+]]:
  // CHECK-SAME: [[PRED:%arg[0-9]+]]:
  tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                            %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                            %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                            %pred: i1) {
    // CHECK-NEXT: ttng.async_tma_gather [[DESC]][[[X_OFFSETS]], [[Y_OFFSET]]] [[RESULT]], [[BAR]], [[PRED]] : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared2, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared, #smem, mutable>, i1
    ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
    tt.return
  }

  // CHECK-LABEL: @async_tma_scatter
  // CHECK-SAME: [[DESC:%arg[0-9]+]]:
  // CHECK-SAME: [[X_OFFSETS:%arg[0-9]+]]:
  // CHECK-SAME: [[Y_OFFSET:%arg[0-9]+]]:
  // CHECK-SAME: [[SRC:%arg[0-9]+]]:
  tt.func @async_tma_scatter(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                             %src: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>) {
    // CHECK-NEXT: ttng.async_tma_scatter [[DESC]][[[X_OFFSETS]], [[Y_OFFSET]]] [[SRC]] : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<32x128xbf16, #shared, #smem, mutable>
    ttng.async_tma_scatter %desc[%x_offsets, %y_offset] %src : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @wait_barrier
  // CHECK-SAME: [[ALLOC:%arg[0-9]+]]:
  // CHECK-SAME: [[PHASE:%arg[0-9]+]]:
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, %phase: i32) {
    // CHECK-NEXT: ttng.wait_barrier [[ALLOC]], [[PHASE]] : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @wait_barrier
  // CHECK-SAME: [[ALLOC:%arg[0-9]+]]:
  // CHECK-SAME: [[PHASE:%arg[0-9]+]]:
  // CHECK-SAME: [[DEP1:%arg[0-9]+]]:
  // CHECK-SAME: [[DEP2:%arg[0-9]+]]:
  tt.func @wait_barrier_deps(%alloc: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, %phase: i32, %dep1: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, %dep2: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory, mutable>) {
    // CHECK-NEXT: ttng.wait_barrier [[ALLOC]], [[PHASE]] deps [[DEP1]], [[DEP2]] : !ttg.memdesc<1xi64, #shared2, #smem, mutable>, !ttg.memdesc<1xi64, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable>
    ttng.wait_barrier %alloc, %phase deps %dep1, %dep2 : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @arrive_barrier
  tt.func @arrive_barrier(%alloc: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, %pred: i1) {
    // CHECK-NEXT: ttng.arrive_barrier %arg0, 2 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.arrive_barrier %alloc, 2 : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    // CHECK-NEXT: ttng.arrive_barrier %arg0, 2, %arg1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.arrive_barrier %alloc, 2, %pred : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }

  tt.func @scale_encoding(%arg0: tensor<128x8xi8, #scales>, %arg1: tensor<128x8xf8E5M2, #scales>) {
    %0 = ttng.tmem_alloc %arg0 : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %1 = ttng.tmem_alloc %arg1 : (tensor<128x8xf8E5M2, #scales>) -> !ttg.memdesc<128x8xf8E5M2, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }
}

// Tests for TMA im2col (3D/4D/5D) and tiled mode
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tma_load_im2col_3d
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} offsets = [{{.*}}] {{.*}} : !ttng.tensordesc_im2col
  tt.func public @tma_load_im2col_3d(%desc: !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %off = arith.constant 1 : i16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0, %c0] offsets = [%off] %buf, %bar, %true : !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load_im2col_4d
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} offsets = [{{.*}}, {{.*}}] {{.*}} : !ttng.tensordesc_im2col
  tt.func public @tma_load_im2col_4d(%desc: !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %off1 = arith.constant 1 : i16
    %off2 = arith.constant 2 : i16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0, %c0, %c0] offsets = [%off1, %off2] %buf, %bar, %true : !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load_im2col_5d
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} offsets = [{{.*}}, {{.*}}, {{.*}}] {{.*}} : !ttng.tensordesc_im2col
  tt.func public @tma_load_im2col_5d(%desc: !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %off1 = arith.constant 1 : i16
    %off2 = arith.constant 2 : i16
    %off3 = arith.constant 3 : i16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0, %c0, %c0, %c0] offsets = [%off1, %off2, %off3] %buf, %bar, %true : !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load_tiled_mode
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}}[{{.*}}, {{.*}}] %{{.*}}, %{{.*}}, {{.*}} : !tt.tensordesc
  // CHECK-NOT: offsets
  tt.func public @tma_load_tiled_mode(%desc: !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true : !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tensordesc_im2col
  // CHECK-SAME: !ttng.tensordesc_im2col<tensor<64x128xf16, {{.*}}>>
  tt.func public @tensordesc_im2col(%desc: !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>) {
    // CHECK: tt.return
    tt.return
  }
}
