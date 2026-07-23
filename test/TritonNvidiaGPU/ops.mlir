// RUN: triton-opt %s | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_f16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 2>
#tmem_int32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#tmem_lhs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_lhs_fp4 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_lhs_fp4_padded = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, fp4Padded = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#offsets = #ttg.slice<{dim = 0, parent = #blocked}>
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

  // CHECK-LABEL: @tcgen5_int8
  //       CHECK:   ttng.tc_gen5_mma {{.*}} {is_async, is_unsigned}
  //       CHECK:   ttng.tc_gen5_mma {{.*}} {is_unsigned}
  tt.func @tcgen5_int8(
                  %a: !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xi8, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xi32, #tmem_int32, #ttng.tensor_memory, mutable>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                  %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier[%barrierPred] {is_async, is_unsigned} :
       !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xi8, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xi32, #tmem_int32, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>

    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred {is_unsigned}:
       !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xi8, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xi32, #tmem_int32, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @tcgen5_scaled_tmem_lhs
  tt.func @tcgen5_scaled_tmem_lhs(
                  %a_fp8: !ttg.memdesc<128x128xf8E5M2, #tmem_lhs, #ttng.tensor_memory>,
                  %a_fp4_dense: !ttg.memdesc<128x64xi8, #tmem_lhs_fp4, #ttng.tensor_memory>,
                  %a_fp4_padded: !ttg.memdesc<128x64xi8, #tmem_lhs_fp4_padded, #ttng.tensor_memory>,
                  %b_fp8: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
                  %b_fp4: !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf32, #tmem_int32, #ttng.tensor_memory, mutable>,
                  %scale_a: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
                  %scale_b: !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>,
                  %accUse: i1,
                  %pred: i1) {
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} lhs = e5m2 rhs = e5m2
    ttng.tc_gen5_mma_scaled %a_fp8, %b_fp8, %c, %scale_a, %scale_b, %accUse, %pred lhs = e5m2 rhs = e5m2 :
       !ttg.memdesc<128x128xf8E5M2, #tmem_lhs, #ttng.tensor_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem_int32, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
       !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} lhs = e2m1 rhs = e2m1
    ttng.tc_gen5_mma_scaled %a_fp4_dense, %b_fp4, %c, %scale_a, %scale_b, %accUse, %pred lhs = e2m1 rhs = e2m1 :
       !ttg.memdesc<128x64xi8, #tmem_lhs_fp4, #ttng.tensor_memory>,
       !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem_int32, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
       !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} lhs = e2m1 rhs = e5m2
    ttng.tc_gen5_mma_scaled %a_fp4_padded, %b_fp8, %c, %scale_a, %scale_b, %accUse, %pred lhs = e2m1 rhs = e5m2 :
       !ttg.memdesc<128x64xi8, #tmem_lhs_fp4_padded, #ttng.tensor_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem_int32, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
       !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }

  // CHECK-LABEL: @async_tma_gather
  // CHECK-SAME: [[DESC:%arg[0-9]+]]:
  // CHECK-SAME: [[X_OFFSETS:%arg[0-9]+]]:
  // CHECK-SAME: [[Y_OFFSET:%arg[0-9]+]]:
  // CHECK-SAME: [[BAR:%arg[0-9]+]]:
  // CHECK-SAME: [[RESULT:%arg[0-9]+]]:
  // CHECK-SAME: [[PRED:%arg[0-9]+]]:
  tt.func @async_tma_gather(%desc: !tt.tensordesc<1x128xbf16, #shared>, %x_offsets: tensor<32xi32, #offsets>, %y_offset: i32,
                            %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                            %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                            %pred: i1) {
    // CHECK-NEXT: ttng.async_tma_gather [[DESC]][[[X_OFFSETS]], [[Y_OFFSET]]] [[RESULT]], [[BAR]], [[PRED]] : !tt.tensordesc<1x128xbf16, #shared>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared2, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared, #smem, mutable>, i1
    ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<1x128xbf16, #shared>, tensor<32xi32, #offsets>, i32, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
    tt.return
  }

  // CHECK-LABEL: @async_tma_scatter
  // CHECK-SAME: [[DESC:%arg[0-9]+]]:
  // CHECK-SAME: [[X_OFFSETS:%arg[0-9]+]]:
  // CHECK-SAME: [[Y_OFFSET:%arg[0-9]+]]:
  // CHECK-SAME: [[SRC:%arg[0-9]+]]:
  tt.func @async_tma_scatter(%desc: !tt.tensordesc<1x128xbf16, #shared>, %x_offsets: tensor<32xi32, #offsets>, %y_offset: i32,
                             %src: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>) {
    // CHECK-NEXT: ttng.async_tma_scatter [[DESC]][[[X_OFFSETS]], [[Y_OFFSET]]] [[SRC]] : !tt.tensordesc<1x128xbf16, #shared>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<32x128xbf16, #shared, #smem, mutable>
    ttng.async_tma_scatter %desc[%x_offsets, %y_offset] %src : !tt.tensordesc<1x128xbf16, #shared>, tensor<32xi32, #offsets>, i32, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>
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

#shared_cga = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @arrive_barrier_multicast_absent
  tt.func @arrive_barrier_multicast_absent(%alloc: !ttg.memdesc<2xi64, #shared_cga, #ttg.shared_memory, mutable>) {
    // CHECK: ttng.arrive_barrier %arg0, 1 :
    // CHECK-NOT: multicastCTA
    ttng.arrive_barrier %alloc, 1 : !ttg.memdesc<2xi64, #shared_cga, #ttg.shared_memory, mutable>
    tt.return
  }

  // Explicit multicastCTA = 0 is valid and elides on print like the default.
  // CHECK-LABEL: @arrive_barrier_multicast_zero
  tt.func @arrive_barrier_multicast_zero(%alloc: !ttg.memdesc<2xi64, #shared_cga, #ttg.shared_memory, mutable>) {
    // CHECK: ttng.arrive_barrier %arg0, 1 :
    // CHECK-NOT: multicastCTA
    ttng.arrive_barrier %alloc, 1 {multicastCTA = 0 : i32} : !ttg.memdesc<2xi64, #shared_cga, #ttg.shared_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @arrive_barrier_multicast
  tt.func @arrive_barrier_multicast(%alloc: !ttg.memdesc<2xi64, #shared_cga, #ttg.shared_memory, mutable>) {
    // CHECK: ttng.arrive_barrier %arg0, 1 {multicastCTA = 1 : i32} : !ttg.memdesc<2xi64,
    ttng.arrive_barrier %alloc, 1 {multicastCTA = 1 : i32} : !ttg.memdesc<2xi64, #shared_cga, #ttg.shared_memory, mutable>
    tt.return
  }

  // Round-trip the optional-pred-then-attr-dict spelling.
  // CHECK-LABEL: @arrive_barrier_multicast_pred
  tt.func @arrive_barrier_multicast_pred(%alloc: !ttg.memdesc<2xi64, #shared_cga, #ttg.shared_memory, mutable>, %pred: i1) {
    // CHECK: ttng.arrive_barrier %arg0, 1, %arg1 {multicastCTA = 1 : i32} : !ttg.memdesc<2xi64,
    ttng.arrive_barrier %alloc, 1, %pred {multicastCTA = 1 : i32} : !ttg.memdesc<2xi64, #shared_cga, #ttg.shared_memory, mutable>
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
  tt.func public @tma_load_im2col_3d(%desc: !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %off = arith.constant 1 : i16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0, %c0] offsets = [%off] %buf, %bar, %true : !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load_im2col_4d
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} offsets = [{{.*}}, {{.*}}] {{.*}} : !ttng.tensordesc_im2col
  tt.func public @tma_load_im2col_4d(%desc: !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %off1 = arith.constant 1 : i16
    %off2 = arith.constant 2 : i16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0, %c0, %c0] offsets = [%off1, %off2] %buf, %bar, %true : !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load_im2col_5d
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} offsets = [{{.*}}, {{.*}}, {{.*}}] {{.*}} : !ttng.tensordesc_im2col
  tt.func public @tma_load_im2col_5d(%desc: !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %off1 = arith.constant 1 : i16
    %off2 = arith.constant 2 : i16
    %off3 = arith.constant 3 : i16
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0, %c0, %c0, %c0] offsets = [%off1, %off2, %off3] %buf, %bar, %true : !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load_tiled_mode
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}}[{{.*}}, {{.*}}] %{{.*}}, %{{.*}}, {{.*}} : !tt.tensordesc
  // CHECK-NOT: offsets
  tt.func public @tma_load_tiled_mode(%desc: !tt.tensordesc<64x128xf16, #nvmma_128>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true : !tt.tensordesc<64x128xf16, #nvmma_128>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tensordesc_im2col
  // CHECK-SAME: !ttng.tensordesc_im2col<64x128xf16, {{.*}}>
  tt.func public @tensordesc_im2col(%desc: !ttng.tensordesc_im2col<64x128xf16, #nvmma_128>) {
    // CHECK: tt.return
    tt.return
  }
}

// Packed arithmetic keeps scalar tensor semantics in TTNGIR and only groups
// adjacent lanes when lowering to PTX.
#packed_blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith
  // CHECK: ttng.packed_arith add, f32x2, [f32x2, f32x2], {{.*}} axis = 1
  // CHECK: ttng.packed_arith mul, f16x2, [f16x2, f16x2], {{.*}} axis = 1
  // CHECK: ttng.packed_arith fma, bf16x2, [bf16x2, bf16x2, bf16x2], {{.*}} axis = 1
  // CHECK: ttng.packed_arith add, f32x2, [f16x2, f32x2], {{.*}} axis = 1
  // CHECK: ttng.packed_arith sub, bf16x2, [f32x2, f32x2], {{.*}} axis = 1
  // CHECK: ttng.packed_arith mul, f16x2, [f16x2, bf16x2], {{.*}} axis = 1
  // CHECK: ttng.packed_arith fma, f32x2, [bf16x2, f32x2, f32x2], {{.*}} axis = 1
  // CHECK: ttng.packed_arith min, f16x2, [f16x2, f16x2], {{.*}} axis = 1
  // CHECK: ttng.packed_arith max, bf16x2, [bf16x2, bf16x2], {{.*}} axis = 1
  tt.func @packed_arith(
      %f32a: tensor<128x2xf32, #packed_blocked>,
      %f32b: tensor<128x2xf32, #packed_blocked>,
      %f32c: tensor<128x2xf32, #packed_blocked>,
      %f16a: tensor<128x2xf16, #packed_blocked>,
      %f16b: tensor<128x2xf16, #packed_blocked>,
      %bf16a: tensor<128x2xbf16, #packed_blocked>,
      %bf16b: tensor<128x2xbf16, #packed_blocked>,
      %bf16c: tensor<128x2xbf16, #packed_blocked>) {
    %f32 = ttng.packed_arith add, f32x2, [f32x2, f32x2], %f32a, %f32b axis = 1 : (tensor<128x2xf32, #packed_blocked>, tensor<128x2xf32, #packed_blocked>) -> tensor<128x2xf32, #packed_blocked>
    %f16 = ttng.packed_arith mul, f16x2, [f16x2, f16x2], %f16a, %f16b axis = 1 : (tensor<128x2xf16, #packed_blocked>, tensor<128x2xf16, #packed_blocked>) -> tensor<128x2xf16, #packed_blocked>
    %bf16 = ttng.packed_arith fma, bf16x2, [bf16x2, bf16x2, bf16x2], %bf16a, %bf16b, %bf16c axis = 1 : (tensor<128x2xbf16, #packed_blocked>, tensor<128x2xbf16, #packed_blocked>, tensor<128x2xbf16, #packed_blocked>) -> tensor<128x2xbf16, #packed_blocked>
    %up = ttng.packed_arith add, f32x2, [f16x2, f32x2], %f16a, %f32b axis = 1 : (tensor<128x2xf16, #packed_blocked>, tensor<128x2xf32, #packed_blocked>) -> tensor<128x2xf32, #packed_blocked>
    %down = ttng.packed_arith sub, bf16x2, [f32x2, f32x2], %f32a, %f32b axis = 1 : (tensor<128x2xf32, #packed_blocked>, tensor<128x2xf32, #packed_blocked>) -> tensor<128x2xbf16, #packed_blocked>
    %cross = ttng.packed_arith mul, f16x2, [f16x2, bf16x2], %f16a, %bf16b axis = 1 : (tensor<128x2xf16, #packed_blocked>, tensor<128x2xbf16, #packed_blocked>) -> tensor<128x2xf16, #packed_blocked>
    %mixed_fma = ttng.packed_arith fma, f32x2, [bf16x2, f32x2, f32x2], %bf16a, %f32b, %f32c axis = 1 : (tensor<128x2xbf16, #packed_blocked>, tensor<128x2xf32, #packed_blocked>, tensor<128x2xf32, #packed_blocked>) -> tensor<128x2xf32, #packed_blocked>
    %min = ttng.packed_arith min, f16x2, [f16x2, f16x2], %f16a, %f16b axis = 1 : (tensor<128x2xf16, #packed_blocked>, tensor<128x2xf16, #packed_blocked>) -> tensor<128x2xf16, #packed_blocked>
    %max = ttng.packed_arith max, bf16x2, [bf16x2, bf16x2], %bf16a, %bf16b axis = 1 : (tensor<128x2xbf16, #packed_blocked>, tensor<128x2xbf16, #packed_blocked>) -> tensor<128x2xbf16, #packed_blocked>
    tt.return
  }
}

#tmem_window = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, CGALayout = [[1, 0]]>
#tmem_window_broadcast = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, CGALayout = [[0, 0]]>
#tmem_window_scale_a = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
#tmem_window_scale_b = #ttng.tensor_memory_scales_encoding<CGALayout = [[0, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tmem_subslice_reinterpret_scales
  // CHECK: ttng.tmem_subslice {{.*}} {offset = 208 : i32}
  // CHECK: ttg.memdesc_reinterpret
  // CHECK: ttng.tmem_subslice {{.*}} {offset = 464 : i32}
  // CHECK: ttg.memdesc_reinterpret
  // CHECK: ttng.tmem_subslice {{.*}} {offset = 480 : i32}
  // CHECK: ttg.memdesc_reinterpret
  tt.func public @tmem_subslice_reinterpret_scales(%arg0: !ttg.memdesc<256x512xf32, #tmem_window, #ttng.tensor_memory, mutable>) {
    %window = ttng.tmem_subslice %arg0 {offset = 208 : i32} : !ttg.memdesc<256x512xf32, #tmem_window, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x256xf32, #tmem_window, #ttng.tensor_memory, mutable, 256x512>
    %acc = ttg.memdesc_reinterpret %window : !ttg.memdesc<256x256xf32, #tmem_window, #ttng.tensor_memory, mutable, 256x512> -> !ttg.memdesc<256x256xf32, #tmem_window, #ttng.tensor_memory, mutable>
    %scale_a_base = ttng.tmem_subslice %arg0 {offset = 464 : i32} : !ttg.memdesc<256x512xf32, #tmem_window, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x16xf32, #tmem_window, #ttng.tensor_memory, mutable, 256x512>
    %scale_a = ttg.memdesc_reinterpret %scale_a_base : !ttg.memdesc<256x16xf32, #tmem_window, #ttng.tensor_memory, mutable, 256x512> -> !ttg.memdesc<256x16xi8, #tmem_window_scale_a, #ttng.tensor_memory, mutable>
    %scale_b_base = ttng.tmem_subslice %arg0 {offset = 480 : i32} : !ttg.memdesc<256x512xf32, #tmem_window, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x32xf32, #tmem_window, #ttng.tensor_memory, mutable, 256x512>
    %scale_b = ttg.memdesc_reinterpret %scale_b_base : !ttg.memdesc<256x32xf32, #tmem_window, #ttng.tensor_memory, mutable, 256x512> -> !ttg.memdesc<256x16xi8, #tmem_window_scale_b, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_reinterpret_sharding
  tt.func public @tmem_reinterpret_sharding(%broadcast: !ttg.memdesc<256x128xf32, #tmem_window_broadcast, #ttng.tensor_memory, mutable>) {
    %sharded = ttg.memdesc_reinterpret %broadcast : !ttg.memdesc<256x128xf32, #tmem_window_broadcast, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x128xf32, #tmem_window, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// The alternate packed instructions use byte-sized FP8 elements. E2M1 is
// stored as two logical values per i8 and has compact/padded register forms.
#packed_x4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#packed_fp4 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith_alternate_types
  // CHECK: ttng.packed_arith add, e4m3x4, [e5m2x4, e4m3x4],
  // CHECK: ttng.packed_arith sub, e5m2x4, [e2m1x4, e5m2x4],
  // CHECK: ttng.packed_arith mul, e4m3x4, [e2m1x4, e5m2x4],
  // CHECK: ttng.packed_arith fma, e5m2x4, [e2m1p4x4, e4m3x4, e5m2x4],
  tt.func @packed_arith_alternate_types(
      %e5m2: tensor<128x4xf8E5M2, #packed_x4>,
      %e4m3: tensor<128x4xf8E4M3FN, #packed_x4>,
      %e2m1: tensor<128x2xi8, #packed_fp4>) {
    %add = ttng.packed_arith add, e4m3x4, [e5m2x4, e4m3x4], %e5m2, %e4m3 axis = 1 : (tensor<128x4xf8E5M2, #packed_x4>, tensor<128x4xf8E4M3FN, #packed_x4>) -> tensor<128x4xf8E4M3FN, #packed_x4>
    %sub = ttng.packed_arith sub, e5m2x4, [e2m1x4, e5m2x4], %e2m1, %e5m2 axis = 1 : (tensor<128x2xi8, #packed_fp4>, tensor<128x4xf8E5M2, #packed_x4>) -> tensor<128x4xf8E5M2, #packed_x4>
    %mul = ttng.packed_arith mul, e4m3x4, [e2m1x4, e5m2x4], %e2m1, %e5m2 axis = 1 : (tensor<128x2xi8, #packed_fp4>, tensor<128x4xf8E5M2, #packed_x4>) -> tensor<128x4xf8E4M3FN, #packed_x4>
    %fma = ttng.packed_arith fma, e5m2x4, [e2m1p4x4, e4m3x4, e5m2x4], %e2m1, %e4m3, %e5m2 axis = 1 : (tensor<128x2xi8, #packed_fp4>, tensor<128x4xf8E4M3FN, #packed_x4>, tensor<128x4xf8E5M2, #packed_x4>) -> tensor<128x4xf8E5M2, #packed_x4>
    tt.return
  }
}

// Packed arithmetic accepts register permutations and broadcast bases as long
// as every thread owns each logical lane in a pack.
#packed_perm = #ttg.linear<{register = [[2], [1]], lane = [[4], [8], [16], [32], [64]], warp = [[128], [256]], block = []}>
#packed_bcast = #ttg.linear<{register = [[0], [1]], lane = [[2], [4], [8], [16], [32]], warp = [[64], [128]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith_register_layouts
  // CHECK: ttng.packed_arith add, f32x2, [f32x2, f32x2],
  // CHECK: ttng.packed_arith mul, f32x2, [f32x2, f32x2],
  tt.func @packed_arith_register_layouts(
      %pa: tensor<512xf32, #packed_perm>,
      %pb: tensor<512xf32, #packed_perm>,
      %ba: tensor<256xf32, #packed_bcast>,
      %bb: tensor<256xf32, #packed_bcast>) {
    %perm = ttng.packed_arith add, f32x2, [f32x2, f32x2], %pa, %pb axis = 0 : (tensor<512xf32, #packed_perm>, tensor<512xf32, #packed_perm>) -> tensor<512xf32, #packed_perm>
    %bcast = ttng.packed_arith mul, f32x2, [f32x2, f32x2], %ba, %bb axis = 0 : (tensor<256xf32, #packed_bcast>, tensor<256xf32, #packed_bcast>) -> tensor<256xf32, #packed_bcast>
    tt.return
  }
}

#tmem_slice = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tmem_subslice_source_layout_contiguous
  // CHECK: ttng.tmem_subslice {{.*}} {offset = 32 : i32}
  tt.func public @tmem_subslice_source_layout_contiguous(%arg0: !ttg.memdesc<128x256xf32, #tmem_slice, #ttng.tensor_memory, mutable>) {
    %sub = ttng.tmem_subslice %arg0 {offset = 32 : i32} : !ttg.memdesc<128x256xf32, #tmem_slice, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem_slice, #ttng.tensor_memory, mutable, 128x256>
    tt.return
  }

  // CHECK-LABEL: @tmem_subslice_source_layout_contiguous_m_rep
  // CHECK: ttng.tmem_subslice {{.*}} {offset = 16 : i32}
  tt.func public @tmem_subslice_source_layout_contiguous_m_rep(%arg0: !ttg.memdesc<256x128xf32, #tmem_slice, #ttng.tensor_memory, mutable>) {
    %sub = ttng.tmem_subslice %arg0 {offset = 16 : i32} : !ttg.memdesc<256x128xf32, #tmem_slice, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x32xf32, #tmem_slice, #ttng.tensor_memory, mutable, 256x128>
    tt.return
  }
}
