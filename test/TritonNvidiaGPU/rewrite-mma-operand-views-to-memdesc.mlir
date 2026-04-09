// RUN: triton-opt %s -split-input-file -triton-nvidia-rewrite-mma-operand-views-to-memdesc -canonicalize | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 16], threadsPerWarp = [1, 1, 4, 8, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [1, 4, 8, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [4, 8, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 0, 2]}>
#linear0 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [128, 0], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared5 = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0], [64, 0]]}, alignment = 128>
#smem = #ttg.shared_memory
#tmem0 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
// CHECK-DAG: #{{.*}} = #ttg.shared_linear<{{.*}}alignment = 128>
// CHECK-DAG: #{{.*}} = #ttg.shared_linear<{{.*}}alignment = 128>
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @swizzle0_operand_views
  // CHECK-DAG: %[[A_DESC:.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #{{.*}}>
  // CHECK-DAG: %[[A_BASE:.*]] = ttg.local_alloc %[[A_DESC]] : (tensor<128x128xf8E4M3FN, #{{.*}}>) -> !ttg.memdesc<128x128xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_DESC:.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<1x1x256x8x16xf8E4M3FN>> -> tensor<1x1x256x8x16xf8E4M3FN, #{{.*}}>
  // CHECK-DAG: %[[B_BASE:.*]] = ttg.local_alloc %[[B_DESC]] : (tensor<1x1x256x8x16xf8E4M3FN, #{{.*}}>) -> !ttg.memdesc<1x1x256x8x16xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_RS0:.*]] = ttg.memdesc_reshape %[[B_BASE]] : !ttg.memdesc<1x1x256x8x16xf8E4M3FN, #{{.*}}, #smem{{.*}}> -> !ttg.memdesc<8x32x8x16xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_TR0:.*]] = ttg.memdesc_trans %[[B_RS0]] {order = array<i32: 1, 2, 0, 3>} : !ttg.memdesc<8x32x8x16xf8E4M3FN, #{{.*}}, #smem{{.*}}> -> !ttg.memdesc<32x8x8x16xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_RS1:.*]] = ttg.memdesc_reshape %[[B_TR0]] : !ttg.memdesc<32x8x8x16xf8E4M3FN, #{{.*}}, #smem{{.*}}> -> !ttg.memdesc<256x128xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_TR1:.*]] = ttg.memdesc_trans %[[B_RS1]] {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf8E4M3FN, #{{.*}}, #smem{{.*}}> -> !ttg.memdesc<128x256xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-NOT: tt.reshape
  // CHECK-NOT: tt.trans
  // CHECK-NOT: ttg.local_load
  // CHECK: ttng.tc_gen5_mma %[[A_BASE]], %[[B_TR1]], %arg2, %true, %true
  tt.func @swizzle0_operand_views(
      %a_desc: !tt.tensordesc<tensor<128x128xf8E4M3FN>>,
      %b_desc: !tt.tensordesc<tensor<1x1x256x8x16xf8E4M3FN>>,
      %acc: !ttg.memdesc<128x256xf32, #tmem0, #ttng.tensor_memory>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked0>
    %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<tensor<1x1x256x8x16xf8E4M3FN>> -> tensor<1x1x256x8x16xf8E4M3FN, #blocked1>
    %b1 = tt.reshape %b : tensor<1x1x256x8x16xf8E4M3FN, #blocked1> -> tensor<8x32x8x16xf8E4M3FN, #blocked2>
    %b2 = tt.trans %b1 {order = array<i32: 1, 2, 0, 3>} : tensor<8x32x8x16xf8E4M3FN, #blocked2> -> tensor<32x8x8x16xf8E4M3FN, #blocked3>
    %b3 = tt.reshape %b2 : tensor<32x8x8x16xf8E4M3FN, #blocked3> -> tensor<256x128xf8E4M3FN, #linear0>
    %b4 = tt.trans %b3 {order = array<i32: 1, 0>} : tensor<256x128xf8E4M3FN, #linear0> -> tensor<128x256xf8E4M3FN, #linear2>
    %a_s = ttg.local_alloc %a : (tensor<128x128xf8E4M3FN, #blocked0>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared0, #smem>
    %b_s = ttg.local_alloc %b4 : (tensor<128x256xf8E4M3FN, #linear2>) -> !ttg.memdesc<128x256xf8E4M3FN, #shared5, #smem>
    ttng.tc_gen5_mma %a_s, %b_s, %acc, %true, %true : !ttg.memdesc<128x128xf8E4M3FN, #shared0, #smem>, !ttg.memdesc<128x256xf8E4M3FN, #shared5, #smem>, !ttg.memdesc<128x256xf32, #tmem0, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blockedA_desc = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB_desc = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 16], threadsPerWarp = [1, 1, 4, 8, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#blockedB0_desc = #ttg.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [1, 4, 8, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blockedB1_desc = #ttg.blocked<{sizePerThread = [1, 1, 1, 16], threadsPerWarp = [4, 8, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 0, 2]}>
#linear_desc = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [128, 0], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#mma_desc = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 32]}>
#sharedA_desc = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#sharedB3_desc = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16], [0, 32], [0, 64]]}, alignment = 128>
#sharedB4_desc = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0], [64, 0]]}, alignment = 128>
#smem = #ttg.shared_memory
// CHECK-DAG: #{{.*}} = #ttg.shared_linear<{{.*}}alignment = 128>
// CHECK-DAG: #{{.*}} = #ttg.shared_linear<{{.*}}alignment = 128>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @swizzle0_operand_views_warp_group_dot
  // CHECK-DAG: %[[B_DESC_LOAD:.*]] = tt.descriptor_load %arg1[%{{.*}}] : !tt.tensordesc<tensor<1x1x256x8x16xf8E4M3FN>> -> tensor<1x1x256x8x16xf8E4M3FN, #{{.*}}>
  // CHECK-DAG: %[[B_BASE:.*]] = ttg.local_alloc %[[B_DESC_LOAD]] : (tensor<1x1x256x8x16xf8E4M3FN, {{.*}}>) -> !ttg.memdesc<1x1x256x8x16xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_RS0:.*]] = ttg.memdesc_reshape %[[B_BASE]] : !ttg.memdesc<1x1x256x8x16xf8E4M3FN, #{{.*}}, #smem{{.*}}> -> !ttg.memdesc<8x32x8x16xf8E4M3FN, {{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_TR:.*]] = ttg.memdesc_trans %[[B_RS0]] {order = array<i32: 1, 2, 0, 3>} : !ttg.memdesc<8x32x8x16xf8E4M3FN, {{.*}}, #smem{{.*}}> -> !ttg.memdesc<32x8x8x16xf8E4M3FN, {{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_RS1:.*]] = ttg.memdesc_reshape %[[B_TR]] : !ttg.memdesc<32x8x8x16xf8E4M3FN, {{.*}}, #smem{{.*}}> -> !ttg.memdesc<256x128xf8E4M3FN, {{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_T:.*]] = ttg.memdesc_trans %[[B_RS1]] {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf8E4M3FN, {{.*}}, #smem{{.*}}> -> !ttg.memdesc<128x256xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-NOT: tt.reshape
  // CHECK-NOT: tt.trans
  // CHECK: %[[DOT:.*]] = ttng.warp_group_dot %{{.*}}, %[[B_T]], %arg2
  // CHECK: %[[WAIT:.*]]:3 = ttng.warp_group_dot_wait %[[DOT]], {{.*}}, %[[B_T]] {pendings = 0 : i32}
  tt.func @swizzle0_operand_views_warp_group_dot(
      %a_desc: !tt.tensordesc<tensor<128x128xf8E4M3FN, #sharedA_desc>>,
      %b_desc: !tt.tensordesc<tensor<1x1x256x8x16xf8E4M3FN>>,
      %acc: tensor<128x256xf32, #mma_desc>) -> tensor<128x256xf32, #mma_desc> {
    %c0 = arith.constant 0 : i32
    %a = tt.descriptor_load %a_desc[%c0, %c0] : !tt.tensordesc<tensor<128x128xf8E4M3FN, #sharedA_desc>> -> tensor<128x128xf8E4M3FN, #blockedA_desc>
    %a_s = ttg.local_alloc %a : (tensor<128x128xf8E4M3FN, #blockedA_desc>) -> !ttg.memdesc<128x128xf8E4M3FN, #sharedA_desc, #smem>
    %b_cm = tt.descriptor_load %b_desc[%c0, %c0, %c0, %c0, %c0] : !tt.tensordesc<tensor<1x1x256x8x16xf8E4M3FN>> -> tensor<1x1x256x8x16xf8E4M3FN, #blockedB_desc>
    %b0 = tt.reshape %b_cm : tensor<1x1x256x8x16xf8E4M3FN, #blockedB_desc> -> tensor<8x32x8x16xf8E4M3FN, #blockedB0_desc>
    %b1 = tt.trans %b0 {order = array<i32: 1, 2, 0, 3>} : tensor<8x32x8x16xf8E4M3FN, #blockedB0_desc> -> tensor<32x8x8x16xf8E4M3FN, #blockedB1_desc>
    %b2 = tt.reshape %b1 : tensor<32x8x8x16xf8E4M3FN, #blockedB1_desc> -> tensor<256x128xf8E4M3FN, #linear_desc>
    %b_s = ttg.local_alloc %b2 : (tensor<256x128xf8E4M3FN, #linear_desc>) -> !ttg.memdesc<256x128xf8E4M3FN, #sharedB3_desc, #smem>
    %b_t = ttg.memdesc_trans %b_s {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf8E4M3FN, #sharedB3_desc, #smem> -> !ttg.memdesc<128x256xf8E4M3FN, #sharedB4_desc, #smem>
    %r = ttng.warp_group_dot %a_s, %b_t, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 1073741824 : i32}
      : !ttg.memdesc<128x128xf8E4M3FN, #sharedA_desc, #smem> * !ttg.memdesc<128x256xf8E4M3FN, #sharedB4_desc, #smem> -> tensor<128x256xf32, #mma_desc>
    %w:3 = ttng.warp_group_dot_wait %r, %a_s, %b_t {pendings = 0 : i32} : tensor<128x256xf32, #mma_desc>, !ttg.memdesc<128x128xf8E4M3FN, #sharedA_desc, #smem>, !ttg.memdesc<128x256xf8E4M3FN, #sharedB4_desc, #smem>
    tt.return %w#0 : tensor<128x256xf32, #mma_desc>
  }
}
