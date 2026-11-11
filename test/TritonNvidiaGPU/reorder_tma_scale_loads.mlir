// RUN: triton-opt %s --triton-nvidia-reorder-tma-scale-loads | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#scale_blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#a_scale = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[8, 0], [0, 0], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#b_scale_load = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [0, 128]], lane = [[0, 0], [0, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 8], [0, 16], [0, 0]], block = []}>
#b_scale = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [128, 0]], lane = [[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 3}>
#out_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.target" = "cuda:120", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @scaled_dot_rhs_load_order(
  // CHECK-NOT: tt.descriptor_load %arg0
  // CHECK: tt.descriptor_load %arg1
  // CHECK: tt.descriptor_load %arg0
  tt.func public @scaled_dot_rhs_load_order(
      %b_desc: !tt.tensordesc<1x128x256xf8E4M3FN, #shared>,
      %scale_desc: !tt.tensordesc<1x4x256xi8, #shared>,
      %offset: i32,
      %a: tensor<128x128xf8E4M3FN, #a>,
      %a_scale: tensor<128x4xi8, #a_scale>,
      %acc: tensor<128x256xf32, #mma>) -> tensor<128x256xf32, #mma> {
    %b = tt.descriptor_load %b_desc[%offset, %offset, %offset] : !tt.tensordesc<1x128x256xf8E4M3FN, #shared> -> tensor<128x256xf8E4M3FN, #blocked>
    %scale = tt.descriptor_load %scale_desc[%offset, %offset, %offset] : !tt.tensordesc<1x4x256xi8, #shared> -> tensor<4x256xi8, #scale_blocked>
    %scale_converted = ttg.convert_layout %scale : tensor<4x256xi8, #scale_blocked> -> tensor<4x256xi8, #b_scale_load>
    %b_scale = tt.trans %scale_converted {order = array<i32: 1, 0>} : tensor<4x256xi8, #b_scale_load> -> tensor<256x4xi8, #b_scale>
    %b_dot = ttg.convert_layout %b : tensor<128x256xf8E4M3FN, #blocked> -> tensor<128x256xf8E4M3FN, #b>
    %out = tt.dot_scaled %a scale %a_scale, %b_dot scale %b_scale, %acc lhs = e4m3 rhs = e4m3 {fastMath = false}
      : tensor<128x128xf8E4M3FN, #a>, tensor<128x4xi8, #a_scale>
        * tensor<128x256xf8E4M3FN, #b>, tensor<256x4xi8, #b_scale>
        -> tensor<128x256xf32, #mma>
    tt.return %out : tensor<128x256xf32, #mma>
  }

  // CHECK-LABEL: @scaled_dot_lhs_load_order(
  // CHECK-NOT: tt.descriptor_load %arg0
  // CHECK: tt.descriptor_load %arg1
  // CHECK: tt.descriptor_load %arg0
  tt.func public @scaled_dot_lhs_load_order(
      %a_desc: !tt.tensordesc<1x128x128xf8E4M3FN, #shared>,
      %scale_desc: !tt.tensordesc<1x128x4xi8, #shared>,
      %offset: i32,
      %b: tensor<128x256xf8E4M3FN, #b>,
      %b_scale: tensor<256x4xi8, #b_scale>,
      %acc: tensor<128x256xf32, #mma>) -> tensor<128x256xf32, #mma> {
    %a = tt.descriptor_load %a_desc[%offset, %offset, %offset] : !tt.tensordesc<1x128x128xf8E4M3FN, #shared> -> tensor<128x128xf8E4M3FN, #blocked>
    %scale = tt.descriptor_load %scale_desc[%offset, %offset, %offset] : !tt.tensordesc<1x128x4xi8, #shared> -> tensor<128x4xi8, #scale_blocked>
    %a_scale = ttg.convert_layout %scale : tensor<128x4xi8, #scale_blocked> -> tensor<128x4xi8, #a_scale>
    %a_dot = ttg.convert_layout %a : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #a>
    %out = tt.dot_scaled %a_dot scale %a_scale, %b scale %b_scale, %acc lhs = e4m3 rhs = e4m3 {fastMath = false}
      : tensor<128x128xf8E4M3FN, #a>, tensor<128x4xi8, #a_scale>
        * tensor<128x256xf8E4M3FN, #b>, tensor<256x4xi8, #b_scale>
        -> tensor<128x256xf32, #mma>
    tt.return %out : tensor<128x256xf32, #mma>
  }

  // The store may overwrite the data read by the operand load.
  // CHECK-LABEL: @scaled_dot_load_order_with_store(
  // CHECK: tt.descriptor_load %arg0
  // CHECK: tt.descriptor_store %arg0
  // CHECK: tt.descriptor_load %arg1
  tt.func public @scaled_dot_load_order_with_store(
      %b_desc: !tt.tensordesc<1x128x256xf8E4M3FN, #shared>,
      %scale_desc: !tt.tensordesc<1x4x256xi8, #shared>,
      %offset: i32,
      %a: tensor<128x128xf8E4M3FN, #a>,
      %a_scale: tensor<128x4xi8, #a_scale>,
      %acc: tensor<128x256xf32, #mma>,
      %replacement: tensor<128x256xf8E4M3FN, #blocked>) -> tensor<128x256xf32, #mma> {
    %b = tt.descriptor_load %b_desc[%offset, %offset, %offset] : !tt.tensordesc<1x128x256xf8E4M3FN, #shared> -> tensor<128x256xf8E4M3FN, #blocked>
    tt.descriptor_store %b_desc[%offset, %offset, %offset], %replacement : !tt.tensordesc<1x128x256xf8E4M3FN, #shared>, tensor<128x256xf8E4M3FN, #blocked>
    %scale = tt.descriptor_load %scale_desc[%offset, %offset, %offset] : !tt.tensordesc<1x4x256xi8, #shared> -> tensor<4x256xi8, #scale_blocked>
    %scale_converted = ttg.convert_layout %scale : tensor<4x256xi8, #scale_blocked> -> tensor<4x256xi8, #b_scale_load>
    %b_scale = tt.trans %scale_converted {order = array<i32: 1, 0>} : tensor<4x256xi8, #b_scale_load> -> tensor<256x4xi8, #b_scale>
    %b_dot = ttg.convert_layout %b : tensor<128x256xf8E4M3FN, #blocked> -> tensor<128x256xf8E4M3FN, #b>
    %out = tt.dot_scaled %a scale %a_scale, %b_dot scale %b_scale, %acc lhs = e4m3 rhs = e4m3 {fastMath = false}
      : tensor<128x128xf8E4M3FN, #a>, tensor<128x4xi8, #a_scale>
        * tensor<128x256xf8E4M3FN, #b>, tensor<256x4xi8, #b_scale>
        -> tensor<128x256xf32, #mma>
    tt.return %out : tensor<128x256xf32, #mma>
  }
  // CHECK-LABEL: @scaled_dot_warp_specialized(
  // CHECK: partition0(%[[DATA:[a-zA-Z0-9_]+]]: !tt.tensordesc<1x128x256xf8E4M3FN, {{[^>]+}}>, %[[SCALE:[a-zA-Z0-9_]+]]:
  // CHECK: tt.descriptor_load %[[DATA]]
  // CHECK: tt.descriptor_load %[[SCALE]]
  tt.func public @scaled_dot_warp_specialized(
      %b_desc: !tt.tensordesc<1x128x256xf8E4M3FN, #shared>,
      %scale_desc: !tt.tensordesc<1x4x256xi8, #shared>,
      %offset: i32,
      %a: tensor<128x128xf8E4M3FN, #a>,
      %a_scale: tensor<128x4xi8, #a_scale>,
      %acc: tensor<128x256xf32, #mma>,
      %out_desc: !tt.tensordesc<128x256xf32, #out_shared>) {
    ttg.warp_specialize(%b_desc, %scale_desc, %offset, %a, %a_scale, %acc, %out_desc)
    default {
      ttg.warp_yield
    }
    partition0(%b_desc_p: !tt.tensordesc<1x128x256xf8E4M3FN, #shared>, %scale_desc_p: !tt.tensordesc<1x4x256xi8, #shared>, %offset_p: i32, %a_p: tensor<128x128xf8E4M3FN, #a>, %a_scale_p: tensor<128x4xi8, #a_scale>, %acc_p: tensor<128x256xf32, #mma>, %out_desc_p: !tt.tensordesc<128x256xf32, #out_shared>) num_warps(8) {
      %b = tt.descriptor_load %b_desc_p[%offset_p, %offset_p, %offset_p] : !tt.tensordesc<1x128x256xf8E4M3FN, #shared> -> tensor<128x256xf8E4M3FN, #blocked>
      %scale = tt.descriptor_load %scale_desc_p[%offset_p, %offset_p, %offset_p] : !tt.tensordesc<1x4x256xi8, #shared> -> tensor<4x256xi8, #scale_blocked>
      %scale_converted = ttg.convert_layout %scale : tensor<4x256xi8, #scale_blocked> -> tensor<4x256xi8, #b_scale_load>
      %b_scale = tt.trans %scale_converted {order = array<i32: 1, 0>} : tensor<4x256xi8, #b_scale_load> -> tensor<256x4xi8, #b_scale>
      %b_dot = ttg.convert_layout %b : tensor<128x256xf8E4M3FN, #blocked> -> tensor<128x256xf8E4M3FN, #b>
      %out = tt.dot_scaled %a_p scale %a_scale_p, %b_dot scale %b_scale, %acc_p lhs = e4m3 rhs = e4m3 {fastMath = false}
        : tensor<128x128xf8E4M3FN, #a>, tensor<128x4xi8, #a_scale>
          * tensor<128x256xf8E4M3FN, #b>, tensor<256x4xi8, #b_scale>
          -> tensor<128x256xf32, #mma>
      tt.descriptor_store %out_desc_p[%offset_p, %offset_p], %out : !tt.tensordesc<128x256xf32, #out_shared>, tensor<128x256xf32, #mma>
      ttg.warp_return
    } : (!tt.tensordesc<1x128x256xf8E4M3FN, #shared>, !tt.tensordesc<1x4x256xi8, #shared>, i32, tensor<128x128xf8E4M3FN, #a>, tensor<128x4xi8, #a_scale>, tensor<128x256xf32, #mma>, !tt.tensordesc<128x256xf32, #out_shared>) -> ()
    tt.return
  }

}
