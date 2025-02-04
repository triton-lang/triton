// RUN: triton-opt %s -split-input-file -tritongpu-optimize-dot-operands -canonicalize | FileCheck %s


#blockedA = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_operand_A
//    CHECK: %[[A:.+]] = ttg.convert_layout %{{.*}} : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: ttng.warp_group_dot %[[A]], {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
tt.func @mma_v3_reg_operand_A(%arg0: tensor<128x64xf16, #mma>, %arg1: !ttg.memdesc<64x64xf16, #shared, #smem>, %arg2: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
  %A = ttg.local_alloc %arg0 : (tensor<128x64xf16, #mma>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
  %r = ttng.warp_group_dot %A, %arg1, %arg2 : !ttg.memdesc<128x64xf16, #shared1, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.return %r : tensor<128x64xf32, #mma>
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_operand_A_fp8
//    CHECK: %[[A:.+]] = ttg.convert_layout %{{.*}} : tensor<128x64xf8E5M2, #mma> -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
//    CHECK: ttng.warp_group_dot %[[A]], {{.*}} : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * !ttg.memdesc<64x64xf8E5M2, #shared, #smem> -> tensor<128x64xf32, #mma>
tt.func @mma_v3_reg_operand_A_fp8(%arg0: tensor<128x64xf8E5M2, #mma>, %arg1: !ttg.memdesc<64x64xf8E5M2, #shared, #smem>, %arg2: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
  %A = ttg.local_alloc %arg0 : (tensor<128x64xf8E5M2, #mma>) -> !ttg.memdesc<128x64xf8E5M2, #shared1, #smem>
  %r = ttng.warp_group_dot %A, %arg1, %arg2 : !ttg.memdesc<128x64xf8E5M2, #shared1, #smem> * !ttg.memdesc<64x64xf8E5M2, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.return %r : tensor<128x64xf32, #mma>
}
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @a_impl
// CHECK-NOT: %[[SELECT:.*]] = arith.select {{.*}} : tensor<128x128xi1, #ttg.dot_op<{{.*}}>, tensor<128x128xf16, #ttg.dot_op<{{.*}}>
  tt.func @a_impl(%pa: tensor<128x128x!tt.ptr<f16>, #blocked>) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_3 = arith.constant dense<5> : tensor<128x1xi32, #blocked>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %tl = tt.load %pa : tensor<128x128x!tt.ptr<f16>, #blocked>
    %tr = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %te = tt.expand_dims %tr {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %tc = arith.cmpi slt, %te, %cst_3 : tensor<128x1xi32, #blocked>
    %tb = tt.broadcast %tc : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %ts = arith.select %tb, %tl, %cst_4 : tensor<128x128xi1, #blocked>, tensor<128x128xf16, #blocked>
    %conv = ttg.convert_layout %ts : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %td = tt.dot %cst_0, %conv, %cst : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    tt.return %td : tensor<128x128xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_push_elementwise
//    CHECK: %[[A_BLOCK:.*]] = tt.load %{{.*}} : tensor<128x64x!tt.ptr<bf16>, #blocked>
//    CHECK: %[[A_DOTOP:.*]] = ttg.convert_layout %[[A_BLOCK]] : tensor<128x64xbf16, #blocked> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_CASTED:.*]] = tt.fp_to_fp %[[A_DOTOP]] : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[R:.*]] = ttng.warp_group_dot %[[A_CASTED]], %{{.*}}, %{{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.func @mma_v3_reg_push_elementwise(%pa: tensor<128x64x!tt.ptr<bf16>, #blocked>, %dotb: !ttg.memdesc<64x64xf16, #shared, #smem>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %a_bf16 = tt.load %pa : tensor<128x64x!tt.ptr<bf16>, #blocked>
    %a = tt.fp_to_fp %a_bf16 : tensor<128x64xbf16, #blocked> -> tensor<128x64xf16, #blocked>
    %dota = ttg.local_alloc %a: (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    %r = ttng.warp_group_dot %dota, %dotb, %dotc : !ttg.memdesc<128x64xf16, #shared1, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_push_elementwise_chained
//    CHECK: %[[CST_DOTOP:.*]] = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_BLOCK:.*]] = tt.load %{{.*}} : tensor<128x64x!tt.ptr<i8>, #blocked>
//    CHECK: %[[A_DOTOP:.*]] = ttg.convert_layout %[[A_BLOCK]] : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_CASTED:.*]] = arith.sitofp %[[A_DOTOP]] : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> to tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_SCALED:.*]] = arith.mulf %[[A_CASTED]], %[[CST_DOTOP]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_NEGATED:.*]] = arith.negf %[[A_SCALED]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[R:.*]] = ttng.warp_group_dot %[[A_NEGATED]], %{{.*}}, %{{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.func @mma_v3_reg_push_elementwise_chained(%pa: tensor<128x64x!tt.ptr<i8>, #blocked>, %dotb: !ttg.memdesc<64x64xf16, #shared, #smem>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %a_i8 = tt.load %pa : tensor<128x64x!tt.ptr<i8>, #blocked>
    %a_f16 = arith.sitofp %a_i8 : tensor<128x64xi8, #blocked> to tensor<128x64xf16, #blocked>
    %a_scaled = arith.mulf %a_f16, %cst : tensor<128x64xf16, #blocked>
    %a_negated = arith.negf %a_scaled : tensor<128x64xf16, #blocked>
    %dota = ttg.local_alloc %a_negated: (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    %r = ttng.warp_group_dot %dota, %dotb, %dotc : !ttg.memdesc<128x64xf16, #shared1, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: mma_reorder_transpose
// CHECK: ttg.local_alloc
// CHECK: ttg.memdesc_trans
// CHECK: ttng.warp_group_dot
  tt.func @mma_reorder_transpose(%t: tensor<64x128xf16, #blocked1>, %dotb: !ttg.memdesc<64x64xf16, #shared, #smem>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %a = tt.trans %t {order = array<i32: 1, 0>} : tensor<64x128xf16, #blocked1> -> tensor<128x64xf16, #blocked>
    %dota = ttg.local_alloc %a: (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    %r = ttng.warp_group_dot %dota, %dotb, %dotc : !ttg.memdesc<128x64xf16, #shared1, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: mmav2_reorder_transpose
// CHECK: ttg.local_alloc
// CHECK: ttg.memdesc_trans
// CHECK: ttg.local_load
// CHECK: tt.dot
  tt.func @mmav2_reorder_transpose(%t: tensor<32x128xf16, #blocked1>, %dotb: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %a = tt.trans %t {order = array<i32: 1, 0>} : tensor<32x128xf16, #blocked1> -> tensor<128x32xf16, #blocked>
    %cv = ttg.convert_layout %a : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %r = tt.dot %cv, %dotb, %dotc, inputPrecision = tf32 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: mmav2_transpose_indirect
// CHECK: tt.trans
// CHECK: ttg.convert_layout
// CHECK: arith.addf
// CHECK: tt.dot
  tt.func @mmav2_transpose_indirect(%t: tensor<32x128xf16, #blocked1>, %dotb: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %a = tt.trans %t {order = array<i32: 1, 0>} : tensor<32x128xf16, #blocked1> -> tensor<128x32xf16, #blocked>
    %cv = ttg.convert_layout %a : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %add = arith.addf %cv, %cst : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %r = tt.dot %add, %dotb, %dotc, inputPrecision = tf32 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2, 4], threadsPerWarp = [1, 1, 16, 2, 1], warpsPerCTA = [2, 1, 2, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 2, 1, 1, 4], threadsPerWarp = [1, 2, 16, 1, 1], warpsPerCTA = [2, 1, 2, 1, 1], order = [4, 1, 2, 3, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 32, 1, 1], warpsPerCTA = [1, 1, 1, 1, 4], order = [4, 3, 2, 1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @inject_tmem_copy
  // CHECK:   ttng.tmem_copy

  tt.func public @inject_tmem_copy(%scale: tensor<2x512x!tt.ptr<i8>, #blocked4> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) attributes {noinline = false} {
      %75 = ttg.local_alloc  : () -> !ttg.memdesc<2x512xi8, #shared1, #smem, mutable>
      %180 = ttg.local_load %75 : !ttg.memdesc<2x512xi8, #shared1, #smem, mutable, 3x2x512> -> tensor<2x512xi8, #blocked4>
      %183 = tt.reshape %180 : tensor<2x512xi8, #blocked4> -> tensor<2x1x32x4x4xi8, #blocked8>
      %184 = tt.trans %183 {order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8, #blocked8> -> tensor<2x4x32x1x4xi8, #blocked9>
      %187 = ttg.convert_layout %184 : tensor<2x4x32x1x4xi8, #blocked9> -> tensor<2x4x32x1x4xi8, #blocked10>
      %188 = tt.reshape %187 : tensor<2x4x32x1x4xi8, #blocked10> -> tensor<256x4xi8, #blocked11>
      %190 = ttng.tmem_alloc %188 : (tensor<256x4xi8, #blocked11>) -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      tt.return
}

}
