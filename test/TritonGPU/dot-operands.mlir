// RUN: triton-opt %s -split-input-file -tritongpu-optimize-dot-operands -canonicalize | FileCheck %s


#blockedA = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>

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
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// CHECK: #[[$SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: mma_reorder_transpose_mmav5
  tt.func @mma_reorder_transpose_mmav5(%t: tensor<64x256xf8E4M3FN, #blocked1>, %dotb: !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem>, %dotc: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory>) {
    %true = arith.constant true
    %a = tt.trans %t {order = array<i32: 1, 0>} : tensor<64x256xf8E4M3FN, #blocked1> -> tensor<256x64xf8E4M3FN, #blocked>
    // CHECK: %[[A:.+]] = ttg.local_alloc {{.*}} -> !ttg.memdesc<64x256xf8E4M3FN, #[[$SHARED]], #smem>
    // CHECK: %[[T:.+]] = ttg.memdesc_trans %[[A]] {order = array<i32: 1, 0>}
    // CHECK: ttng.tc_gen5_mma %[[T]]
    %dota = ttg.local_alloc %a: (tensor<256x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x64xf8E4M3FN, #shared1, #smem>
    ttng.tc_gen5_mma %dota, %dotb, %dotc, %true, %true : !ttg.memdesc<256x64xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: mmav2_reorder_transpose
// CHECK: ttg.local_alloc
// CHECK: ttg.memdesc_trans
// CHECK: %[[T0:.+]] = ttg.local_load
// CHECK: %[[T1:.*]] = tt.trans
// CHECK: tt.dot %[[T0]]
// CHECK: arith.extf %[[T1]]
  tt.func @mmav2_reorder_transpose(%t: tensor<32x128xf16, #blocked1>, %dotb: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %dotc: tensor<128x64xf32, #mma>) -> (tensor<128x64xf32, #mma>, tensor<128x32xf32, #blocked>){
    %a = tt.trans %t {order = array<i32: 1, 0>} : tensor<32x128xf16, #blocked1> -> tensor<128x32xf16, #blocked>
    %cv = ttg.convert_layout %a : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %r = tt.dot %cv, %dotb, %dotc, inputPrecision = tf32 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
    %trans_use = arith.extf %a : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    tt.return %r, %trans_use : tensor<128x64xf32, #mma>, tensor<128x32xf32, #blocked>
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

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2, 4], threadsPerWarp = [1, 1, 16, 2, 1], warpsPerCTA = [2, 1, 2, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 2, 1, 1, 4], threadsPerWarp = [1, 2, 16, 1, 1], warpsPerCTA = [2, 1, 2, 1, 1], order = [4, 1, 2, 3, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 32, 1, 1], warpsPerCTA = [1, 1, 1, 1, 4], order = [4, 3, 2, 1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @scales_in_shmem
  // CHECK: %[[A_LA:.*]] = ttg.local_alloc
  // CHECK: %[[B_LA:.*]] = ttg.local_alloc
  // CHECK: %[[A_RS:.*]] = ttg.memdesc_reshape %[[A_LA]]
  // CHECK: %[[A_TR:.*]] = ttg.memdesc_trans %[[A_RS]]
  // CHECK: %[[A_FINAL:.*]] = ttg.memdesc_reshape %[[A_TR]]
  // CHECK: %[[B_RS:.*]] = ttg.memdesc_reshape %[[B_LA]]
  // CHECK: %[[B_TR:.*]] = ttg.memdesc_trans %[[B_RS]]
  // CHECK: %[[B_FINAL:.*]] = ttg.memdesc_reshape %[[B_TR]]
  // CHECK-NOT: ttg.local_load
  // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, %[[A_FINAL]], %[[B_FINAL]],

  tt.func public @scales_in_shmem(
    %scale: tensor<2x512x!tt.ptr<i8>, #blocked4> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32},
    %A_sh: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
    %B_sh: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
    %acc_tm: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    ) {
      %true = arith.constant true
      %A_la = ttg.local_alloc : () -> !ttg.memdesc<2x512xi8, #shared1, #smem, mutable>
      %B_la = ttg.local_alloc : () -> !ttg.memdesc<2x512xi8, #shared1, #smem, mutable>
      %A_ll = ttg.local_load %A_la : !ttg.memdesc<2x512xi8, #shared1, #smem, mutable> -> tensor<2x512xi8, #blocked4>
      %B_ll = ttg.local_load %B_la : !ttg.memdesc<2x512xi8, #shared1, #smem, mutable> -> tensor<2x512xi8, #blocked4>
      %A_r = tt.reshape %A_ll : tensor<2x512xi8, #blocked4> -> tensor<2x1x32x4x4xi8, #blocked8>
      %B_r = tt.reshape %B_ll : tensor<2x512xi8, #blocked4> -> tensor<2x1x32x4x4xi8, #blocked8>
      %A_tr = tt.trans %A_r {order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8, #blocked8> -> tensor<2x4x32x1x4xi8, #blocked9>
      %B_tr = tt.trans %B_r {order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8, #blocked8> -> tensor<2x4x32x1x4xi8, #blocked9>
      %A_cv = ttg.convert_layout %A_tr : tensor<2x4x32x1x4xi8, #blocked9> -> tensor<2x4x32x1x4xi8, #blocked10>
      %B_cv = ttg.convert_layout %B_tr : tensor<2x4x32x1x4xi8, #blocked9> -> tensor<2x4x32x1x4xi8, #blocked10>
      %A_r2 = tt.reshape %A_cv : tensor<2x4x32x1x4xi8, #blocked10> -> tensor<256x4xi8, #blocked11>
      %B_r2 = tt.reshape %B_cv : tensor<2x4x32x1x4xi8, #blocked10> -> tensor<256x4xi8, #blocked11>
      %A_tm = ttng.tmem_alloc %A_r2 : (tensor<256x4xi8, #blocked11>) -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      %B_tm = ttng.tmem_alloc %B_r2 : (tensor<256x4xi8, #blocked11>) -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %A_tm, %B_tm, %true, %true lhs = e5m2 rhs = e5m2 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      tt.return
}
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0], CGALayout = [[1, 0]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CGALayout = [[0, 1]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  // CHECK-DAG: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
  // CHECK-DAG: #[[SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = {{\[\[0, 1\]\]}}}>
  // CHECK-DAG: #[[SHARED_TRANS:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16, CGALayout = {{\[\[1, 0\]\]}}}>
  // CHECK: %[[ALLOC:.*]] = ttg.local_alloc %arg0 : (tensor<128x64xf8E4M3FN, #[[BLOCKED]]>) -> !ttg.memdesc<128x64xf8E4M3FN, #[[SHARED_TRANS]], #smem>
  // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[ALLOC]] {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf8E4M3FN, #[[SHARED_TRANS]], #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #[[SHARED]], #smem>
  // CHECK: ttng.tc_gen5_mma %arg1, %[[TRANS]]
  tt.func @mmav5_reorder_transpose_2cta(%b_trans: tensor<128x64xf8E4M3FN, #blocked1>, %dota: !ttg.memdesc<256x64xf8E4M3FN, #shared, #smem>, %dotc: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory>) {
    %true = arith.constant true
    %trans = tt.trans %b_trans {order = array<i32: 1, 0>} : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #blocked2>
    %dotb = ttg.local_alloc %trans : (tensor<64x128xf8E4M3FN, #blocked2>) -> !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem>
    ttng.tc_gen5_mma %dota, %dotb, %dotc, %true, %true {two_ctas}: !ttg.memdesc<256x64xf8E4M3FN, #shared, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 2, 16], warpsPerCTA = [1, 1, 1, 8, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, fp4Padded = true}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#smem = #ttg.shared_memory
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4], [0, 0, 0, 0, 8]], lane = [[0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 0, 4, 0], [0, 0, 0, 8, 0], [0, 0, 0, 16, 0]], warp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0]], lane = [[0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0]], warp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0]], lane = [[0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0]], warp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0], [128, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[BLOCKED5:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 2, 16], warpsPerCTA = [1, 1, 1, 8, 1], order = [4, 3, 2, 1, 0]}>
  // CHECK-DAG: #[[SHARED2:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
  // CHECK-DAG: #[[SMEM:.*]] = #ttg.shared_memory
  tt.func public @descriptor_load_scales_in_shmem(
      %scale_desc_ptr: !tt.ptr<i8>,
      %shmemA: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>,
      %shmemB: !ttg.memdesc<64x256xi8, #shared1, #smem>,
      %acc: tensor<128x256xf32, #blocked1>
    ) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst_scales = arith.constant dense<127> : tensor<128x4xi8, #linear>
    %true = arith.constant true

    %desc = tt.make_tensor_descriptor %scale_desc_ptr, [%c1_i32, %c2_i32, %c1_i32, %c32_i32, %c16_i32], [%c1024_i64, %c512_i64, %c512_i64, %c16_i64, %c1_i64] : !tt.ptr<i8>, <1x2x1x32x16xi8>
    // CHECK: %[[DESC_LOAD:.*]] = tt.descriptor_load {{.*}} !tt.tensordesc<1x2x1x32x16xi8> -> tensor<1x2x1x32x16xi8, #[[BLOCKED5]]>
    %83 = tt.descriptor_load %desc[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x2x1x32x16xi8> -> tensor<1x2x1x32x16xi8, #blocked5>
    // CHECK: %[[DESC_LA:.*]] = ttg.local_alloc %[[DESC_LOAD]] : (tensor<1x2x1x32x16xi8, #[[BLOCKED5]]>) -> !ttg.memdesc<1x2x1x32x16xi8, #[[SHARED2]], #[[SMEM]]>
    %84 = ttg.local_alloc %83 : (tensor<1x2x1x32x16xi8, #blocked5>) -> !ttg.memdesc<1x2x1x32x16xi8, #shared2, #smem>
    // CHECK-NOT: ttg.local_load
    %85 = ttg.local_load %84 : !ttg.memdesc<1x2x1x32x16xi8, #shared2, #smem> -> tensor<1x2x1x32x16xi8, #linear1>
    // CHECK-NOT: tt.reshape
    %86 = tt.reshape %85 : tensor<1x2x1x32x16xi8, #linear1> -> tensor<2x1x32x4x4xi8, #linear2>
    // CHECK-NOT: tt.trans
    %87 = tt.trans %86 {order = array<i32: 0, 3, 2, 1, 4>} : tensor<2x1x32x4x4xi8, #linear2> -> tensor<2x4x32x1x4xi8, #linear3>
    // CHECK-NOT: tt.reshape
    %88 = tt.reshape %87 : tensor<2x4x32x1x4xi8, #linear3> -> tensor<256x4xi8, #linear4>
    %89 = ttng.tmem_alloc %acc : (tensor<128x256xf32, #blocked1>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %90 = ttng.tmem_alloc %cst_scales : (tensor<128x4xi8, #linear>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    %91 = ttng.tmem_alloc %88 : (tensor<256x4xi8, #linear4>) -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
    // CHECK: %[[DESC_RS:.*]] = ttg.memdesc_reshape %[[DESC_LA]] : !ttg.memdesc<1x2x1x32x16xi8, #[[SHARED2]], #[[SMEM]]> -> !ttg.memdesc<2x1x32x4x4xi8, {{.*}}, #smem>
    // CHECK: %[[DESC_TR:.*]] = ttg.memdesc_trans %[[DESC_RS]]
    // CHECK: %[[SCALE_ALLOC:.*]] = ttg.memdesc_reshape %[[DESC_TR]] : !ttg.memdesc<2x4x32x1x4xi8, {{.*}}, #smem> -> !ttg.memdesc<256x4xi8, {{.*}}, #smem>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[SCALE_ALLOC]], {{.*}}
    ttng.tc_gen5_mma_scaled %shmemA, %shmemB, %89, %90, %91, %true, %true lhs = e4m3 rhs = e2m1 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<64x256xi8, #shared1, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32], warpsPerCTA = [1, 1, 4, 1], order = [3, 2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-DAG: #[[$SHARED:.+]] = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
// CHECK-DAG: #[[$SHARED1:.+]] = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, rank = 4}>
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reshape_memedesc
  tt.func @reshape_memedesc(%arg: tensor<32x1x4x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem> {
    // CHECK: [[A:.+]] = ttg.local_alloc %{{.+}} : (tensor<32x1x4x64xf16, #{{.*}}>) -> !ttg.memdesc<32x1x4x64xf16, #[[$SHARED1]], #smem>
    %r = tt.reshape %arg : tensor<32x1x4x64xf16, #blocked> -> tensor<128x64xf16, #blocked1>
    // CHECK: %[[R:.+]] = ttg.memdesc_reshape %[[A:.+]] : !ttg.memdesc<32x1x4x64xf16, #[[$SHARED1]], #smem> -> !ttg.memdesc<128x64xf16, #[[$SHARED]], #smem>
    %a = ttg.local_alloc %r : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // CHECK: tt.return %[[R]]
    tt.return %a: !ttg.memdesc<128x64xf16, #shared, #smem>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32], warpsPerCTA = [1, 2, 2, 1], order = [3, 2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reshape_memedesc_negative
  tt.func @reshape_memedesc_negative(%arg: tensor<1x32x2x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem> {
    %r = tt.reshape %arg : tensor<1x32x2x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    // CHECK-NOT: ttg.memdesc_reshape
    %a = ttg.local_alloc %r : (tensor<64x64xf32, #blocked1>) -> !ttg.memdesc<64x64xf32, #shared, #smem>
    // CHECK: tt.return
    tt.return %a: !ttg.memdesc<64x64xf32, #shared, #smem>
  }
}

// -----

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
  // CHECK-DAG: %[[A_DESC:.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN, #{{.*}}>
  // CHECK-DAG: %[[A_BASE:.*]] = ttg.local_alloc %[[A_DESC]] : (tensor<128x128xf8E4M3FN, #{{.*}}>) -> !ttg.memdesc<128x128xf8E4M3FN, #{{.*}}, #smem{{.*}}>
  // CHECK-DAG: %[[B_DESC:.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<1x1x256x8x16xf8E4M3FN> -> tensor<1x1x256x8x16xf8E4M3FN, #{{.*}}>
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
      %a_desc: !tt.tensordesc<128x128xf8E4M3FN>,
      %b_desc: !tt.tensordesc<1x1x256x8x16xf8E4M3FN>,
      %acc: !ttg.memdesc<128x256xf32, #tmem0, #ttng.tensor_memory>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN, #blocked0>
    %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<1x1x256x8x16xf8E4M3FN> -> tensor<1x1x256x8x16xf8E4M3FN, #blocked1>
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
  // CHECK-DAG: %[[B_DESC_LOAD:.*]] = tt.descriptor_load %arg1[%{{.*}}] : !tt.tensordesc<1x1x256x8x16xf8E4M3FN> -> tensor<1x1x256x8x16xf8E4M3FN, #{{.*}}>
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
      %a_desc: !tt.tensordesc<128x128xf8E4M3FN, #sharedA_desc>,
      %b_desc: !tt.tensordesc<1x1x256x8x16xf8E4M3FN>,
      %acc: tensor<128x256xf32, #mma_desc>) -> tensor<128x256xf32, #mma_desc> {
    %c0 = arith.constant 0 : i32
    %a = tt.descriptor_load %a_desc[%c0, %c0] : !tt.tensordesc<128x128xf8E4M3FN, #sharedA_desc> -> tensor<128x128xf8E4M3FN, #blockedA_desc>
    %a_s = ttg.local_alloc %a : (tensor<128x128xf8E4M3FN, #blockedA_desc>) -> !ttg.memdesc<128x128xf8E4M3FN, #sharedA_desc, #smem>
    %b_cm = tt.descriptor_load %b_desc[%c0, %c0, %c0, %c0, %c0] : !tt.tensordesc<1x1x256x8x16xf8E4M3FN> -> tensor<1x1x256x8x16xf8E4M3FN, #blockedB_desc>
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
