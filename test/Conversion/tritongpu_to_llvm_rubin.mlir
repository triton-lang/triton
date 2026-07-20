// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=107 -cse | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107"} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_fp8_sm107
  // CHECK: %[[TMEM_BASE:.+]] = llvm.ptrtoint %arg2 : !llvm.ptr<3> to i32
  // CHECK-COUNT-2: tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32
  // CHECK-NOT: tcgen05.mma
  tt.func @tc_gen5_mma_block_scale_fp8_sm107(%a: !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<64x128xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e4m3 rhs = e2m1, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>,
    !ttg.memdesc<64x128xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales_a = #ttng.tensor_memory_scales_encoding<blockRepOrder = kThenMn>
#tmem_scales_b = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107"} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_nvfp4_sm107_m256
  // CHECK: %[[DESC:.+]] = llvm.mlir.constant(136316040 : i32) : i32
  // CHECK-COUNT-4: tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32
  // CHECK-NOT: tcgen05.mma
  tt.func @tc_gen5_mma_block_scale_nvfp4_sm107_m256(%a: !ttg.memdesc<256x128xi8, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<256x8xf8E4M3FN, #tmem_scales_a, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales_b, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<256x128xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<128x128xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<256x8xf8E4M3FN, #tmem_scales_a, #ttng.tensor_memory>,
    !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales_b, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107"} {
  // CHECK-LABEL: @tc_gen5_mma_fp8_sm107
  // CHECK: %[[TMEM_BASE:.+]] = llvm.ptrtoint %arg2 : !llvm.ptr<3> to i32
  // FP8 x FP8 (non-scaled) with K=128 on sm107, generates 2 MMA instructions
  // CHECK-COUNT-2: tcgen05.mma.cta_group::1.kind::f8f6f4
  // CHECK-NOT: tcgen05.mma
  tt.func @tc_gen5_mma_fp8_sm107(%a: !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xf8E4M3FN, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf8E4M3FN, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

// MMA codegen recognizes a fp4-padded SharedLinear A operand and selects
// the padded mxf8f6f4 MMA_K = 32 variant
#shared_fp4_padded_sl = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 0], [0, 8], [0, 16], [0, 32], [1, 8], [2, 16], [4, 32], [8, 0], [16, 0], [32, 0], [64, 0]], block = [[128, 0]]}, alignment = 1024>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<CGALayout = [[0, 0]]>
#tmem_scales1 = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  // CHECK-LABEL: @tc_gen5_mma_scaled_fp4_padded_shared_linear_a
  // CHECK-COUNT-4: tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32
  // CHECK-NOT: tcgen05.mma
  tt.func @tc_gen5_mma_scaled_fp4_padded_shared_linear_a(
      %a: !ttg.memdesc<256x64xi8, #shared_fp4_padded_sl, #smem, mutable>,
      %b: !ttg.memdesc<128x32xf8E4M3FN, #shared_b, #smem, mutable>,
      %d: !ttg.memdesc<256x32xf32, #tmem, #ttng.tensor_memory, mutable>,
      %sa: !ttg.memdesc<256x4xi8, #tmem_scales1, #ttng.tensor_memory, mutable>,
      %sb: !ttg.memdesc<32x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %useAcc, %pred lhs = e2m1 rhs = e4m3 {two_ctas} :
      !ttg.memdesc<256x64xi8, #shared_fp4_padded_sl, #smem, mutable>,
      !ttg.memdesc<128x32xf8E4M3FN, #shared_b, #smem, mutable>,
      !ttg.memdesc<256x32xf32, #tmem, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<256x4xi8, #tmem_scales1, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<32x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:107"} {
  // CHECK-LABEL: @tc_gen5_mma_breuse
  // CHECK: tcgen05.mma.cta_group::1.kind::f16.collector::b::fill
  // CHECK: tcgen05.mma.cta_group::1.kind::f16.collector::b::lastuse
  tt.func @tc_gen5_mma_breuse(%a: !ttg.memdesc<256x128xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<256x128xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107"} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_breuse
  // CHECK: tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::b::fill
  // CHECK: tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::b::lastuse
  tt.func @tc_gen5_mma_block_scale_breuse(%a: !ttg.memdesc<256x64xf8E4M3FN, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<32x128xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<256x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e4m3 rhs = e2m1, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<256x64xf8E4M3FN, #shared, #ttg.shared_memory>,
    !ttg.memdesc<32x128xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<256x2xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}


// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:107"} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_nvfp4_ue5m3_sm107
  // CHECK: %[[TMEM_BASE:.+]] = llvm.ptrtoint %arg2 : !llvm.ptr<3> to i32
  // UE5M3 uses i8 block16 scales, so split-K lowering must keep the UE5M3
  // scale kind in the descriptor for both K=128 sub-instructions of a
  // logical BLOCK_K=256 tile.
  // CHECK: %[[DESC:.+]] = llvm.mlir.constant(153093256 : i32) : i32
  // CHECK-COUNT-2: tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16
  // CHECK-NOT: tcgen05.mma
  tt.func @tc_gen5_mma_block_scale_nvfp4_ue5m3_sm107(%a: !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x16xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<128x16xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<128x128xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x16xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<128x16xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}
