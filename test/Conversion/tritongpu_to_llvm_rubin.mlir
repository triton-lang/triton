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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith_f32x2
  // CHECK: llvm.insertelement {{.*}} : vector<2xf32>
  // CHECK: llvm.bitcast {{.*}} : vector<2xf32> to i64
  // CHECK: llvm.inline_asm {{.*}} "add.rn.f32x2 $0, $1, $2;", "=l,l,l" {{.*}} : (i64, i64) -> i64
  // CHECK: llvm.inline_asm {{.*}} "sub.rn.f32x2 $0, $1, $2;", "=l,l,l" {{.*}} : (i64, i64) -> i64
  // CHECK: llvm.inline_asm {{.*}} "mul.rn.f32x2 $0, $1, $2;", "=l,l,l" {{.*}} : (i64, i64) -> i64
  // CHECK: llvm.inline_asm {{.*}} "fma.rn.f32x2 $0, $1, $2, $3;", "=l,l,l,l" {{.*}} : (i64, i64, i64) -> i64
  // CHECK: llvm.bitcast {{.*}} : i64 to vector<2xf32>
  tt.func private @packed_arith_f32x2(
      %a: tensor<128x2xf32, #blocked>,
      %b: tensor<128x2xf32, #blocked>,
      %c: tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked> {
    %add = ttng.packed_arith add, f32x2, [f32x2, f32x2], %a, %b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    %sub = ttng.packed_arith sub, f32x2, [f32x2, f32x2], %add, %b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    %mul = ttng.packed_arith mul, f32x2, [f32x2, f32x2], %sub, %b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    %fma = ttng.packed_arith fma, f32x2, [f32x2, f32x2, f32x2], %mul, %b, %c axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    tt.return %fma : tensor<128x2xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith_homogeneous_half
  // CHECK: llvm.inline_asm {{.*}} "add.rn.f16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "sub.rn.f16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "mul.rn.f16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "fma.rn.f16x2 $0, $1, $2, $3;", "=r,r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "min.f16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "max.f16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "add.rn.bf16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "sub.rn.bf16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "mul.rn.bf16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "min.bf16x2 $0, $1, $2;", "=r,r,r"
  // CHECK: llvm.inline_asm {{.*}} "max.bf16x2 $0, $1, $2;", "=r,r,r"
  tt.func @packed_arith_homogeneous_half(
      %f16a: tensor<128x2xf16, #blocked>,
      %f16b: tensor<128x2xf16, #blocked>,
      %f16c: tensor<128x2xf16, #blocked>,
      %bf16a: tensor<128x2xbf16, #blocked>,
      %bf16b: tensor<128x2xbf16, #blocked>,
      %bf16c: tensor<128x2xbf16, #blocked>) {
    %f16add = ttng.packed_arith add, f16x2, [f16x2, f16x2], %f16a, %f16b axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf16, #blocked>) -> tensor<128x2xf16, #blocked>
    %f16sub = ttng.packed_arith sub, f16x2, [f16x2, f16x2], %f16add, %f16b axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf16, #blocked>) -> tensor<128x2xf16, #blocked>
    %f16mul = ttng.packed_arith mul, f16x2, [f16x2, f16x2], %f16sub, %f16b axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf16, #blocked>) -> tensor<128x2xf16, #blocked>
    %f16fma = ttng.packed_arith fma, f16x2, [f16x2, f16x2, f16x2], %f16mul, %f16b, %f16c axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf16, #blocked>, tensor<128x2xf16, #blocked>) -> tensor<128x2xf16, #blocked>
    %f16min = ttng.packed_arith min, f16x2, [f16x2, f16x2], %f16fma, %f16b axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf16, #blocked>) -> tensor<128x2xf16, #blocked>
    %f16max = ttng.packed_arith max, f16x2, [f16x2, f16x2], %f16min, %f16b axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf16, #blocked>) -> tensor<128x2xf16, #blocked>
    %bf16add = ttng.packed_arith add, bf16x2, [bf16x2, bf16x2], %bf16a, %bf16b axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>) -> tensor<128x2xbf16, #blocked>
    %bf16sub = ttng.packed_arith sub, bf16x2, [bf16x2, bf16x2], %bf16add, %bf16b axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>) -> tensor<128x2xbf16, #blocked>
    %bf16mul = ttng.packed_arith mul, bf16x2, [bf16x2, bf16x2], %bf16sub, %bf16b axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>) -> tensor<128x2xbf16, #blocked>
    %bf16fma = ttng.packed_arith fma, bf16x2, [bf16x2, bf16x2, bf16x2], %bf16mul, %bf16b, %bf16c axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>) -> tensor<128x2xbf16, #blocked>
    %bf16min = ttng.packed_arith min, bf16x2, [bf16x2, bf16x2], %bf16fma, %bf16b axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>) -> tensor<128x2xbf16, #blocked>
    %bf16max = ttng.packed_arith max, bf16x2, [bf16x2, bf16x2], %bf16min, %bf16b axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>) -> tensor<128x2xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith_alternate_x4
  // CHECK: llvm.inline_asm {{.*}} "add.rn.e4m3x4.e5m2x4 $0, $1, $2;", "=r,r,r" {{.*}} : (i32, i32) -> i32
  // CHECK: llvm.trunc {{.*}} : i8 to i4
  // CHECK: llvm.bitcast {{.*}} : vector<4xi4> to i16
  // CHECK: llvm.inline_asm {{.*}} "sub.rn.e5m2x4.e2m1x4 $0, $1, $2;", "=r,h,r" {{.*}} : (i16, i32) -> i32
  // CHECK: %[[MASK6:.*]] = llvm.mlir.constant(63 : i8) : i8
  // CHECK: llvm.and {{.*}}, %[[MASK6]] : i8
  // CHECK: llvm.inline_asm {{.*}} "mul.rn.e4m3x4.e3m2x4.e2m3x4 $0, $1, $2;", "=r,r,r" {{.*}} : (i32, i32) -> i32
  // CHECK: %[[MASK4:.*]] = llvm.mlir.constant(15 : i8) : i8
  // CHECK: llvm.and {{.*}}, %[[MASK4]] : i8
  // CHECK: llvm.inline_asm {{.*}} "fma.rn.e5m2x4.e2m1p4x4.ue8m0x4 $0, $1, $2, $3;", "=r,r,r,r" {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<4xi8>
  tt.func @packed_arith_alternate_x4(
      %e5m2: tensor<128x4xf8E5M2, #blocked>,
      %e4m3: tensor<128x4xf8E4M3FN, #blocked>,
      %e3m2: tensor<128x4xf6E3M2FN, #blocked>,
      %e2m3: tensor<128x4xf6E2M3FN, #blocked>,
      %e2m1: tensor<128x4xf4E2M1FN, #blocked>,
      %ue8m0: tensor<128x4xf8E8M0FNU, #blocked>) {
    %add = ttng.packed_arith add, e4m3x4, [e5m2x4, e4m3x4], %e5m2, %e4m3 axis = 1 : (tensor<128x4xf8E5M2, #blocked>, tensor<128x4xf8E4M3FN, #blocked>) -> tensor<128x4xf8E4M3FN, #blocked>
    %sub = ttng.packed_arith sub, e5m2x4, [e2m1x4, e5m2x4], %e2m1, %e5m2 axis = 1 : (tensor<128x4xf4E2M1FN, #blocked>, tensor<128x4xf8E5M2, #blocked>) -> tensor<128x4xf8E5M2, #blocked>
    %mul = ttng.packed_arith mul, e4m3x4, [e3m2x4, e2m3x4], %e3m2, %e2m3 axis = 1 : (tensor<128x4xf6E3M2FN, #blocked>, tensor<128x4xf6E2M3FN, #blocked>) -> tensor<128x4xf8E4M3FN, #blocked>
    %fma = ttng.packed_arith fma, e5m2x4, [e2m1p4x4, ue8m0x4, e5m2x4], %e2m1, %ue8m0, %e5m2 axis = 1 : (tensor<128x4xf4E2M1FN, #blocked>, tensor<128x4xf8E8M0FNU, #blocked>, tensor<128x4xf8E5M2, #blocked>) -> tensor<128x4xf8E5M2, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith_mixed
  // CHECK: llvm.inline_asm {{.*}} "mul.rn.f16x2 $0, $1, $2;", "=r,r,r" {{.*}} : (i32, i32) -> i32
  // CHECK: llvm.inline_asm {{.*}} "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r" {{.*}} : (i32, i32, i32) -> i32
  // CHECK: llvm.inline_asm {{.*}} "add.rn.f32x2.f16x2.f32x2 $0, $1, $2;", "=l,r,l" {{.*}} : (i32, i64) -> i64
  // CHECK: llvm.inline_asm {{.*}} "add.rn.f32x2.bf16x2.f32x2 $0, $1, $2;", "=l,r,l" {{.*}} : (i32, i64) -> i64
  // CHECK: llvm.inline_asm {{.*}} "sub.rn.f32x2.f16x2.f32x2 $0, $1, $2;", "=l,r,l" {{.*}} : (i32, i64) -> i64
  // CHECK: llvm.inline_asm {{.*}} "sub.rn.f32x2.bf16x2.f32x2 $0, $1, $2;", "=l,r,l" {{.*}} : (i32, i64) -> i64
  // CHECK: llvm.inline_asm {{.*}} "add.rz.ftz.f16x2.f32x2.f32x2 $0, $1, $2;", "=r,l,l" {{.*}} : (i64, i64) -> i32
  // CHECK: llvm.inline_asm {{.*}} "add.rz.bf16x2.f32x2.f32x2 $0, $1, $2;", "=r,l,l" {{.*}} : (i64, i64) -> i32
  // CHECK: llvm.inline_asm {{.*}} "sub.rz.ftz.f16x2.f32x2.f32x2 $0, $1, $2;", "=r,l,l" {{.*}} : (i64, i64) -> i32
  // CHECK: llvm.inline_asm {{.*}} "sub.rz.bf16x2.f32x2.f32x2 $0, $1, $2;", "=r,l,l" {{.*}} : (i64, i64) -> i32
  // CHECK: llvm.inline_asm {{.*}} "mul.ftz.rz.f16x2.f32x2.f32x2 $0, $1, $2;", "=r,l,l" {{.*}} : (i64, i64) -> i32
  // CHECK: llvm.inline_asm {{.*}} "mul.rz.bf16x2.f32x2.f32x2 $0, $1, $2;", "=r,l,l" {{.*}} : (i64, i64) -> i32
  // CHECK: llvm.inline_asm {{.*}} "mul.bf16x2.bf16x2.f16x2 $0, $1, $2;", "=r,r,r" {{.*}} : (i32, i32) -> i32
  // CHECK: llvm.inline_asm {{.*}} "mul.f16x2.f16x2.bf16x2 $0, $1, $2;", "=r,r,r" {{.*}} : (i32, i32) -> i32
  // CHECK: llvm.inline_asm {{.*}} "fma.rn.f32x2.f16x2.f32x2.f32x2 $0, $1, $2, $3;", "=l,r,l,l" {{.*}} : (i32, i64, i64) -> i64
  // CHECK: llvm.inline_asm {{.*}} "fma.rn.f32x2.bf16x2.f32x2.f32x2 $0, $1, $2, $3;", "=l,r,l,l" {{.*}} : (i32, i64, i64) -> i64
  tt.func @packed_arith_mixed(
      %f32a: tensor<128x2xf32, #blocked>,
      %f32b: tensor<128x2xf32, #blocked>,
      %f16: tensor<128x2xf16, #blocked>,
      %bf16: tensor<128x2xbf16, #blocked>) {
    %hom_f16 = ttng.packed_arith mul, f16x2, [f16x2, f16x2], %f16, %f16 axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf16, #blocked>) -> tensor<128x2xf16, #blocked>
    %hom_bf16 = ttng.packed_arith fma, bf16x2, [bf16x2, bf16x2, bf16x2], %bf16, %bf16, %bf16 axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>, tensor<128x2xbf16, #blocked>) -> tensor<128x2xbf16, #blocked>
    %add_up_f16 = ttng.packed_arith add, f32x2, [f16x2, f32x2], %f16, %f32a axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    %add_up_bf16 = ttng.packed_arith add, f32x2, [bf16x2, f32x2], %bf16, %f32a axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    %sub_up_f16 = ttng.packed_arith sub, f32x2, [f16x2, f32x2], %f16, %f32a axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    %sub_up_bf16 = ttng.packed_arith sub, f32x2, [bf16x2, f32x2], %bf16, %f32a axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    %down_f16 = ttng.packed_arith add, f16x2, [f32x2, f32x2], %f32a, %f32b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf16, #blocked>
    %add_down_bf16 = ttng.packed_arith add, bf16x2, [f32x2, f32x2], %f32a, %f32b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xbf16, #blocked>
    %sub_down_f16 = ttng.packed_arith sub, f16x2, [f32x2, f32x2], %f32a, %f32b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf16, #blocked>
    %down_bf16 = ttng.packed_arith sub, bf16x2, [f32x2, f32x2], %f32a, %f32b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xbf16, #blocked>
    %mul_down = ttng.packed_arith mul, f16x2, [f32x2, f32x2], %f32a, %f32b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf16, #blocked>
    %mul_down_bf16 = ttng.packed_arith mul, bf16x2, [f32x2, f32x2], %f32a, %f32b axis = 1 : (tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xbf16, #blocked>
    %mul_cross = ttng.packed_arith mul, bf16x2, [bf16x2, f16x2], %bf16, %f16 axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xf16, #blocked>) -> tensor<128x2xbf16, #blocked>
    %mul_cross_reverse = ttng.packed_arith mul, f16x2, [f16x2, bf16x2], %f16, %bf16 axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xbf16, #blocked>) -> tensor<128x2xf16, #blocked>
    %fma_f16 = ttng.packed_arith fma, f32x2, [f16x2, f32x2, f32x2], %f16, %f32a, %f32b axis = 1 : (tensor<128x2xf16, #blocked>, tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    %fma_bf16 = ttng.packed_arith fma, f32x2, [bf16x2, f32x2, f32x2], %bf16, %f32a, %f32b axis = 1 : (tensor<128x2xbf16, #blocked>, tensor<128x2xf32, #blocked>, tensor<128x2xf32, #blocked>) -> tensor<128x2xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith_x4_axis0
  // CHECK: %[[A0:.*]] = llvm.extractvalue %arg0[0]
  // CHECK: %[[A1:.*]] = llvm.extractvalue %arg0[1]
  // CHECK: %[[A2:.*]] = llvm.extractvalue %arg0[2]
  // CHECK: %[[A3:.*]] = llvm.extractvalue %arg0[3]
  // CHECK: %[[A4:.*]] = llvm.extractvalue %arg0[4]
  // CHECK: %[[A5:.*]] = llvm.extractvalue %arg0[5]
  // CHECK: %[[A6:.*]] = llvm.extractvalue %arg0[6]
  // CHECK: %[[A7:.*]] = llvm.extractvalue %arg0[7]
  // CHECK: llvm.insertelement %[[A0]],
  // CHECK: llvm.insertelement %[[A2]],
  // CHECK: llvm.insertelement %[[A4]],
  // CHECK: llvm.insertelement %[[A6]],
  // CHECK: llvm.inline_asm {{.*}} "add.rn.e4m3x4.e4m3x4
  // CHECK: llvm.insertelement %[[A1]],
  // CHECK: llvm.insertelement %[[A3]],
  // CHECK: llvm.insertelement %[[A5]],
  // CHECK: llvm.insertelement %[[A7]],
  // CHECK: llvm.inline_asm {{.*}} "add.rn.e4m3x4.e4m3x4
  tt.func private @packed_arith_x4_axis0(
      %a: tensor<4x256xf8E4M3FN, #blocked>,
      %b: tensor<4x256xf8E4M3FN, #blocked>) -> tensor<4x256xf8E4M3FN, #blocked> {
    %0 = ttng.packed_arith add, e4m3x4, [e4m3x4, e4m3x4], %a, %b axis = 0 : (tensor<4x256xf8E4M3FN, #blocked>, tensor<4x256xf8E4M3FN, #blocked>) -> tensor<4x256xf8E4M3FN, #blocked>
    tt.return %0 : tensor<4x256xf8E4M3FN, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @packed_arith_axis0
  // CHECK: %[[A0:.*]] = llvm.extractvalue %arg0[0]
  // CHECK: %[[A1:.*]] = llvm.extractvalue %arg0[1]
  // CHECK: %[[A2:.*]] = llvm.extractvalue %arg0[2]
  // CHECK: %[[A3:.*]] = llvm.extractvalue %arg0[3]
  // CHECK: llvm.insertelement %[[A0]],
  // CHECK: llvm.insertelement %[[A2]],
  // CHECK: llvm.inline_asm {{.*}} "add.rn.f32x2
  // CHECK: llvm.insertelement %[[A1]],
  // CHECK: llvm.insertelement %[[A3]],
  // CHECK: llvm.inline_asm {{.*}} "add.rn.f32x2
  tt.func private @packed_arith_axis0(
      %a: tensor<2x256xf32, #blocked>,
      %b: tensor<2x256xf32, #blocked>) -> tensor<2x256xf32, #blocked> {
    %0 = ttng.packed_arith add, f32x2, [f32x2, f32x2], %a, %b axis = 0 : (tensor<2x256xf32, #blocked>, tensor<2x256xf32, #blocked>) -> tensor<2x256xf32, #blocked>
    tt.return %0 : tensor<2x256xf32, #blocked>
  }
}
