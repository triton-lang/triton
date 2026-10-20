// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=107 -cse | FileCheck %s

#shared_a16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem_a16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_meta = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sparse_mma_f16_smem_a
  // CHECK: tcgen05.mma.sp.cta_group::1.kind::f16
  tt.func @sparse_mma_f16_smem_a(
      %a: !ttg.memdesc<128x64xf16, #shared_a16, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xf16, #shared_b16, #ttg.shared_memory>,
      %sp: !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1) {
    ttng.tc_gen5_mma_sp %a, %b, %sp, %d, %useAcc, %pred :
      !ttg.memdesc<128x64xf16, #shared_a16, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b16, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_b16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem_a16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_meta = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sparse_mma_f16_tmem_a
  // CHECK: tcgen05.mma.sp.cta_group::1.kind::f16
  tt.func @sparse_mma_f16_tmem_a(
      %a: !ttg.memdesc<128x64xf16, #tmem_a16, #ttng.tensor_memory>,
      %b: !ttg.memdesc<128x128xf16, #shared_b16, #ttg.shared_memory>,
      %sp: !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1) {
    ttng.tc_gen5_mma_sp %a, %b, %sp, %d, %useAcc, %pred :
      !ttg.memdesc<128x64xf16, #tmem_a16, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf16, #shared_b16, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_a16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_meta = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sparse_mma_bf16
  // CHECK: tcgen05.mma.sp.cta_group::1.kind::f16
  tt.func @sparse_mma_bf16(
      %a: !ttg.memdesc<128x64xbf16, #shared_a16, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xbf16, #shared_b16, #ttg.shared_memory>,
      %sp: !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1) {
    ttng.tc_gen5_mma_sp %a, %b, %sp, %d, %useAcc, %pred :
      !ttg.memdesc<128x64xbf16, #shared_a16, #ttg.shared_memory>,
      !ttg.memdesc<128x128xbf16, #shared_b16, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_a8 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_b8 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_meta = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sparse_mma_f8e5
  // CHECK: tcgen05.mma.sp.cta_group::1.kind::f8f6f4
  tt.func @sparse_mma_f8e5(
      %a: !ttg.memdesc<128x64xf8E5M2, #shared_a8, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xf8E5M2, #shared_b8, #ttg.shared_memory>,
      %sp: !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1) {
    ttng.tc_gen5_mma_sp %a, %b, %sp, %d, %useAcc, %pred :
      !ttg.memdesc<128x64xf8E5M2, #shared_a8, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf8E5M2, #shared_b8, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_a8 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_b8 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_meta = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sparse_mma_f8e4
  // CHECK: tcgen05.mma.sp.cta_group::1.kind::f8f6f4
  tt.func @sparse_mma_f8e4(
      %a: !ttg.memdesc<128x64xf8E4M3FN, #shared_a8, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xf8E4M3FN, #shared_b8, #ttg.shared_memory>,
      %sp: !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1) {
    ttng.tc_gen5_mma_sp %a, %b, %sp, %d, %useAcc, %pred :
      !ttg.memdesc<128x64xf8E4M3FN, #shared_a8, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf8E4M3FN, #shared_b8, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_a16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_meta = #ttng.tensor_memory_encoding<blockM = 128, blockN = 4, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sparse_mma_f16_async
  // CHECK: tcgen05.mma.sp.cta_group::1.kind::f16
  // CHECK: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64
  tt.func @sparse_mma_f16_async(
      %a: !ttg.memdesc<128x64xf16, #shared_a16, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xf16, #shared_b16, #ttg.shared_memory>,
      %sp: !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      %d: !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %useAcc: i1, %pred: i1,
      %barrier: !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>,
      %barrierPred: i1) {
    ttng.tc_gen5_mma_sp %a, %b, %sp, %d, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
      !ttg.memdesc<128x64xf16, #shared_a16, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b16, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi32, #tmem_meta, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<1xi64, #shared_bar, #ttg.shared_memory, mutable>
    tt.return
  }
}
