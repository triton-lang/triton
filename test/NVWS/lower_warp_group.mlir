// RUN: triton-opt --split-input-file --nvws-lower-warp-group %s | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

  // CHECK-LABEL: @warp_group
  //       CHECK-NOT: nvws.warp_group
  //       CHECK:   ttg.warp_specialize
  //       CHECK-NEXT:   default
  //       CHECK:   partition0
  //       CHECK-NEXT:   arith.constant
  //       CHECK-NEXT:   ttng.tc_gen5_mma
  tt.func @warp_group(%a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory, mutable>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    %false = arith.constant false
    nvws.warp_group
    partition0  num_warps(8) {
      ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier[%false] {is_async} :
        !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
         !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
         !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory, mutable>,
         !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
        nvws.warp_group.return
      }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

  // CHECK-LABEL: @warp_default
  //       CHECK-NOT: nvws.warp_group
  //       CHECK:   ttg.warp_specialize
  //       CHECK-NEXT:   default
  //       CHECK-NEXT:   ttng.tc_gen5_mma
  tt.func @warp_default(%a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory, mutable>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    %false = arith.constant false
    nvws.warp_group
    partition0  num_warps(4) {
      ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier[%false] {is_async} :
         !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
         !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
         !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory, mutable>,
         !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
        nvws.warp_group.return
      }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

  // CHECK-LABEL: @warp_multiple_group
  //       CHECK-NOT: nvws.warp_group
  //       CHECK:   ttg.warp_specialize(%
  //       CHECK-NEXT:   default
  //       CHECK-NEXT:   ttng.tc_gen5_mma
  //       CHECK:   partition0(%
  //       CHECK-NEXT:   arith.constant
  //       CHECK-NEXT:   ttg.local_load
  //       CHECK-NEXT:   ttng.wait_barrier
  //       CHECK-NEXT:   ttng.tmem_load
  //       CHECK-NEXT:   tt.store
  //       CHECK-NEXT:   ttg.warp_return
  //       CHECK-NEXT:   }
  tt.func @warp_multiple_group(%a: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xf16, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf16, #acc_tmem, #ttng.tensor_memory, mutable>,
                  %d: tensor<128x256x!tt.ptr<f16>, #blocked>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    %false = arith.constant false
    %c0 = arith.constant 0 : i32
    nvws.warp_group
    partition0  num_warps(4) {
      ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier[%false] {is_async} :
         !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
         !ttg.memdesc<128x256xf16, #shared1, #ttg.shared_memory>,
         !ttg.memdesc<128x256xf16, #acc_tmem, #ttng.tensor_memory, mutable>,
         !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
        nvws.warp_group.return
      }
    partition1 num_warps(4) {
      ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
      %c_reg = ttng.tmem_load %c : !ttg.memdesc<128x256xf16, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf16, #blocked>
      tt.store %d, %c_reg : tensor<128x256x!tt.ptr<f16>, #blocked>
      nvws.warp_group.return
    }
    tt.return
  }
}
