// RUN: triton-opt --split-input-file %s | FileCheck %s

#shared = #ttg.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared2 = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0], hasLeadingOffset = false}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

  // CHECK-LABEL: @tcgen5
  //       CHECK:   ttng.tc_gen5_mma
  //       CHECK:   ttng.tc_gen5_mma
  tt.func @tcgen5(%a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttng.tensor_memory, mutable>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier:
      (!ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttng.tensor_memory, mutable>,
       i1, i1,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) -> ()

    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred:
      (!ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttng.tensor_memory, mutable>,
       i1, i1) -> ()
    tt.return
  }

  // CHECK-LABEL: @async_tma_gather
  // CHECK-SAME: [[DESC:%arg[0-9]+]]:
  // CHECK-SAME: [[X_OFFSETS:%arg[0-9]+]]:
  // CHECK-SAME: [[Y_OFFSET:%arg[0-9]+]]:
  // CHECK-SAME: [[BAR:%arg[0-9]+]]:
  // CHECK-SAME: [[RESULT:%arg[0-9]+]]:
  // CHECK-SAME: [[PRED:%arg[0-9]+]]:
  tt.func @async_tma_gather(%desc: !tt.ptr<i8>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                            %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                            %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                            %pred: i1) {
    // CHECK-NEXT: ttng.async_tma_gather [[DESC]][[[X_OFFSETS]], [[Y_OFFSET]]] [[RESULT]], [[BAR]], [[PRED]] : !tt.ptr<i8>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared2, #smem, mutable>, !ttg.memdesc<32x128xbf16, #shared, #smem, mutable>, i1
    ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.ptr<i8>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
    tt.return
  }

  // CHECK-LABEL: @async_tma_scatter
  // CHECK-SAME: [[DESC:%arg[0-9]+]]:
  // CHECK-SAME: [[X_OFFSETS:%arg[0-9]+]]:
  // CHECK-SAME: [[Y_OFFSET:%arg[0-9]+]]:
  // CHECK-SAME: [[SRC:%arg[0-9]+]]:
  tt.func @async_tma_scatter(%desc: !tt.ptr<i8>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                             %src: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>) {
    // CHECK-NEXT: ttng.async_tma_scatter [[DESC]][[[X_OFFSETS]], [[Y_OFFSET]]] [[SRC]] : !tt.ptr<i8>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<32x128xbf16, #shared, #smem, mutable>
    ttng.async_tma_scatter %desc[%x_offsets, %y_offset] %src : !tt.ptr<i8>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }

}
