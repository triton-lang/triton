// RUN: triton-opt %s --split-input-file --triton-nvidia-tma-lowering | FileCheck %s

// Checks that the TMA lowering pass correctly handles a memdesc in the
// iter_args whose type has to be changed to mutable.
// CHECK-LABEL: @tma_memdesc_mutable_mismatch
#blocked_tma = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared_tma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_tma_trans = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem_tma = #ttg.shared_memory
#mma_tma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_memdesc_mutable_mismatch(%desc_q: !tt.tensordesc<tensor<128x128xf16, #shared_tma>>, 
                                                 %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared_tma>>, 
                                                 %N: i32) -> tensor<128x128xf32, #mma_tma> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_tma>
    
    %q = tt.descriptor_load %desc_q[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared_tma>> -> tensor<128x128xf16, #blocked_tma>
    %q_smem = ttg.local_alloc %q : (tensor<128x128xf16, #blocked_tma>) -> !ttg.memdesc<128x128xf16, #shared_tma, #smem_tma>
    
    %v_initial = tt.descriptor_load %desc_v[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared_tma>> -> tensor<128x128xf16, #blocked_tma>
    %v_smem_initial = ttg.local_alloc %v_initial : (tensor<128x128xf16, #blocked_tma>) -> !ttg.memdesc<128x128xf16, #shared_tma, #smem_tma>
    
    // CHECK: scf.for
    // CHECK-SAME: iter_args
    // CHECK-SAME: -> (tensor<128x128xf32, #mma>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>)
    %result:2 = scf.for %iv = %c0_i32 to %N step %c128_i32 iter_args(%acc = %cst, %v_smem = %v_smem_initial) -> (tensor<128x128xf32, #mma_tma>, !ttg.memdesc<128x128xf16, #shared_tma, #smem_tma>) : i32 {
      %v_new = tt.descriptor_load %desc_v[%iv, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared_tma>> -> tensor<128x128xf16, #blocked_tma>
      %v_smem_new = ttg.local_alloc %v_new : (tensor<128x128xf16, #blocked_tma>) -> !ttg.memdesc<128x128xf16, #shared_tma, #smem_tma>
      
      %acc_new = ttng.warp_group_dot %q_smem, %v_smem, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<128x128xf16, #shared_tma, #smem_tma> * !ttg.memdesc<128x128xf16, #shared_tma, #smem_tma> -> tensor<128x128xf32, #mma_tma>
      
      scf.yield %acc_new, %v_smem_new : tensor<128x128xf32, #mma_tma>, !ttg.memdesc<128x128xf16, #shared_tma, #smem_tma>
    }
    
    tt.return %result#0 : tensor<128x128xf32, #mma_tma>
  }
}
