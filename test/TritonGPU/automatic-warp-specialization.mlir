// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-automatic-warp-specialization                     | FileCheck %s --check-prefix=CHECK --check-prefix=BASE
// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-automatic-warp-specialization -tritongpu-pipeline | FileCheck %s --check-prefix=CHECK --check-prefix=PIPELINE

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @matmul_change_desc_in_prologue
tt.func @matmul_change_desc_in_prologue(
  %a_base: !tt.ptr<f16>,
  %b_base: !tt.ptr<f16>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %false = arith.constant false
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32
  %a_desc_undef = ub.poison : !tt.tensordesc<tensor<128x64xf16, #shared>>
  %b_desc_undef = ub.poison : !tt.tensordesc<tensor<64x128xf16, #shared>>
  // CHECK-LABEL: ttg.warp_specialize
  // CHECK-LABEL: default
  // BASE-NOT: tt.make_tensor_descriptor
  // PIPELINE-NOT: tt.experimental_tensormap_create
  // CHECK-LABEL: partition0
  // BASE-NOT: tt.make_tensor_descriptor
  // PIPELINE-NOT: tt.experimental_tensormap_create
  // CHECK-LABEL: partition1
  // BASE-COUNT-2: tt.make_tensor_descriptor
  // PIPELINE-COUNT-2: ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 512 : i32}
  // PIPELINE-COUNT-2: tt.experimental_tensormap_create
  // CHECK-LABEL: partition2
  // BASE-NOT: tt.make_tensor_descriptor
  // PIPELINE-NOT: tt.experimental_tensormap_create
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero, %flag = %true, %a_desc = %a_desc_undef, %b_desc = %b_desc_undef) -> (tensor<128x128xf32, #acc_layout>, i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>) : i32 {
    %do_prologue = "prologue_cond"(%k) : (i32) -> i1
    %cur_a_desc, %cur_b_desc = scf.if %do_prologue -> (!tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>) {
      %c1_i64 = arith.constant 1 : i64
      %next_a_desc = tt.make_tensor_descriptor %a_base, [%k, %k], [%c1_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
      %next_b_desc = tt.make_tensor_descriptor %b_base, [%k, %k], [%c1_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<64x128xf16, #shared>>
      scf.yield %next_a_desc, %next_b_desc : !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
    } else {
      scf.yield %a_desc, %b_desc : !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
    }

    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %flag, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %do_epilogue = "epilogue_cond"(%k) : (i32) -> i1
    %use_acc = arith.select %do_epilogue, %false, %true : i1
    scf.if %do_epilogue {
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    }
    scf.yield %c, %use_acc, %cur_a_desc, %cur_b_desc : tensor<128x128xf32, #acc_layout>, i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 4 : i32}

  tt.return
}

}
