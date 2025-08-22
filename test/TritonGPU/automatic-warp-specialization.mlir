// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-hoist-tmem-alloc -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-automatic-warp-specialization | FileCheck %s --check-prefix=CHECK --check-prefix=BASE
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-hoist-tmem-alloc -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-automatic-warp-specialization -tritongpu-pipeline | FileCheck %s --check-prefix=CHECK --check-prefix=PIPELINE

#indices_layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#b_layout = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
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
  // PIPELINE-NOT: ttng.tensormap_create
  // CHECK-LABEL: partition0
  // CHECK-SAME: num_warps(1)
  // BASE-NOT: tt.make_tensor_descriptor
  // PIPELINE-NOT: ttng.tensormap_create
  // PIPELINE-COUNT-1: tc_gen5_mma
  // PIPELINE-NOT: tc_gen5_mma
  // CHECK-LABEL: partition1
  // CHECK-SAME: num_warps(2)
  // BASE-COUNT-2: tt.make_tensor_descriptor
  // PIPELINE-COUNT-2: ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 512 : i32}
  // PIPELINE-COUNT-2: ttng.tensormap_create
  // PIPELINE-NOT: ttng.tensormap_create
  // PIPELINE-COUNT-2: async_tma_copy_global_to_local
  // PIPELINE-NOT: async_tma_copy_global_to_local
  // CHECK-NOT: partition2
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
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    %use_acc = arith.select %do_epilogue, %false, %true : i1
    scf.if %do_epilogue {
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    }
    scf.yield %c, %use_acc, %cur_a_desc, %cur_b_desc : tensor<128x128xf32, #acc_layout>, i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 4 : i32}

  tt.return
}

// CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use
tt.func @matmul_tma_acc_with_conditional_def_and_use(
  %a_desc: !tt.tensordesc<tensor<1x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %false = arith.constant false
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32
  // CHECK-LABEL: ttg.warp_specialize
  // CHECK-LABEL: default
  // CHECK-LABEL: partition0
  // CHECK-SAME: num_warps(1)
  // CHECK-LABEL: partition1
  // CHECK-SAME: num_warps(2)
  // CHECK: [[INDICES:%.*]] = tt.splat %{{.*}} : i32 -> tensor<128xi32,
  // CHECK: ttng.async_tma_gather %{{.*}}[[[INDICES]],
  // CHECK-NOT: partition2
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero, %flag = %true) -> (tensor<128x128xf32, #acc_layout>, i1) : i32 {
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)
    %indices = tt.splat %off_m : i32 -> tensor<128xi32, #indices_layout>
    %a = tt.descriptor_gather %a_desc[%indices, %off_k] : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #indices_layout>, i32) -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    %use_acc = arith.select %do_epilogue, %false, %true : i1
    scf.if %do_epilogue {
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    }
    scf.yield %c, %use_acc : tensor<128x128xf32, #acc_layout>, i1
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}
  tt.return
}

// CHECK-LABEL: @matmul_tma_and_regular_load
tt.func @matmul_tma_and_regular_load(
  %a_desc: !tt.tensordesc<tensor<1x64xf16, #shared>>,
  %b_ptr_init: tensor<64x128x!tt.ptr<f16>, #b_layout> {tt.divisibility = 16 : i32, tt.contiguity = 64 : i32}
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %false = arith.constant false
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32
  // CHECK-LABEL: ttg.warp_specialize
  // CHECK-LABEL: default
  // CHECK-LABEL: partition0
  // CHECK-SAME: num_warps(4)
  // PIPELINE-COUNT-3: async_copy_global_to_local
  // PIPELINE-NOT: async_copy_global_to_local
  // CHECK-LABEL: partition1
  // CHECK-SAME: num_warps(4)
  // CHECK: [[INDICES:%.*]] = tt.splat %{{.*}} : i32 -> tensor<128xi32,
  // CHECK: ttng.async_tma_gather %{{.*}}[[[INDICES]],
  // CHECK-NOT: partition2
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero, %flag = %true, %b_ptr = %b_ptr_init) -> (tensor<128x128xf32, #acc_layout>, i1, tensor<64x128x!tt.ptr<f16>, #b_layout>) : i32 {
    %off_m, %offs_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, tensor<64x128xi32, #b_layout>, i32)
    %indices = tt.splat %off_m : i32 -> tensor<128xi32, #indices_layout>

    %a = tt.descriptor_gather %a_desc[%indices, %off_k] : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #indices_layout>, i32) -> tensor<128x64xf16, #oper_layout>

    %b_ptrs = tt.addptr %b_ptr, %offs_n {tt.divisibility = dense<16> : tensor<64x128xi32>, tt.contiguity = dense<64> : tensor<64x128xi32>, tt.constancy = dense<1> : tensor<64x128xi32>} : tensor<64x128x!tt.ptr<f16>, #b_layout>, tensor<64x128xi32, #b_layout>
    %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f16>, #b_layout>

    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #b_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    %use_acc = arith.select %do_epilogue, %false, %true : i1
    scf.if %do_epilogue {
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    }
    scf.yield %c, %use_acc, %b_ptrs : tensor<128x128xf32, #acc_layout>, i1, tensor<64x128x!tt.ptr<f16>, #b_layout>
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>

#smem = #ttg.shared_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem_lhs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @attention_forward
tt.func public @attention_forward(
  %Q_shared: !ttg.memdesc<256x64xf16, #shared, #smem>,
  %K_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %V_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %qk_scale: f32,
  %n_tiles: i32,
  %idx_ptr: !tt.ptr<f32>
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32

  %neg_inf = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %zero = arith.constant dense<0.0> : tensor<256x64xf32, #blocked>
  %one = arith.constant dense<1.0> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

  // CHECK-LABEL: ttg.warp_specialize
  // PIPELINE: partition0
  // PIPELINE-COUNT-4: ttng.tc_gen5_mma
  // PIPELINE-NOT: ttng.tc_gen5_mma
  // PIPELINE: partition1
  // PIPELINE-COUNT-4: ttng.async_tma_copy_global_to_local
  // PIPELINE-NOT: ttng.async_tma_copy_global_to_local
  %loop_outs:3 = scf.for %i = %c0_i32 to %n_tiles step %c64_i32 iter_args(
    %l_i = %one,
    %acc = %zero,
    %m_i = %neg_inf
  ) -> (
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
    tensor<256x64xf32, #blocked>,
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  ) : i32 {

    %K = tt.descriptor_load %K_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %K_shared = ttg.local_alloc %K : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

    %K_trans = ttg.memdesc_trans %K_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>
    %QK_tmem, %QK_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %QK_mma_tok = ttng.tc_gen5_mma %Q_shared, %K_trans, %QK_tmem[%QK_tok], %false, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>

    %QK, %QK_load_tok = ttng.tmem_load %QK_tmem[%QK_mma_tok] : !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    %row_max = "compute_row_max"(%QK, %qk_scale) : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %QK_adj = "sub_row_max"(%QK, %row_max, %qk_scale) : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
    %softmax = math.exp2 %QK_adj : tensor<256x64xf32, #blocked>

    %diff = arith.subf %m_i, %row_max : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %alpha = math.exp2 %diff : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %l_ij = "tt.reduce"(%softmax) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %68 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %68 : f32
    }) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %l_i_scaled = arith.mulf %l_i, %alpha : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %next_l_i = arith.addf %l_i_scaled, %l_ij : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %alpha_0 = tt.expand_dims %alpha {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
    %alpha_1 = tt.broadcast %alpha_0 : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>

    %cur_idx_ptr = tt.addptr %idx_ptr, %i : !tt.ptr<f32>, i32
    %idx = tt.load %cur_idx_ptr : !tt.ptr<f32>
    %bias = tt.splat %idx : f32 -> tensor<256x64xf32, #blocked>

    %acc_step = arith.mulf %acc, %alpha_1 : tensor<256x64xf32, #blocked>
    %acc_corrected = arith.addf %acc_step, %bias : tensor<256x64xf32, #blocked>

    %62 = tt.descriptor_load %V_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %63 = ttg.local_alloc %62 : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

    %P = arith.truncf %softmax : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>

    %P_tmem = ttng.tmem_alloc %P : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem_lhs, #ttng.tensor_memory>
    %acc_tmem, %acc_tok = ttng.tmem_alloc %acc_corrected : (tensor<256x64xf32, #blocked>) -> (!ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %PV_mma_tok = ttng.tc_gen5_mma %P_tmem, %63, %acc_tmem[%acc_tok], %true, %true : !ttg.memdesc<256x64xf16, #tmem_lhs, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    %O, %O_tok = ttng.tmem_load %acc_tmem[%PV_mma_tok] : !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

    scf.yield %next_l_i, %O, %row_max : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  } {tt.warp_specialize}

  "use"(%loop_outs#0, %loop_outs#1, %loop_outs#2) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()

  tt.return
}

}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[8, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @matmul_kernel_tma_two_warps
  tt.func public @matmul_kernel_tma_two_warps(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64 , %arg4: i64, %arg5: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg15, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg16, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.addi %arg17, %c63_i32 : i32
    %15 = arith.divsi %14, %c64_i32 : i32
    %16 = arith.muli %11, %c128_i32 : i32
    %17 = arith.muli %13, %c128_i32 : i32
    // CHECK-NOT: partition0
    %18 = scf.for %arg18 = %c0_i32 to %15 step %c1_i32 iter_args(%arg19 = %cst) -> (tensor<128x128xf32, #mma>)  : i32 {
      %21 = arith.muli %arg18, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %22 = tt.descriptor_load %arg0[%16, %21] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
      %23 = tt.descriptor_load %arg5[%17, %21] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
      %24 = ttg.convert_layout %23 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #linear>
      %25 = tt.trans %24 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : tensor<128x64xf16, #linear> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %26 = ttg.convert_layout %22 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %26, %25, %arg19, inputPrecision = tf32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma> loc(#loc23)
      scf.yield %27 : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    %19 = arith.truncf %18 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %20 = ttg.convert_layout %19 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked>
    tt.descriptor_store %arg10[%16, %17], %20 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>
    tt.return
  }
}
