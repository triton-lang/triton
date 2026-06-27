// RUN: triton-opt %s --split-input-file --tritongpu-hoist-tmem-alloc --tritongpu-partition-scheduling -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared_f32 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>

#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @attention_forward
tt.func public @attention_forward(
  %Q_shared: !ttg.memdesc<256x64xf16, #shared, #smem>,
  %K_desc: !tt.tensordesc<64x64xf16, #shared>,
  %V_desc: !tt.tensordesc<64x64xf16, #shared>,
  %qk_scale: f32,
  %n_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32

  %neg_inf = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %zero = arith.constant dense<0.0> : tensor<256x64xf32, #blocked>
  %one = arith.constant dense<1.0> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>


  %loop_outs:4 = scf.for %i = %c0_i32 to %n_tiles step %c64_i32 iter_args(
    %l_i = %one,
    %acc = %zero,
    %m_i = %neg_inf,
    %e_i = %one
  ) -> (
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
    tensor<256x64xf32, #blocked>,
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  ) : i32 {

    // CHECK-COUNT-2: ttg.partition = array<i32: 3>
    %K = tt.descriptor_load %K_desc[%i, %c0_i32] : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #load_blocked>
    %K_shared = ttg.local_alloc %K : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

    %QK_tmem, %QK_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-COUNT-2: ttg.partition = array<i32: 2>
    %K_trans = ttg.memdesc_trans %K_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>
    %QK_mma_tok = ttng.tc_gen5_mma %Q_shared, %K_trans, %QK_tmem[%QK_tok], %false, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK-COUNT-3: ttg.partition = array<i32: 0>
    %QK, %QK_load_tok = ttng.tmem_load %QK_tmem[%QK_mma_tok] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    %row_max = "compute_row_max"(%QK, %qk_scale) : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %QK_adj = "sub_row_max"(%QK, %row_max, %qk_scale) : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
    // CHECK: [[SOFTMAX:%.*]] = math.exp2 {{.*}} {ttg.partition = array<i32: 0>} : tensor<256x64xf32
    %softmax = math.exp2 %QK_adj : tensor<256x64xf32, #blocked>
    // CHECK-COUNT-4: ttg.partition = array<i32:
    %diff = arith.subf %m_i, %row_max : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %alpha = math.exp2 %diff : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    // CHECK-NEXT: tt.reduce
    %l_ij = "tt.reduce"(%softmax) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      // CHECK-COUNT-2: ttg.partition = array<i32: 0>
      %68 = arith.addf %arg29, %arg30 : f32
      tt.reduce.return %68 : f32
      // CHECK-NEXT: ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]
    }) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK-COUNT-6: ttg.partition = array<i32:
    %l_i_scaled = arith.mulf %l_i, %alpha : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %next_l_i = arith.addf %l_i_scaled, %l_ij : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %alpha_0 = tt.expand_dims %alpha {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
    %alpha_1 = tt.broadcast %alpha_0 : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>

    %acc_corrected = arith.mulf %acc, %alpha_1 : tensor<256x64xf32, #blocked>

    // CHECK-NEXT: [[X:%.*]] = arith.addf [[SOFTMAX]], [[SOFTMAX]] {ttg.partition = array<i32: 1>}
    %x = arith.addf %softmax, %softmax : tensor<256x64xf32, #blocked>
    // CHECK-NEXT: [[ACC_X:%.*]] = arith.addf %{{.*}}, [[X]] {ttg.partition = array<i32: 1>}
    // CHECK-COUNT-8: ttg.partition = array<i32:
    %acc_x = arith.addf %acc, %x : tensor<256x64xf32, #blocked>
    %e = "sum"(%acc_x) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %next_e_i = arith.addf %e_i, %e : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %V = tt.descriptor_load %V_desc[%i, %c0_i32] : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #load_blocked>
    %V_shared = ttg.local_alloc %V : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %P = arith.truncf %softmax : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>

    %P_tmem = ttng.tmem_alloc %P : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory>
    %acc_tmem, %acc_tok = ttng.tmem_alloc %acc_corrected : (tensor<256x64xf32, #blocked>) -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %PV_mma_tok = ttng.tc_gen5_mma %P_tmem, %V_shared, %acc_tmem[%acc_tok], %true, %true : !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %O, %O_tok = ttng.tmem_load %acc_tmem[%PV_mma_tok] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

    // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>}
    scf.yield %next_l_i, %O, %row_max, %next_e_i : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT: ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 2>, array<i32: 1>]
  } {tt.warp_specialize}

  "use"(%loop_outs#0, %loop_outs#1, %loop_outs#2) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()

  tt.return
}

// CHECK-LABEL: @mma_operand_view
tt.func public @mma_operand_view(
  %Q_shared: !ttg.memdesc<256x64xf16, #shared, #smem>,
  %K_desc: !tt.tensordesc<64x64xf16, #shared>,
  %V_desc: !tt.tensordesc<64x64xf16, #shared>,
  %qk_scale: f32,
  %n_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32

  %neg_inf = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %zero = arith.constant dense<0.0> : tensor<256x64xf32, #blocked>
  %one = arith.constant dense<1.0> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

  %QK_tmem, %QK_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

  scf.for %i = %c0_i32 to %n_tiles step %c64_i32 : i32 {
    %K = tt.descriptor_load %K_desc[%i, %c0_i32] : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #load_blocked>
    // CHECK: [[K_SHARED:%.*]] = ttg.local_alloc {{.*}}partition = array<i32: 2>
    %K_shared = ttg.local_alloc %K : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

    // CHECK-DAG: [[TRANS_MMA:%.*]] = ttg.memdesc_trans [[K_SHARED]] {{.*}}partition = array<i32: 1>
    // CHECK-DAG: [[K_VIEW:%.*]] = ttg.memdesc_subslice [[TRANS_MMA]]{{.*}}partition = array<i32: 1>
    // CHECK-DAG: [[TRANS_USER:%.*]] = ttg.memdesc_trans [[K_SHARED]] {{.*}}partition = array<i32: 0>
    %K_trans = ttg.memdesc_trans %K_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>
    %K_view = ttg.memdesc_subslice %K_trans [0, 0]  : !ttg.memdesc<64x64xf16, #shared_T, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>

    // CHECK: ttng.tc_gen5_mma %arg0, [[K_VIEW]]{{.*}}partition = array<i32: 1>
    %QK_mma_tok = ttng.tc_gen5_mma %Q_shared, %K_view, %QK_tmem[%QK_tok], %false, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: local_load [[TRANS_USER]] {{.*}}partition = array<i32: 0>
    %x = ttg.local_load %K_trans : !ttg.memdesc<64x64xf16, #shared_T, #smem> -> tensor<64x64xf16, #load_blocked>

    // CHECK: tmem_load {{.*}}partition = array<i32: 0>
    %QK, %QK_load_tok = ttng.tmem_load %QK_tmem[%QK_mma_tok] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

    "use"(%x, %QK) {data} : (tensor<64x64xf16, #load_blocked>, tensor<256x64xf32, #blocked>) -> ()
    // CHECK: "use"
    // CHECK-NEXT: ttg.partition = array<i32: 0, 1, 2>
  } {tt.warp_specialize}

  tt.return
}

// CHECK-LABEL: @optimize_broadcast
tt.func @optimize_broadcast(%arg0: i32, %arg1: !tt.tensordesc<128x128xf32, #shared_f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: scf.for
  scf.for %i = %c0_i32 to %arg0 step %c1_i32 : i32 {
    %md = tt.descriptor_load %arg1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x128xf32, #shared_f32> -> tensor<128x128xf32, #load_blocked>
    %smem = ttg.local_alloc %md {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #load_blocked>) -> !ttg.memdesc<128x128xf32, #shared_f32, #smem>
    %tmp = ttg.local_load %smem {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #shared_f32, #smem> -> tensor<128x128xf32, #load_blocked>
    "use_memdesc"(%tmp) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #load_blocked>) -> ()

    // CHECK: [[X:%.*]] = "producer"{{.*}}partition = array<i32: 0>
    %x = "producer"() {ttg.partition = array<i32: 0>, data} : () -> tensor<128xf32>

    // CHECK-DAG: [[X0_P0:%.*]] = tt.expand_dims [[X]] {{.*}}partition = array<i32: 0>
    // CHECK-DAG: [[X0_P1:%.*]] = tt.expand_dims [[X]] {{.*}}partition = array<i32: 1>
    %x0 = tt.expand_dims %x {axis = 0 : i32} : tensor<128xf32> -> tensor<1x128xf32>
    // CHECK-DAG: [[X1_P0:%.*]] = tt.broadcast [[X0_P0]] {{.*}}partition = array<i32: 0>
    // CHECK-DAG: [[X1_P1:%.*]] = tt.broadcast [[X0_P1]] {{.*}}partition = array<i32: 1>
    %x1 = tt.broadcast %x0 : tensor<1x128xf32> -> tensor<128x128xf32>

    // CHECK: "use"([[X1_P0]]) {{.*}}partition = array<i32: 0>
    "use"(%x1) {ttg.partition = array<i32: 0>, data} : (tensor<128x128xf32>) -> ()
    // CHECK: "use"([[X1_P1]]) {{.*}}partition = array<i32: 1>
    "use"(%x1) {ttg.partition = array<i32: 1>, data} : (tensor<128x128xf32>) -> ()
    // CHECK-NEXT: ttg.partition = array<i32: 0, 1>
  } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @no_partitions
tt.func @no_partitions(%arg0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: scf.for %{{.*}} = %c0_i32 to %arg0 step %c1_i32 : i32
  // CHECK-NOT: ttg.partition
  // CHECK-NOT: ttg.warp_specialize.tag
  scf.for %i = %c0_i32 to %arg0 step %c1_i32 : i32 {
    "use"(%c0_i32) : (i32) -> ()
  } {tt.warp_specialize}
  tt.return
}

// CHECK-LABEL: @mma_no_memory_ops
tt.func @mma_no_memory_ops(%arg0: i32, %arg1: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg2: !ttg.memdesc<64x64xf16, #shared_T, #smem>, %arg3: !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %false = arith.constant false
  %true = arith.constant true
  // CHECK: scf.for %{{.*}} = %c0_i32 to %arg0 step %c1_i32 : i32
  // CHECK-NOT: ttg.partition
  // CHECK-NOT: ttg.warp_specialize.tag
  scf.for %i = %c0_i32 to %arg0 step %c1_i32 : i32 {
    %0 = ttng.tc_gen5_mma %arg1, %arg2, %arg3[], %false, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
  } {tt.warp_specialize}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @matmul_change_desc_in_prologue
  tt.func @matmul_change_desc_in_prologue(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %0 = ub.poison : !tt.tensordesc<128x64xf16, #shared>
    %1 = ub.poison : !tt.tensordesc<64x128xf16, #shared>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %2 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for
    %3:4 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0, %arg5 = %1, %arg6 = %2) -> (i1, !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !ttg.async.token)  : i32 {
      // CHECK-NEXT: "prologue_cond"({{.*}}) {ttg.partition = array<i32: 2>}
      %4 = "prologue_cond"(%arg2) : (i32) -> i1
      // CHECK-NEXT: scf.if
      %5:2 = scf.if %4 -> (!tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>) {
        // CHECK-COUNT-2: ttg.partition = array<i32: 2>
        %15 = tt.make_tensor_descriptor %arg0, [%arg2, %arg2], [%c1_i64, %c1_i64] : <f16>, <128x64xf16, #shared>
        %16 = tt.make_tensor_descriptor %arg1, [%arg2, %arg2], [%c1_i64, %c1_i64] : <f16>, <64x128xf16, #shared>
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 2>}
        scf.yield %15, %16 : !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>
      } else {
        // CHECK-NEXT: } else {
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 2>}
        scf.yield %arg4, %arg5 : !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>
        // CHECK-NEXT: ttg.partition = array<i32: 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]
      }
      // CHECK-COUNT-5: ttg.partition = array<i32: 2>
      %6:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %7 = tt.descriptor_load %arg4[%6#0, %6#2] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %8 = tt.descriptor_load %arg5[%6#1, %6#2] : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %9 = ttg.local_alloc %7 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %10 = ttg.local_alloc %8 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK-NEXT: tc_gen5_mma {{.*}} {ttg.partition = array<i32: 1>} {{.*}}
      %11 = ttng.tc_gen5_mma %9, %10, %result[%arg6], %arg3, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: ttg.partition = array<i32: 0, 1>
      %12 = arith.cmpi eq, %arg2, %c0_i32 : i32
      // CHECK-NEXT: ttg.partition = array<i32: 1>
      %13 = arith.select %12, %false, %true : i1
      // CHECK-NEXT: scf.if
      %14 = scf.if %12 -> (!ttg.async.token) {
        // CHECK-COUNT-2: ttg.partition = array<i32: 0>
        %result_0, %token_1 = ttng.tmem_load %result[%11] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>}
        scf.yield %token_1 : !ttg.async.token
      } else {
        // CHECK-NEXT: } else {
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>}
        // CHECK-NEXT: ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]
        scf.yield %11 : !ttg.async.token
      }
      // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1, 2>}
      scf.yield %13, %5#0, %5#1, %14 : i1, !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !ttg.async.token
      // CHECK-NEXT: ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 2>, array<i32: 2>, array<i32: 1>]
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 4 : i32, tt.warp_specialize}
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use
  tt.func @matmul_tma_acc_with_conditional_def_and_use(%arg0: !tt.tensordesc<1x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for
    %1:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0) -> (i1, !ttg.async.token)  : i32 {
      // CHECK-COUNT-6: ttg.partition = array<i32: 2>
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.splat %2#0 : i32 -> tensor<128xi32, #blocked2>
      %4 = tt.descriptor_gather %arg0[%3, %2#2] : (!tt.tensordesc<1x64xf16, #shared>, tensor<128xi32, #blocked2>, i32) -> tensor<128x64xf16, #blocked1>
      %5 = tt.descriptor_load %arg1[%2#1, %2#2] : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %6 = ttg.local_alloc %4 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %7 = ttg.local_alloc %5 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK-NEXT: ttg.partition = array<i32: 1>
      %8 = ttng.tc_gen5_mma %6, %7, %result[%arg4], %arg3, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: ttg.partition = array<i32: 0, 1>
      %9 = arith.cmpi eq, %arg2, %c0_i32 : i32
      // CHECK-NEXT: ttg.partition = array<i32: 1>
      %10 = arith.select %9, %false, %true : i1
      // CHECK-NEXT: scf.if
      %11 = scf.if %9 -> (!ttg.async.token) {
        // CHECK-COUNT-2: ttg.partition = array<i32: 0>
        %result_0, %token_1 = ttng.tmem_load %result[%8] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>}
        scf.yield %token_1 : !ttg.async.token
      } else {
        // CHECK-NEXT: } else {
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>}
        // CHECK-NEXT: ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]
        scf.yield %8 : !ttg.async.token
      }
      // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1, 2>}
      scf.yield %10, %11 : i1, !ttg.async.token
      // CHECK-NEXT: ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize}
    tt.return
  }

}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0], [0, 32]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, rank = 3}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @if_stmt_yield_outputs
  tt.func @if_stmt_yield_outputs(%lb: i32, %ub: i32, %step: i32,
                                 %a0: i32, %b0: i32,
                                 %arg1: !tt.tensordesc<1x128x64xbf16, #shared> {tt.nv_tma_desc = 1 : i32},
                                 %arg2: !tt.tensordesc<1x64x64xf32, #shared1> {tt.nv_tma_desc = 1 : i32}) {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<448> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #linear>
    // CHECK: scf.for
    scf.for %arg3 = %lb to %ub step %step : i32 {
      // CHECK-NEXT: tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>} {{.*}}
      %20 = tt.descriptor_load %arg1[%a0, %b0, %c0_i32] : !tt.tensordesc<1x128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
      %22 = arith.cmpi sge, %arg3, %c3_i32 : i32
      // CHECK: scf.if
      %23 = scf.if %22 -> (tensor<128x64xbf16, #blocked>) {
        %32 = arith.muli %arg3, %c128_i32 : i32
        %36 = tt.splat %32 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %38 = arith.cmpi slt, %36, %cst : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %39 = tt.expand_dims %38 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
        %40 = tt.broadcast %39 : tensor<128x1xi1, #blocked> -> tensor<128x64xi1, #blocked>
        //  CHECK: arith.select {{.*}} {ttg.partition = array<i32: 0>} {{.*}}
        //  CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0>}
        %41 = arith.select %40, %20, %cst_1 : tensor<128x64xi1, #blocked>, tensor<128x64xbf16, #blocked>
        scf.yield %41 : tensor<128x64xbf16, #blocked>
      } else {
        scf.yield %20 : tensor<128x64xbf16, #blocked>
      }
      // CHECK-NEXT: } else {
      // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]
      "use"(%23) {data, mma} : (tensor<128x64xbf16, #blocked>) -> ()
      // CHECK: "use"
      // CHECK-NEXT ttg.warp_specialize.tag = 0 : i32
    } {tt.warp_specialize = true}

    // CHECK: scf.for
    scf.for %arg3 = %lb to %ub step %step : i32 {
      %20 = tt.descriptor_load %arg1[%a0, %b0, %c0_i32] : !tt.tensordesc<1x128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
      %22 = arith.cmpi sge, %arg3, %c3_i32 : i32
      %23 = scf.if %22 -> (tensor<128x64xbf16, #blocked>) {
        %32 = arith.muli %arg3, %c128_i32 {ttg.partition = array<i32: 0>} : i32
        %36 = tt.splat %32 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %38 = arith.cmpi slt, %36, %cst {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %39 = tt.expand_dims %38 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
        %40 = tt.broadcast %39 {ttg.partition = array<i32: 0>} : tensor<128x1xi1, #blocked> -> tensor<128x64xi1, #blocked>
        %41 = arith.select %40, %20, %cst_1 : tensor<128x64xi1, #blocked>, tensor<128x64xbf16, #blocked>
        scf.yield %41 : tensor<128x64xbf16, #blocked>
      } else {
        scf.yield %20 : tensor<128x64xbf16, #blocked>
      }
      "use"(%23) {data} : (tensor<128x64xbf16, #blocked>) -> ()
      // CHECK: "use"
      // CHECK-NEXT: ttg.warp_specialize.tag = 1 : i32
    } {tt.warp_specialize = true}


    // CHECK: scf.for
    scf.for %arg4 = %lb to %ub step %step : i32 {
      %20 = tt.descriptor_load %arg1[%a0, %b0, %c0_i32] : !tt.tensordesc<1x128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
      %22 = arith.cmpi sge, %arg4, %c3_i32 : i32
      // CHECK: scf.if
      %23 = scf.if %22 -> (tensor<128x64xbf16, #blocked>) {
        scf.yield %20 : tensor<128x64xbf16, #blocked>
        // CHECK: scf.yield {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: } else {
      } else {
        %32 = arith.muli %arg4, %c128_i32 : i32
        %36 = tt.splat %32 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %38 = arith.cmpi slt, %36, %cst : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %39 = tt.expand_dims %38 {axis = 1 : i32} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi1, #blocked>
        %40 = tt.broadcast %39 : tensor<128x1xi1, #blocked> -> tensor<128x64xi1, #blocked>
        //  CHECK: arith.select {{.*}} {ttg.partition = array<i32: 0>} {{.*}}
        //  CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0>}
        %41 = arith.select %40, %20, %cst_1 : tensor<128x64xi1, #blocked>, tensor<128x64xbf16, #blocked>
        scf.yield %41 : tensor<128x64xbf16, #blocked>
      }
      // CHECK-NEXT: ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]
      "use"(%23) {data, mma} : (tensor<128x64xbf16, #blocked>) -> ()
    } {tt.warp_specialize = true}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_nested_persistent_ws_kernel
  tt.func public @matmul_nested_persistent_ws_kernel(%a_desc_0: !tt.tensordesc<128x128xf8E4M3FN, #shared>, %b_desc_1: !tt.tensordesc<128x128xf8E4M3FN, #shared>, %c_desc_2: !tt.tensordesc<128x128xf8E4M3FN, #shared>, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %start_pid = tt.get_program_id x : i32
    %num_pid_m_3 = arith.divsi %M, %c128_i32 : i32
    %num_pid_n_4 = arith.divsi %N, %c128_i32 : i32
    %k_tiles_5 = arith.divsi %K, %c128_i32 : i32
    %num_tiles = arith.muli %num_pid_m_3, %num_pid_n_4 : i32
    %num_pid_in_group = arith.muli %num_pid_n_4, %c8_i32 : i32
    // CHECK: scf.for
    scf.for %tile_id = %start_pid to %num_tiles step %c148_i32  : i32 {
      // CHECK-COUNT-10: {ttg.partition = array<i32: 0, 2>}
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %num_pid_m_3, %first_pid_m : i32
      %group_size_m_6 = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_6 : i32
      %pid_m_7 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_8 = arith.divsi %pid_n, %group_size_m_6 : i32
      %off_am = arith.muli %pid_m_7, %c128_i32 : i32
      %off_bn = arith.muli %pid_n_8, %c128_i32 : i32
      // CHECK-NEXT: {ttg.partition = array<i32: 0, 1>}
      %accumulator, %accumulator_9 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK-NEXT: {ttg.partition = array<i32: 0>}
      %accumulator_10 = ttng.tmem_store %cst, %accumulator[%accumulator_9], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: scf.for
      %accumulator_11:2 = scf.for %accumulator_15 = %c0_i32 to %k_tiles_5 step %c1_i32 iter_args(%arg11 = %false, %accumulator_16 = %accumulator_10) -> (i1, !ttg.async.token)  : i32 {
	// CHECK: arith.muli {{.*}}ttg.partition = array<i32: 2>}
        %off_k = arith.muli %accumulator_15, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        // CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: 2>}
        %a = tt.descriptor_load %a_desc_0[%off_am, %off_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf8E4M3FN, #shared> -> tensor<128x128xf8E4M3FN, #blocked1>
        %a_17 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %b = tt.descriptor_load %b_desc_1[%off_bn, %off_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf8E4M3FN, #shared> -> tensor<128x128xf8E4M3FN, #blocked1>
        %accumulator_18 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %accumulator_19 = ttg.memdesc_trans %accumulator_18 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        // CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: 1>}
        %accumulator_20 = ttng.tc_gen5_mma %a_17, %accumulator_19, %accumulator[%accumulator_16], %arg11, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_20 : i1, !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
      } {tt.scheduled_max_stage = 2 : i32}
      // CHECK-COUNT-4: {ttg.partition = array<i32: 0>}
      %accumulator_12, %accumulator_13 = ttng.tmem_load %accumulator[%accumulator_11#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %c = tt.fp_to_fp %accumulator_12, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %c_14 = ttg.convert_layout %c : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %c_desc_2[%off_am, %off_bn], %c_14 : !tt.tensordesc<128x128xf8E4M3FN, #shared>, tensor<128x128xf8E4M3FN, #blocked1>
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}

// -----

// Verify that TCGen5MMAScaledOp is classified as a data value in partition
// scheduling, just like TCGen5MMAOp. Both ops have an optional async token
// as output 0, and initialDataValues should mark it as a data value so that
// partition scheduling properly propagates the data dependency.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared_scales = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>

#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @scaled_mma_with_loads
tt.func public @scaled_mma_with_loads(
  %A_shared: !ttg.memdesc<128x128xf16, #shared, #smem>,
  %B_desc: !tt.tensordesc<128x128xf16, #shared>,
  %A_scale_shared: !ttg.memdesc<1x2x32x4x4xi8, #shared_scales, #smem>,
  %B_scale_shared: !ttg.memdesc<1x2x32x4x4xi8, #shared_scales, #smem>,
  %n_tiles: i32
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  %acc_tmem, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

  // CHECK: scf.for
  %loop_out:2 = scf.for %i = %c0_i32 to %n_tiles step %c1_i32 iter_args(
    %iter_acc_tok = %acc_tok,
    %iter_acc_tmem = %acc_tmem
  ) -> (
    !ttg.async.token,
    !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ) : i32 {

    // Load partition. Feeding this load into the MMA keeps the test live after
    // canonicalization while still requiring the scaled MMA token result to
    // propagate the dependency to tmem_load.
    // CHECK-COUNT-2: ttg.partition = array<i32: 2>
    %B = tt.descriptor_load %B_desc[%i, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #load_blocked>
    %B_shared = ttg.local_alloc %B : (tensor<128x128xf16, #load_blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>

    // Compute partition: tc_gen5_mma_scaled should get partition 1
    // just like tc_gen5_mma does in the existing tests.
    // CHECK: ttg.memdesc_trans {{.*}} {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>}
    %B_trans = ttg.memdesc_trans %B_shared {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared_T, #smem>
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {ttg.partition = array<i32: 1>}
    %mma_tok = ttng.tc_gen5_mma_scaled %A_shared, %B_trans, %iter_acc_tmem[%iter_acc_tok], %A_scale_shared, %B_scale_shared, %true, %true lhs = e5m2 rhs = e5m2 : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared_T, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared_scales, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared_scales, #smem>

    // Data partition: tmem_load should get partition 0
    // CHECK-COUNT-2: ttg.partition = array<i32: 0>
    %QK, %QK_load_tok = ttng.tmem_load %iter_acc_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

    "use"(%QK) {data} : (tensor<128x128xf32, #blocked>) -> ()

    scf.yield %QK_load_tok, %iter_acc_tmem : !ttg.async.token, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]
  } {tt.warp_specialize}

  "use"(%loop_out#0) : (!ttg.async.token) -> ()
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32, 1], warpsPerCTA = [1, 2, 2, 1, 1], order = [3, 2, 1, 0, 4]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared_scale_tma = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#shared_scale_a = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4], [0, 0, 0, 0, 8], [0, 0, 0, 0, 16], [0, 0, 0, 0, 32], [0, 0, 0, 0, 64], [0, 0, 0, 0, 128], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 1, 0, 0, 0]]}, alignment = 128>
#shared_scale_a_rs = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 0, 0, 0, 0]]}, alignment = 128>
#shared_scale_a_tr = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [1, 0, 0, 0, 0]]}, alignment = 128>
#shared_scale_a_final = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [128, 0]]}, alignment = 128>
#shared_scale_b = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4], [0, 0, 0, 0, 8], [0, 0, 0, 0, 16], [0, 0, 0, 0, 32], [0, 0, 0, 0, 64], [0, 0, 0, 0, 128], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0]]}, alignment = 128>
#shared_scale_b_rs = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0]]}, alignment = 128>
#shared_scale_b_tr = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0]]}, alignment = 128>
#shared_scale_b_final = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4], [0, 8]]}, alignment = 128>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @scaled_mma_descriptor_scales
tt.func public @scaled_mma_descriptor_scales(
  %A_shared: !ttg.memdesc<256x128xi8, #shared, #smem>,
  %B_shared: !ttg.memdesc<128x128xi8, #shared_T, #smem>,
  %A_scale_desc: !tt.tensordesc<1x2x4x2x256xf8E4M3FN, #shared_scale_tma>,
  %B_scale_desc: !tt.tensordesc<1x1x4x2x256xf8E4M3FN, #shared_scale_tma>,
  %n_tiles: i32
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  %acc_tmem, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

  %loop_out = scf.for %i = %c0_i32 to %n_tiles step %c1_i32 iter_args(
    %iter_acc_tok = %acc_tok
  ) -> (!ttg.async.token) : i32 {
    // CHECK: %[[A_SCALE:[0-9]+]] = tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
    %A_scale = tt.descriptor_load %A_scale_desc[%c0_i32, %c0_i32, %i, %c0_i32, %c0_i32] : !tt.tensordesc<1x2x4x2x256xf8E4M3FN, #shared_scale_tma> -> tensor<1x2x4x2x256xf8E4M3FN, #blocked>
    // CHECK: %[[B_SCALE:[0-9]+]] = tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
    %B_scale = tt.descriptor_load %B_scale_desc[%c0_i32, %c0_i32, %i, %c0_i32, %c0_i32] : !tt.tensordesc<1x1x4x2x256xf8E4M3FN, #shared_scale_tma> -> tensor<1x1x4x2x256xf8E4M3FN, #blocked>
    // CHECK: ttg.local_alloc %[[A_SCALE]] {ttg.partition = array<i32: 2>} : (tensor<1x2x4x2x256xf8E4M3FN, #blocked>) -> !ttg.memdesc<1x2x4x2x256xf8E4M3FN, {{#[A-Za-z0-9_]+}}, #smem>
    %A_scale_shared = ttg.local_alloc %A_scale : (tensor<1x2x4x2x256xf8E4M3FN, #blocked>) -> !ttg.memdesc<1x2x4x2x256xf8E4M3FN, #shared_scale_a, #smem>
    // CHECK: ttg.local_alloc %[[B_SCALE]] {ttg.partition = array<i32: 2>} : (tensor<1x1x4x2x256xf8E4M3FN, #blocked>) -> !ttg.memdesc<1x1x4x2x256xf8E4M3FN, {{#[A-Za-z0-9_]+}}, #smem>
    %B_scale_shared = ttg.local_alloc %B_scale : (tensor<1x1x4x2x256xf8E4M3FN, #blocked>) -> !ttg.memdesc<1x1x4x2x256xf8E4M3FN, #shared_scale_b, #smem>

    // CHECK: ttg.memdesc_reshape {{.*}} {ttg.partition = array<i32: 1>}
    %A_scale_rs = ttg.memdesc_reshape %A_scale_shared : !ttg.memdesc<1x2x4x2x256xf8E4M3FN, #shared_scale_a, #smem> -> !ttg.memdesc<2x4x32x4x4xf8E4M3FN, #shared_scale_a_rs, #smem>
    // CHECK: ttg.memdesc_trans {{.*}} {order = array<i32: 0, 3, 2, 1, 4>, ttg.partition = array<i32: 1>}
    %A_scale_tr = ttg.memdesc_trans %A_scale_rs {order = array<i32: 0, 3, 2, 1, 4>} : !ttg.memdesc<2x4x32x4x4xf8E4M3FN, #shared_scale_a_rs, #smem> -> !ttg.memdesc<2x4x32x4x4xf8E4M3FN, #shared_scale_a_tr, #smem>
    %A_scale_final = ttg.memdesc_reshape %A_scale_tr : !ttg.memdesc<2x4x32x4x4xf8E4M3FN, #shared_scale_a_tr, #smem> -> !ttg.memdesc<256x16xf8E4M3FN, #shared_scale_a_final, #smem>
    %B_scale_rs = ttg.memdesc_reshape %B_scale_shared : !ttg.memdesc<1x1x4x2x256xf8E4M3FN, #shared_scale_b, #smem> -> !ttg.memdesc<1x4x32x4x4xf8E4M3FN, #shared_scale_b_rs, #smem>
    %B_scale_tr = ttg.memdesc_trans %B_scale_rs {order = array<i32: 0, 3, 2, 1, 4>} : !ttg.memdesc<1x4x32x4x4xf8E4M3FN, #shared_scale_b_rs, #smem> -> !ttg.memdesc<1x4x32x4x4xf8E4M3FN, #shared_scale_b_tr, #smem>
    %B_scale_final = ttg.memdesc_reshape %B_scale_tr : !ttg.memdesc<1x4x32x4x4xf8E4M3FN, #shared_scale_b_tr, #smem> -> !ttg.memdesc<128x16xf8E4M3FN, #shared_scale_b_final, #smem>

    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {ttg.partition = array<i32: 1>}
    %mma_tok = ttng.tc_gen5_mma_scaled %A_shared, %B_shared, %acc_tmem[%iter_acc_tok], %A_scale_final, %B_scale_final, %true, %true lhs = e2m1 rhs = e2m1 : !ttg.memdesc<256x128xi8, #shared, #smem>, !ttg.memdesc<128x128xi8, #shared_T, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xf8E4M3FN, #shared_scale_a_final, #smem>, !ttg.memdesc<128x16xf8E4M3FN, #shared_scale_b_final, #smem>

    // CHECK-COUNT-2: ttg.partition = array<i32: 0>
    %acc, %load_tok = ttng.tmem_load %acc_tmem[%mma_tok] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
    "use"(%acc) {data} : (tensor<256x128xf32, #linear>) -> ()

    // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>}
    scf.yield %load_tok : !ttg.async.token
    // CHECK: ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]
  } {tt.warp_specialize}

  "use"(%loop_out) : (!ttg.async.token) -> ()
  tt.return
}

}

// -----

// CHECK-LABEL: attention_persistent_inner_loop_kernel
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_persistent_inner_loop_kernel(%desc_q: !tt.tensordesc<128x128xf16, #shared>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<128x128xf16, #shared>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<128x128xf16, #shared>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %desc_acc: !tt.tensordesc<128x128xf16, #shared>, %desc_acc_12: i32, %desc_acc_13: i32, %desc_acc_14: i64, %desc_acc_15: i64, %l_i_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %m_i_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %qk_scale: f32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_16 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %prog_id = tt.get_program_id x : i32
    %num_sm = tt.get_num_programs x : i32
    %num_tiles = arith.divsi %M, %c128_i32 : i32
    %tiles_per_sm = arith.divsi %num_tiles, %num_sm : i32
    // CHECK: scf.for
    %tile_idx = scf.for %_ = %c0_i32 to %tiles_per_sm step %c1_i32 iter_args(%tile_idx_20 = %prog_id) -> (i32)  : i32 {
      %off_m = arith.muli %tile_idx_20, %c128_i32 : i32
      %q = tt.descriptor_load %desc_q[%off_m, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked2>
      %q_21 = ttg.local_alloc %q : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %qk_22, %qk_23 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc, %acc_24 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_25 = ttng.tmem_store %cst_17, %acc[%acc_24], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: scf.for
      %acc_26:4 = scf.for %acc_30 = %c0_i32 to %N step %c128_i32 iter_args(%arg28 = %cst_16, %arg29 = %cst, %qk_31 = %qk_23, %acc_32 = %acc_25) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %k = tt.descriptor_load %desc_k[%acc_30, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked2>
        %k_33 = ttg.local_alloc %k : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %k_34 = ttg.memdesc_trans %k_33 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>
        %qk_35 = ttng.tc_gen5_mma %q_21, %k_34, %qk_22[%qk_31], %false, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: tmem_load {{.*}} {ttg.partition = array<i32: 0>}
        %qk_36, %qk_37 = ttng.tmem_load %qk_22[%qk_35] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: "softmax_work"{{.*}}ttg.partition = array<i32: 0>}
        %acc_47, %p, %next_l_i, %row_max = "softmax_work"(%qk_36, %arg29, %arg28) : (tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf16, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)
        %p_53 = ttg.local_alloc %p : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>

        // CHECK-COUNT-3: {ttg.partition = array<i32: 1>}
        %acc_48, %acc_49 = ttng.tmem_load %acc[%acc_32] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_50 = arith.mulf %acc_48, %acc_47 : tensor<128x128xf32, #blocked>
        %acc_54 = ttng.tmem_store %acc_50, %acc[%acc_49], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %v = tt.descriptor_load %desc_v[%acc_30, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked2>
        %v_51 = ttg.local_alloc %v : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>

        %acc_55 = ttng.tc_gen5_mma %p_53, %v_51, %acc[%acc_54], %true, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

        scf.yield %row_max, %next_l_i, %qk_37, %acc_55 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 2>, array<i32: 1>]}
      }
      // CHECK: arith.addi {{.*}}, {{.*}} {ttg.partition = array<i32: 3>}
      %tile_idx_29 = arith.addi %tile_idx_20, %num_sm : i32
      scf.yield %tile_idx_29 : i32
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}
