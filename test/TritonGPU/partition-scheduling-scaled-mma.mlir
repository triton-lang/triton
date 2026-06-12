// RUN: triton-opt %s --split-input-file --tritongpu-hoist-tmem-alloc --tritongpu-partition-scheduling -allow-unregistered-dialect | FileCheck %s

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
