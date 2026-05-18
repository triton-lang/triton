// RUN: not --crash triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-test-pipeline-lower-loop 2>&1 | FileCheck %s

// Regression test for the `CoarseSchedule::splitClusterBefore`
// implicit-insert bug.
//
// Before the fix, `splitClusterBefore` indexed `opToStageAndCluster`
// (an `llvm::MapVector`) with `operator[]`, which silently inserts a
// default entry for any key that is not present. Any op in the loop
// body without `loop.cluster` / `loop.stage` attributes would therefore
// be added to the schedule with stage 0 and a default-constructed
// (invalid) cluster iterator. That "phantom" entry was later asserted
// on as "Op with invalid cluster!" inside
// `CoarseSchedule::getOpsInOrder`, crashing the pipeliner deep in the
// pipeline with no actionable diagnostic.
//
// With the fix, `splitClusterBefore` uses `find` and skips ops that
// are not in the schedule. The pass instead reports the unscheduled op
// cleanly via `lowerLoads`'s "op not found in the schedule" diagnostic
// at the offending operation.

// CHECK: error: op not found in the schedule
// CHECK-NEXT: "unscheduled.op"() : () -> ()
// CHECK-NOT: Op with invalid cluster

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @split_cluster_with_unscheduled_op(
      %arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %arg3: i32) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %arg5 = %c0_i32 to %arg3 step %c1_i32 iter_args(%tok = %acc_tok) -> !ttg.async.token : i32 {
      %2 = tt.load %arg0 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %3 = ttg.local_alloc %2 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %4 = tt.load %arg1 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg2 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
      // An op missing loop.cluster / loop.stage attributes inside the
      // window scanned by splitClusterBefore. The buggy version would
      // default-insert this op into opToStageAndCluster and crash later
      // with "Op with invalid cluster!". The fix must skip it instead.
      "unscheduled.op"() : () -> ()
      %store_tok = ttng.tmem_store %6, %0[%tok], %true {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma_tok = ttng.tc_gen5_mma %3, %5, %0[%store_tok], %true, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %res, %load_tok = ttng.tmem_load %0[%mma_tok] {loop.cluster = 2 : i32, loop.stage = 3 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %load_tok : !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}
    tt.return
  }
}
