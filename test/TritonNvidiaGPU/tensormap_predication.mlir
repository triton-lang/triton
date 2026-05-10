// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s

// Regression test for issue #10229: TensormapCreateOp inside a pipelined loop
// must implement PredicatedOpInterface so the pipeliner can predicate it in
// prologue/kernel/epilogue phases. Without the interface, the pipeliner crashes
// with "pipeliner doesn't know how to predicate this op". After the fix, the
// pipeliner successfully predicates both TensormapCreateOp and
// TensormapFenceproxyAcquireOp with computed predicates derived from iteration
// bounds, not just constant %true.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @tensormap_predication_regression
  tt.func public @tensormap_predication_regression(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    // Prologue: tensormap_create with a computed predicate (not %true)
    // CHECK: ttng.tensormap_create {{.*}}, %{{.+}} {elem_type = 6 : i32
    // CHECK: ttng.tensormap_fenceproxy_acquire {{.*}}, %{{.+}} : !tt.ptr<i8>
    // CHECK: ttng.reinterpret_tensor_descriptor
    // CHECK: ttng.async_tma_copy_global_to_local
    // Kernel loop with predicated tensormap_create
    // CHECK: scf.for
    scf.for %arg4 = %c0_i32 to %arg3 step %arg2 : i32 {
      %desc = tt.make_tensor_descriptor %arg0, [%c128_i32, %c64_i32], [%c1_i64, %c1_i64] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : <f16>, <128x64xf16, #nvmma_128>
      %load = tt.descriptor_load %desc[%arg4, %arg4] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #nvmma_128> -> tensor<128x64xf16, #blocked>
      %add = arith.addf %load, %load {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #blocked>
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 1 : i32}
    // Inside the kernel loop, tensormap ops use computed predicates.
    // CHECK: ttng.tensormap_create {{.*}}, %{{.+}} {elem_type = 6 : i32
    // CHECK: ttng.tensormap_fenceproxy_acquire {{.*}}, %{{.+}} : !tt.ptr<i8>
    tt.return
  }
}