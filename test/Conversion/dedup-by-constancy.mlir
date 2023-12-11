// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=target=rocdl | FileCheck %s --check-prefixes=CHECK,GCN

// CHECK-LABEL: dedup_by_constancy_full
// GCN-COUNT-26: llvm.add
// PTX-COUNT-5: llvm.add
// CHECK-NOT: llvm.add
// CHECK: llvm.icmp "slt"
// PTX-NOT: llvm.icmp "slt"
// CHECK: llvm.sdiv
// PTX-NOT: llvm.sdiv
// PTX: llvm.getelementptr %arg0[[[REGISTER:%[0-9]+]]]
// PTX-COUNT-7: llvm.getelementptr %arg0[[[REGISTER]]]
// GCN-COUNT-16: llvm.getelementptr
// PTX-NOT: llvm.getelementptr %arg0[[[REGISTER]]]
#blocked = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_by_constancy_full(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<256> : tensor<1024xi32, #blocked>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : (i32) -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg2 : (i32) -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = arith.divsi %4, %cst : tensor<1024xi32, #blocked>
    %8 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<1024x!tt.ptr<f16, 1>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f16, 1>, #blocked>, tensor<1024xi32, #blocked>
    %10 = tt.load %9, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf16, #blocked>
    %11 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<1024x!tt.ptr<f16, 1>, #blocked>
    %12 = tt.addptr %11, %4 : tensor<1024x!tt.ptr<f16, 1>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %12, %10, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf16, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: dedup_by_constancy_partial
// PTX-COUNT-8: llvm.add
// GCN-COUNT-26: llvm.add
// PTX-NOT: llvm.add
// CHECK: llvm.icmp "slt"
// PTX-NOT: llvm.icmp "slt"
// PTX-COUNT-2: llvm.sdiv
// GCN-COUNT-8: llvm.sdiv
// PTX-NOT: llvm.sdiv
// PTX: llvm.getelementptr %arg0[[[REGISTER1:%[0-9]+]]]
// PTX-COUNT-3: llvm.getelementptr %arg0[[[REGISTER1]]]
// GCN-COUNT-16: llvm.getelementptr
// PTX-NOT: llvm.getelementptr %arg0[[[REGISTER1]]]
// PTX: llvm.getelementptr %arg0[[[REGISTER2:%[0-9]+]]]
// PTX-COUNT-3: llvm.getelementptr %arg0[[[REGISTER2]]]
// PTX-NOT: llvm.getelementptr %arg0[[[REGISTER2]]]
#blocked = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @dedup_by_constancy_partial(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<1024xi32, #blocked>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : (i32) -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg2 : (i32) -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    %7 = arith.divsi %4, %cst : tensor<1024xi32, #blocked>
    %8 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<1024x!tt.ptr<f16, 1>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f16, 1>, #blocked>, tensor<1024xi32, #blocked>
    %10 = tt.load %9, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf16, #blocked>
    %11 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<1024x!tt.ptr<f16, 1>, #blocked>
    %12 = tt.addptr %11, %4 : tensor<1024x!tt.ptr<f16, 1>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %12, %10, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf16, #blocked>
    tt.return
  }
}
