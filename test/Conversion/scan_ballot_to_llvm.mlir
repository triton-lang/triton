// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

#l32    = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#l32x2  = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#l32x2t = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#l64x2  = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @bool_extui
// CHECK: %[[BALLOT:.*]] = nvvm.vote.sync ballot
// CHECK: %[[MASKED:.*]] = llvm.and %[[BALLOT]],
// CHECK: llvm.intr.ctpop(%[[MASKED]])
tt.func private @bool_extui(%arg0: tensor<32xi1, #l32>) -> tensor<32xi32, #l32> {
  %b = arith.extui %arg0 : tensor<32xi1, #l32> to tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// CHECK-LABEL: @bool_select_and
// CHECK: llvm.intr.ctpop
tt.func private @bool_select_and(%mask: tensor<32xi1, #l32>, %x: tensor<32xi32, #l32>) -> tensor<32xi32, #l32> {
  %c0 = arith.constant dense<0> : tensor<32xi32, #l32>
  %c1 = arith.constant dense<1> : tensor<32xi32, #l32>
  %t = arith.andi %x, %c1 : tensor<32xi32, #l32>
  %b = arith.select %mask, %t, %c0 : tensor<32xi1, #l32>, tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// width < 32: the i32 popcount is truncated to the scan's element type.
// CHECK-LABEL: @bool_i1
// CHECK: %[[CNT:.*]] = llvm.intr.ctpop
// CHECK: llvm.trunc %[[CNT]] : i32 to i1
tt.func private @bool_i1(%arg0: tensor<32xi1, #l32>) -> tensor<32xi1, #l32> {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i1, %c: i1):
    %1 = arith.addi %a, %c : i1
    tt.scan.return %1 : i1
  }) : (tensor<32xi1, #l32>) -> tensor<32xi1, #l32>
  tt.return %0 : tensor<32xi1, #l32>
}

// width > 32: the i32 popcount is zero-extended to the scan's element type.
// CHECK-LABEL: @bool_extui_i64
// CHECK: %[[CNT:.*]] = llvm.intr.ctpop
// CHECK: llvm.zext %[[CNT]] : i32 to i64
tt.func private @bool_extui_i64(%arg0: tensor<32xi1, #l32>) -> tensor<32xi64, #l32> {
  %b = arith.extui %arg0 : tensor<32xi1, #l32> to tensor<32xi64, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i64, %c: i64):
    %1 = arith.addi %a, %c : i64
    tt.scan.return %1 : i64
  }) : (tensor<32xi64, #l32>) -> tensor<32xi64, #l32>
  tt.return %0 : tensor<32xi64, #l32>
}

// 2d scan with parallel elements per threads. The {0, 1} prover sees through convert_layout.
// CHECK-LABEL: @convert_layout_2d
// CHECK: llvm.intr.ctpop
// CHECK: llvm.intr.ctpop
tt.func private @convert_layout_2d(%arg0: tensor<32x2xi1, #l32x2t>) -> tensor<32x2xi32, #l32x2> {
  %e = arith.extui %arg0 : tensor<32x2xi1, #l32x2t> to tensor<32x2xi32, #l32x2t>
  %cl = ttg.convert_layout %e : tensor<32x2xi32, #l32x2t> -> tensor<32x2xi32, #l32x2>
  %0 = "tt.scan"(%cl) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32x2xi32, #l32x2>) -> tensor<32x2xi32, #l32x2>
  tt.return %0 : tensor<32x2xi32, #l32x2>
}

// The fast path should not fire with more than 1 contiguous element per thread.
// CHECK-LABEL: @not_gate_layout
// CHECK-NOT: llvm.intr.ctpop
tt.func private @not_gate_layout(%arg0: tensor<64xi1, #l64x2>) -> tensor<64xi32, #l64x2> {
  %b = arith.extui %arg0 : tensor<64xi1, #l64x2> to tensor<64xi32, #l64x2>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<64xi32, #l64x2>) -> tensor<64xi32, #l64x2>
  tt.return %0 : tensor<64xi32, #l64x2>
}

}
