// RUN: split-file %s %t
// RUN: triton-opt %t/sm80.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm='compute-capability=80' | FileCheck %t/sm80.mlir
// RUN: triton-opt %t/sm90.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm='compute-capability=90' | FileCheck %t/sm90.mlir

//--- sm80.mlir

// CHECK: llvm.atomicrmw fadd %{{[^, ]+}}, %{{[^ ]+}} acq_rel

#unique = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:80",
                   "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
                          %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                          %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                          %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    %0 = tt.load %arg0 : !tt.ptr<i64>
    %1 = tt.load %arg1 : !tt.ptr<bf16>
    %2 = tt.addptr %arg2, %0 : !tt.ptr<bf16>, i64
    %3 = tt.atomic_rmw fadd, acq_rel, sys, %2, %1, %true {allocation.offset = 0 : i32} : (!tt.ptr<bf16>, bf16, i1) -> bf16
    tt.store %arg3, %3 : !tt.ptr<bf16>
    tt.return
  }
  // CHECK-LABEL: @unmasked_bf16_add
  // CHECK-NOT: llvm.cond_br
  // CHECK: llvm.atomicrmw fadd
  tt.func private @unmasked_bf16_add(%ptr: tensor<128x!tt.ptr<bf16>, #unique>, %val: tensor<128xbf16, #unique>) -> tensor<128xbf16, #unique> {
    %old = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val : (tensor<128x!tt.ptr<bf16>, #unique>, tensor<128xbf16, #unique>) -> tensor<128xbf16, #unique>
    tt.return %old : tensor<128xbf16, #unique>
  }
}

//--- sm90.mlir
// CHECK: atom.global.gpu.acq_rel.add.noftz.bf16

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:90",
                   "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
                          %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                          %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                          %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    %0 = tt.load %arg0 : !tt.ptr<i64>
    %1 = tt.load %arg1 : !tt.ptr<bf16>
    %2 = tt.addptr %arg2, %0 : !tt.ptr<bf16>, i64
    %3 = tt.atomic_rmw fadd, acq_rel, gpu, %2, %1, %true {allocation.offset = 0 : i32} : (!tt.ptr<bf16>, bf16, i1) -> bf16
    tt.store %arg3, %3 : !tt.ptr<bf16>
    tt.return
  }
}
