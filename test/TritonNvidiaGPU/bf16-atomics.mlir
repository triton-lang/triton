// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s

// CHECK: llvm.atomicrmw fadd

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:80",
                   "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
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


// CHECK: atom.global.gpu.acq_rel.add.noftz.bf16

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:90",
                   "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
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
