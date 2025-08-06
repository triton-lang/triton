// RUN: triton-opt %s -convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

// CHECK: llvm.inline_asm {{.*}} "mov.u64 $0, 0x0;\0A\09@$4 atom.global.acq_rel.cta.cas.b64 $0, [ $1 + 0 ], $2, $3;", "=l,l,l,l,b"
// CHECK: st.shared
// CHECK: nvvm.barrier0
// CHECK: llvm.load

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_cas_kernel_0d1d2e(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<2> : tensor<2xi64, #blocked>
    %cst_0 = arith.constant dense<1> : tensor<2xi64, #blocked>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c2_i32 : i32
    %2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<2xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<2xi32, #blocked>
    %5 = tt.splat %arg2 : i32 -> tensor<2xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<2xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<2x!tt.ptr<i64>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<2x!tt.ptr<i64>, #blocked>, tensor<2xi32, #blocked>
    %9 = tt.atomic_cas acq_rel, cta, %8, %cst_0, %cst {allocation.offset = 0 : i32} : (tensor<2x!tt.ptr<i64>, #blocked>, tensor<2xi64, #blocked>, tensor<2xi64, #blocked>) -> tensor<2xi64, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<2x!tt.ptr<i64>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<2x!tt.ptr<i64>, #blocked>, tensor<2xi32, #blocked>
    tt.store %11, %9, %6 : tensor<2x!tt.ptr<i64>, #blocked>
    tt.return
  }
}
