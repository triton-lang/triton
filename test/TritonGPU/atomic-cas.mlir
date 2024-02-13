// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu 2>&1 | FileCheck %s --check-prefix=GPU
// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu -convert-triton-gpu-to-llvm 2>&1 | FileCheck %s --check-prefix=LLVM

// GPU: %9 = "tt.atomic_cas"(%8, %cst_0, %cst) <{scope = 2 : i32, sem = 4 : i32}> : (tensor<2x!tt.ptr<i64, 1>, #blocked>, tensor<2xi64, #blocked>, tensor<2xi64, #blocked>) -> tensor<2xi64, #blocked>
// LLVM: llvm.inline_asm {{.*}} "mov.u64 $0, 0x0;\0A\09@$4 atom.global.acq_rel.cta.cas.b64 $0, [ $1 + 0 ], $2, $3;", "=l,l,l,l,b"

module {
  tt.func public @atomic_cas_kernel_0d1d2e(%arg0: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<2xi64>
    %cst_0 = arith.constant dense<1> : tensor<2xi64>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c2_i32 : i32
    %2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %3 = tt.splat %1 : i32 -> tensor<2xi32>
    %4 = arith.addi %3, %2 : tensor<2xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<2xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<2xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i64, 1> -> tensor<2x!tt.ptr<i64, 1>>
    %8 = tt.addptr %7, %4 : tensor<2x!tt.ptr<i64, 1>>, tensor<2xi32>
    %9 = "tt.atomic_cas"(%8, %cst_0, %cst) <{sem = 4 : i32, scope = 2 : i32}> : (tensor<2x!tt.ptr<i64, 1>>, tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %10 = tt.splat %arg1 : !tt.ptr<i64, 1> -> tensor<2x!tt.ptr<i64, 1>>
    %11 = tt.addptr %10, %4 : tensor<2x!tt.ptr<i64, 1>>, tensor<2xi32>
    tt.store %11, %9, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<2xi64>
    tt.return
  }
}
