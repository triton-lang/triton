// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_r(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = arith.cmpi slt, %1, %c512_i32 : i32

    // CHECK: ld.global.gpu.acquire
    %3 = tt.atomic_rmw fadd, acquire, gpu, %arg0, %cst, %2 : (!tt.ptr<f32>, f32, i1) -> f32
    %4 = tt.atomic_rmw exch, release, gpu, %arg0, %cst, %true : (!tt.ptr<f32>, f32, i1) -> f32

    // CHECK: ld.global.cta.acquire
    %5 = tt.atomic_rmw fadd, acquire, cta, %arg0, %cst, %true : (!tt.ptr<f32>, f32, i1) -> f32
    %6 = tt.atomic_rmw exch, release, cta, %arg0, %cst, %2 : (!tt.ptr<f32>, f32, i1) -> f32

    // CHECK: ld.global.sys.acquire
    %7 = tt.atomic_rmw fadd, acquire, sys, %arg0, %cst, %2 : (!tt.ptr<f32>, f32, i1) -> f32
    %8 = tt.atomic_rmw exch, release, sys, %arg0, %cst, %true : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}
