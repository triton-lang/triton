// RUN: triton-opt %s -split-input-file --allocate-shared-memory --canonicalize --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_r(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant 1.000000e+00 : f32
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = arith.cmpi slt, %1, %c512_i32 : i32

    // CHECK: ld.global.gpu.acquire
    // CHECK: st.global.gpu.release
    %3 = tt.atomic_load %arg0, memSemantic = acquire, memSyncScope = gpu, %2 : !tt.ptr<f32>
    tt.atomic_store %arg0, %3, memSemantic = release, memSyncScope = gpu : !tt.ptr<f32>

    // CHECK: ld.global.cta.acquire
    // CHECK: st.global.cta.release
    %4 = tt.atomic_load %arg0, memSemantic = acquire, memSyncScope = cta : !tt.ptr<f32>
    %5 = arith.addf %4, %cst : f32
    tt.atomic_store %arg0, %5, memSemantic = release, memSyncScope = cta, %2 : !tt.ptr<f32>

    // CHECK: ld.global.sys.acquire
    // CHECK: st.global.sys.release
    %6 = tt.atomic_load %arg0, memSemantic = acquire, memSyncScope = sys, %2 : !tt.ptr<f32>
    %7 = arith.addf %4, %6 : f32
    %8 = arith.addf %7, %cst : f32
    tt.atomic_store %arg0, %8, memSemantic = release, memSyncScope = sys : !tt.ptr<f32>

    // CHECK: ld.global.gpu.relaxed
    // CHECK: st.global.gpu.relaxed
    %9 = tt.atomic_load %arg0, memSemantic = relaxed, memSyncScope = gpu, %2 : !tt.ptr<f32>
    tt.atomic_store %arg0, %9, memSemantic = relaxed, memSyncScope = gpu : !tt.ptr<f32>

    // CHECK: ld.global.cta.relaxed
    // CHECK: st.global.cta.relaxed
    %10 = tt.atomic_load %arg0, memSemantic = relaxed, memSyncScope = cta : !tt.ptr<f32>
    %11 = arith.addf %10, %cst : f32
    tt.atomic_store %arg0, %11, memSemantic = relaxed, memSyncScope = cta : !tt.ptr<f32>

    // CHECK: ld.global.sys.relaxed
    // CHECK: st.global.sys.relaxed
    %12 = tt.atomic_load %arg0, memSemantic = relaxed, memSyncScope = sys, %2 : !tt.ptr<f32>
    %13 = arith.addf %10, %12 : f32
    %14 = arith.addf %13, %cst : f32
    tt.atomic_store %arg0, %14, memSemantic = relaxed, memSyncScope = sys : !tt.ptr<f32>
    tt.return
  }
}
