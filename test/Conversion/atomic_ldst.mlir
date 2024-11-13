// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_r(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %cst_0 = arith.constant dense<512> : tensor<128x1xi32, #blocked>
    %cst_1 = arith.constant dense<64> : tensor<128x1xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %4 = tt.splat %1 : i32 -> tensor<128x1xi32, #blocked>
    %5 = arith.addi %4, %3 : tensor<128x1xi32, #blocked>
    %6 = arith.cmpi slt, %5, %cst_0 : tensor<128x1xi32, #blocked>
    %7 = arith.remsi %5, %cst_1 : tensor<128x1xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>

    // CHECK: ld.global.cta.relaxed.b32
    %10 = tt.load %9, %6 memSemantic = relaxed memSyncScope = cta : tensor<128x1x!tt.ptr<f32>, #blocked>

    // CHECK: st.global.gpu.release.b32
    tt.store %9, %10 memSemantic = release memSyncScope = gpu : tensor<128x1x!tt.ptr<f32>, #blocked>

    // CHECK: ld.global.cta.relaxed.b32
    %11 = tt.load %9 memSemantic = relaxed memSyncScope = cta : tensor<128x1x!tt.ptr<f32>, #blocked>
    %12 = arith.addf %11, %cst : tensor<128x1xf32, #blocked>

    // CHECK: st.global.gpu.release.b32
    tt.store %9, %12 memSemantic = release memSyncScope = gpu : tensor<128x1x!tt.ptr<f32>, #blocked>

    // CHECK: ld.global.gpu.relaxed.b32
    %13 = tt.load %9, %6 memSemantic = relaxed memSyncScope = gpu : tensor<128x1x!tt.ptr<f32>, #blocked>

    %14 = arith.addf %11, %13 : tensor<128x1xf32, #blocked>
    %15 = arith.addf %14, %cst : tensor<128x1xf32, #blocked>

    // CHECK: st.global.sys.relaxed.b32
    tt.store %9, %15 memSemantic = relaxed memSyncScope = sys : tensor<128x1x!tt.ptr<f32>, #blocked>

    // CHECK: ld.global.gpu.acquire.b32
    %16 = tt.load %9 memSemantic = acquire memSyncScope = gpu : tensor<128x1x!tt.ptr<f32>, #blocked>

    %17 = arith.addf %14, %16 : tensor<128x1xf32, #blocked>
    %18 = arith.addf %17, %cst : tensor<128x1xf32, #blocked>

    // CHECK: st.global.sys.relaxed.b32
    tt.store %9, %18 memSemantic = relaxed memSyncScope = sys : tensor<128x1x!tt.ptr<f32>, #blocked>

    // CHECK: ld.global.sys.acquire.b32
    %19 = tt.load %9 memSemantic = acquire memSyncScope = sys : tensor<128x1x!tt.ptr<f32>, #blocked>

    %20 = arith.addf %17, %19 : tensor<128x1xf32, #blocked>
    %21 = arith.addf %20, %cst : tensor<128x1xf32, #blocked>

    // CHECK: st.global.cta.relaxed.b32
    tt.store %9, %21 memSemantic = relaxed memSyncScope = cta : tensor<128x1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
