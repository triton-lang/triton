// RUN: triton-opt -split-input-file -convert-triton-gpu-to-llvm='compute-capability=90' %s 2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @clamp_kernel_0d1d2d3d4de(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/root/code/sandbox/misc/triton_kernel.py":144:0), %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/root/code/sandbox/misc/triton_kernel.py":144:0), %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/root/code/sandbox/misc/triton_kernel.py":144:0), %arg3: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/root/code/sandbox/misc/triton_kernel.py":144:0), %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32} loc("/root/code/sandbox/misc/triton_kernel.py":144:0)) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %1 = tt.splat %arg4 : (i32) -> tensor<1024xi32, #blocked>
    %2 = arith.cmpi slt, %0, %1 : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %4 = tt.addptr %3, %0 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
    %5 = tt.load %4, %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
    %6 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
    %8 = tt.load %7, %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
    %9 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %10 = tt.addptr %9, %0 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
    %11 = arith.subf %cst, %8 : tensor<1024xf32, #blocked>

    // CHECK: %{{[a-zA-Z0-9]+}} = llvm.inline_asm asm_dialect = att operand_attrs = [] "min.xorsign.abs.f32 $0, $1, $2;", "=f,f,f" %{{[a-zA-Z0-9]+}}, %{{[a-zA-Z0-9]+}} : (f32, f32) -> f32
    %12 = tt.clampf %5, %11, %8 {propagateNan = 0 : i32} -> tensor<1024xf32, #blocked>
    tt.store %10, %12, %2 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32, #blocked>
    tt.return
  }
}
