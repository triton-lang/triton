// RUN: triton-opt -split-input-file -convert-triton-gpu-to-llvm='compute-capability=90' %s 2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @clamp(%x : tensor<1024xf32, #blocked>, %limit : tensor<1024xf32, #blocked>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    %neg_limit = arith.subf %cst, %limit : tensor<1024xf32, #blocked>

    // CHECK: %{{[a-zA-Z0-9]+}} = llvm.inline_asm asm_dialect = att operand_attrs = [] "min.xorsign.abs.f32 $0, $1, $2;", "=f,f,f" %{{[a-zA-Z0-9]+}}, %{{[a-zA-Z0-9]+}} : (f32, f32) -> f32
    %12 = tt.clampf %x, %neg_limit, %limit {propagateNan = 0 : i32} -> tensor<1024xf32, #blocked>
    tt.return
  }
}
