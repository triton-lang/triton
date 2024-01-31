// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul=arch-generation-name=gfx90a | FileCheck --check-prefixes=CHECK %s

// CHECK: #mfma
// CHECK-SAME: isTransposed = true

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_0d1d2d3d(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_chain_dot.py":14:0), %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_chain_dot.py":14:0), %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_chain_dot.py":14:0), %arg3: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_chain_dot.py":14:0)) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked1>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %5 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %6 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %7 = tt.expand_dims %3 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %8 = arith.muli %4, %cst : tensor<32x1xi32, #blocked>
    %9 = arith.muli %5, %cst : tensor<32x1xi32, #blocked>
    %10 = arith.muli %6, %cst : tensor<32x1xi32, #blocked>
    %11 = arith.muli %7, %cst : tensor<32x1xi32, #blocked>
    %12 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked>
    %13 = tt.addptr %12, %8 : tensor<32x1x!tt.ptr<f32, 1>, #blocked>, tensor<32x1xi32, #blocked>
    %14 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %18 = tt.expand_dims %14 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %19 = tt.expand_dims %15 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %20 = tt.expand_dims %16 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %21 = tt.expand_dims %17 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %22 = tt.broadcast %13 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %23 = tt.broadcast %18 : (tensor<1x32xi32, #blocked>) -> tensor<32x32xi32, #blocked>
    %24 = tt.broadcast %19 : (tensor<1x32xi32, #blocked>) -> tensor<32x32xi32, #blocked>
    %25 = tt.broadcast %20 : (tensor<1x32xi32, #blocked>) -> tensor<32x32xi32, #blocked>
    %26 = tt.broadcast %21 : (tensor<1x32xi32, #blocked>) -> tensor<32x32xi32, #blocked>
    %27 = tt.addptr %22, %23 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
    %28 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked>
    %29 = tt.addptr %28, %9 : tensor<32x1x!tt.ptr<f32, 1>, #blocked>, tensor<32x1xi32, #blocked>
    %30 = tt.broadcast %29 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %31 = tt.addptr %30, %24 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
    %32 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked>
    %33 = tt.addptr %32, %10 : tensor<32x1x!tt.ptr<f32, 1>, #blocked>, tensor<32x1xi32, #blocked>
    %34 = tt.broadcast %33 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %35 = tt.addptr %34, %25 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
    %36 = tt.load %27 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
    %37 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
    %38 = tt.load %35 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
    %39 = triton_gpu.convert_layout %36 : (tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
    %40 = triton_gpu.convert_layout %37 : (tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>
    %41 = tt.dot %39, %40, %cst_0 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<32x32xf32, #blocked1>
    %42 = triton_gpu.convert_layout %41 : (tensor<32x32xf32, #blocked1>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
    %43 = triton_gpu.convert_layout %38 : (tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>
    %44 = tt.dot %42, %43, %cst_0 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<32x32xf32, #blocked1>
    %45 = tt.splat %arg3 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked>
    %46 = tt.addptr %45, %11 : tensor<32x1x!tt.ptr<f32, 1>, #blocked>, tensor<32x1xi32, #blocked>
    %47 = tt.broadcast %46 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %48 = tt.addptr %47, %26 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
    %49 = triton_gpu.convert_layout %44 : (tensor<32x32xf32, #blocked1>) -> tensor<32x32xf32, #blocked>
    tt.store %48, %49 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf32, #blocked>
    tt.return
  }
}


// CHECK: #mfma1
// CHECK-SAME: isTransposed = false

#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_0d1d2d3d(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_chain_dot.py":14:0), %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_chain_dot.py":14:0), %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_chain_dot.py":14:0), %arg3: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_chain_dot.py":14:0)) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x1xi32, #blocked2> 
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked3>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xi32, #blocked2>
    %5 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xi32, #blocked2>
    %6 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xi32, #blocked2>
    %7 = tt.expand_dims %3 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xi32, #blocked2>
    %8 = arith.muli %4, %cst : tensor<32x1xi32, #blocked2>
    %9 = arith.muli %5, %cst : tensor<32x1xi32, #blocked2>
    %10 = arith.muli %6, %cst : tensor<32x1xi32, #blocked2>
    %11 = arith.muli %7, %cst : tensor<32x1xi32, #blocked2>
    %12 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked2>
    %13 = tt.addptr %12, %8 : tensor<32x1x!tt.ptr<f32, 1>, #blocked2>, tensor<32x1xi32, #blocked2>
    %14 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %18 = tt.expand_dims %14 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %19 = tt.expand_dims %15 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %20 = tt.expand_dims %16 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %21 = tt.expand_dims %17 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %22 = tt.broadcast %13 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked2>
    %23 = tt.broadcast %18 : (tensor<1x32xi32, #blocked2>) -> tensor<32x32xi32, #blocked2>
    %24 = tt.broadcast %19 : (tensor<1x32xi32, #blocked2>) -> tensor<32x32xi32, #blocked2>
    %25 = tt.broadcast %20 : (tensor<1x32xi32, #blocked2>) -> tensor<32x32xi32, #blocked2>
    %26 = tt.broadcast %21 : (tensor<1x32xi32, #blocked2>) -> tensor<32x32xi32, #blocked2>
    %27 = tt.addptr %22, %23 : tensor<32x32x!tt.ptr<f32, 1>, #blocked2>, tensor<32x32xi32, #blocked2>
    %28 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked2>
    %29 = tt.addptr %28, %9 : tensor<32x1x!tt.ptr<f32, 1>, #blocked2>, tensor<32x1xi32, #blocked2>
    %30 = tt.broadcast %29 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked2>
    %31 = tt.addptr %30, %24 : tensor<32x32x!tt.ptr<f32, 1>, #blocked2>, tensor<32x32xi32, #blocked2>
    %32 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked2>
    %33 = tt.addptr %32, %10 : tensor<32x1x!tt.ptr<f32, 1>, #blocked2>, tensor<32x1xi32, #blocked2>
    %34 = tt.broadcast %33 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked2>
    %35 = tt.addptr %34, %25 : tensor<32x32x!tt.ptr<f32, 1>, #blocked2>, tensor<32x32xi32, #blocked2>
    %36 = tt.load %27 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked2>
    %37 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked2>
    %38 = tt.load %35 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked2>
    %39 = triton_gpu.convert_layout %36 : (tensor<32x32xf32, #blocked2>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked3}>>
    %40 = triton_gpu.convert_layout %37 : (tensor<32x32xf32, #blocked2>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked3}>>
    %41 = tt.dot %39, %40, %cst_0 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked3}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<32x32xf32, #blocked3>
    %42 = triton_gpu.convert_layout %38 : (tensor<32x32xf32, #blocked2>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked3}>>
    %43 = triton_gpu.convert_layout %41 : (tensor<32x32xf32, #blocked3>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked3}>>
    %44 = tt.dot %42, %43, %cst_0 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked3}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<32x32xf32, #blocked3>
    %45 = tt.splat %arg3 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked2>
    %46 = tt.addptr %45, %11 : tensor<32x1x!tt.ptr<f32, 1>, #blocked2>, tensor<32x1xi32, #blocked2>
    %47 = tt.broadcast %46 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked2>
    %48 = tt.addptr %47, %26 : tensor<32x32x!tt.ptr<f32, 1>, #blocked2>, tensor<32x32xi32, #blocked2>
    %49 = triton_gpu.convert_layout %44 : (tensor<32x32xf32, #blocked3>) -> tensor<32x32xf32, #blocked2>
    tt.store %48, %49 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf32, #blocked2>
    tt.return
  }
}

// CHECK: #mfma2
// CHECK-SAME: isTransposed = false

#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked6 = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_0d1d2d(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_dot_reduce.py":14:0), %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_dot_reduce.py":14:0), %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/workspace/projects/triton/scripts/amd/gemm/test_dot_reduce.py":14:0)) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x1xi32, #blocked4> 
    %cst_0 = arith.constant dense<64> : tensor<32x1xi32, #blocked4>
    %cst_1 = arith.constant dense<32> : tensor<16x1xi32, #blocked5>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked6>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked5}>>) -> tensor<16x1xi32, #blocked5>
    %3 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<16x1xi32, #blocked4>
    %4 = arith.muli %2, %cst_1 : tensor<16x1xi32, #blocked5>
    %5 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<16x1x!tt.ptr<f32, 1>, #blocked5>
    %6 = tt.addptr %5, %4 : tensor<16x1x!tt.ptr<f32, 1>, #blocked5>, tensor<16x1xi32, #blocked5>
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked5}>>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked5}>>) -> tensor<1x32xi32, #blocked5>
    %9 = tt.broadcast %6 : (tensor<16x1x!tt.ptr<f32, 1>, #blocked5>) -> tensor<16x32x!tt.ptr<f32, 1>, #blocked5>
    %10 = tt.broadcast %8 : (tensor<1x32xi32, #blocked5>) -> tensor<16x32xi32, #blocked5>
    %11 = tt.addptr %9, %10 : tensor<16x32x!tt.ptr<f32, 1>, #blocked5>, tensor<16x32xi32, #blocked5>
    %12 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<32x1xi32, #blocked4>
    %14 = arith.muli %13, %cst_0 : tensor<32x1xi32, #blocked4>
    %15 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked4>
    %16 = tt.addptr %15, %14 : tensor<32x1x!tt.ptr<f32, 1>, #blocked4>, tensor<32x1xi32, #blocked4>
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %19 = tt.expand_dims %17 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi32, #blocked4>
    %20 = tt.expand_dims %18 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi32, #blocked4>
    %21 = tt.broadcast %16 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked4>) -> tensor<32x64x!tt.ptr<f32, 1>, #blocked4>
    %22 = tt.broadcast %19 : (tensor<1x64xi32, #blocked4>) -> tensor<32x64xi32, #blocked4>
    %23 = tt.addptr %21, %22 : tensor<32x64x!tt.ptr<f32, 1>, #blocked4>, tensor<32x64xi32, #blocked4>
    %24 = tt.load %11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x32xf32, #blocked5>
    %25 = tt.load %23 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf32, #blocked4>
    %26 = triton_gpu.convert_layout %24 : (tensor<16x32xf32, #blocked5>) -> tensor<16x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked6}>>
    %27 = triton_gpu.convert_layout %25 : (tensor<32x64xf32, #blocked4>) -> tensor<32x64xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked6}>>
    %28 = tt.dot %26, %27, %cst_2 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked6}>> * tensor<32x64xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked6}>> -> tensor<16x64xf32, #blocked6>
    %29 = "tt.reduce"(%28) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %42 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %42 : f32
    }) : (tensor<16x64xf32, #blocked6>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>
    %30 = tt.expand_dims %29 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked6}>>) -> tensor<16x1xf32, #blocked6>
    %31 = tt.broadcast %30 : (tensor<16x1xf32, #blocked6>) -> tensor<16x64xf32, #blocked6>
    %32 = triton_gpu.convert_layout %24 : (tensor<16x32xf32, #blocked5>) -> tensor<16x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked6}>>
    %33 = triton_gpu.convert_layout %25 : (tensor<32x64xf32, #blocked4>) -> tensor<32x64xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked6}>>
    %34 = tt.dot %32, %33, %31 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked6}>> * tensor<32x64xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked6}>> -> tensor<16x64xf32, #blocked6>
    %35 = arith.muli %3, %cst : tensor<16x1xi32, #blocked4> 
    %36 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<16x1x!tt.ptr<f32, 1>, #blocked4>
    %37 = tt.addptr %36, %35 : tensor<16x1x!tt.ptr<f32, 1>, #blocked4>, tensor<16x1xi32, #blocked4>
    %38 = tt.broadcast %37 : (tensor<16x1x!tt.ptr<f32, 1>, #blocked4>) -> tensor<16x64x!tt.ptr<f32, 1>, #blocked4>
    %39 = tt.broadcast %20 : (tensor<1x64xi32, #blocked4>) -> tensor<16x64xi32, #blocked4>
    %40 = tt.addptr %38, %39 : tensor<16x64x!tt.ptr<f32, 1>, #blocked4>, tensor<16x64xi32, #blocked4>
    %41 = triton_gpu.convert_layout %34 : (tensor<16x64xf32, #blocked6>) -> tensor<16x64xf32, #blocked4>
    tt.store %40, %41 {cache = 1 : i32, evict = 1 : i32} : tensor<16x64xf32, #blocked4>
    tt.return
  }
}
