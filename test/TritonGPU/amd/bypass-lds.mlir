// RUN: triton-opt %s -tritonamdgpu-bypass-lds-for-dot-operand -tritonamdgpu-stream-pipeline-v2=num_stages=2 -tritongpu-remove-layout-conversions | FileCheck %s

#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 16], isTransposed = true}>
#dot_layout_0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>
#dot_layout_1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>

// CHECK: %[[DOT_LOAD_1:.+]] = tt.load %{{.*}} : tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
// CHECK: %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %[[DOT_LOAD_1]], %{{.*}} = %{{.*}}) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked>, i32, !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>, tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>)
// CHECK: %[[DOT_LOAD_2:.+]] = tt.load %{{.*}} : tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
// CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[DOT_LOAD_2:.+]], %{{.*}} : tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked>, i32, !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>, tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_0 = arith.constant dense<64> : tensor<64x256xi32, #blocked3>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c256_i32 = arith.constant 256 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c76_i32 = arith.constant 76 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mfma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.remsi %0, %c8_i32 : i32
    %6 = arith.divsi %0, %c8_i32 : i32
    %7 = arith.muli %5, %c76_i32 : i32
    %8 = arith.addi %7, %6 : i32
    %9 = arith.muli %4, %c4_i32 : i32
    %10 = arith.divsi %8, %9 : i32
    %11 = arith.muli %10, %c4_i32 : i32
    %12 = arith.subi %2, %11 : i32
    %13 = arith.minsi %12, %c4_i32 : i32
    %14 = arith.remsi %8, %9 : i32
    %15 = arith.remsi %14, %13 : i32
    %16 = arith.addi %11, %15 : i32
    %17 = arith.divsi %14, %13 : i32
    %18 = arith.muli %16, %c256_i32 : i32
    %19 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %23 = tt.splat %18 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %24 = tt.splat %18 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %25 = arith.addi %23, %19 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %26 = arith.addi %24, %20 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %27 = arith.muli %17, %c256_i32 : i32
    %28 = tt.splat %27 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %29 = tt.splat %27 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %30 = arith.addi %28, %21 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %31 = arith.addi %29, %22 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %32 = tt.expand_dims %25 {axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %33 = tt.expand_dims %26 {axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<256x1xi32, #blocked2>
    %34 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #blocked1>
    %35 = arith.muli %32, %34 : tensor<256x1xi32, #blocked1>
    %36 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %37 = tt.addptr %36, %35 : tensor<256x1x!tt.ptr<f16>, #blocked1>, tensor<256x1xi32, #blocked1>
    %38 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %39 = tt.expand_dims %38 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %40 = tt.broadcast %37 : tensor<256x1x!tt.ptr<f16>, #blocked1> -> tensor<256x64x!tt.ptr<f16>, #blocked1>
    %41 = tt.broadcast %39 : tensor<1x64xi32, #blocked1> -> tensor<256x64xi32, #blocked1>
    %42 = tt.addptr %40, %41 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %44 = tt.expand_dims %43 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %45 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked3>
    %46 = tt.addptr %45, %44 : tensor<64x1x!tt.ptr<f16>, #blocked3>, tensor<64x1xi32, #blocked3>
    %47 = tt.expand_dims %30 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
    %48 = tt.expand_dims %31 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
    %49 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked3>
    %50 = arith.muli %47, %49 : tensor<1x256xi32, #blocked3>
    %51 = tt.broadcast %46 : tensor<64x1x!tt.ptr<f16>, #blocked3> -> tensor<64x256x!tt.ptr<f16>, #blocked3>
    %52 = tt.broadcast %50 : tensor<1x256xi32, #blocked3> -> tensor<64x256xi32, #blocked3>
    %53 = tt.addptr %51, %52 : tensor<64x256x!tt.ptr<f16>, #blocked3>, tensor<64x256xi32, #blocked3>
    %54 = arith.addi %arg5, %c63_i32 : i32
    %55 = arith.divsi %54, %c64_i32 : i32
    %56:3 = scf.for %arg10 = %c0_i32 to %55 step %c1_i32 iter_args(%arg11 = %cst_1, %arg12 = %42, %arg13 = %53) -> (tensor<256x256xf32, #mfma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked3>)  : i32 {
      %74 = tt.load %arg12 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %75 = tt.load %arg13 : tensor<64x256x!tt.ptr<f16>, #blocked3>
      %76 = triton_gpu.convert_layout %74 : tensor<256x64xf16, #blocked1> -> tensor<256x64xf16, #dot_layout_0>
      %77 = triton_gpu.convert_layout %75 : tensor<64x256xf16, #blocked3> -> tensor<64x256xf16, #dot_layout_1>
      %78 = tt.dot %76, %77, %arg11 : tensor<256x64xf16, #dot_layout_0> * tensor<64x256xf16, #dot_layout_1> -> tensor<256x256xf32, #mfma>
      %79 = tt.addptr %arg12, %cst : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %80 = tt.addptr %arg13, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked3>, tensor<64x256xi32, #blocked3>
      scf.yield %78, %79, %80 : tensor<256x256xf32, #mfma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked3>
    }
    %57 = arith.truncf %56#0 : tensor<256x256xf32, #mfma> to tensor<256x256xf16, #mfma>
    %58 = tt.splat %arg8 : i32 -> tensor<256x1xi32, #blocked2>
    %59 = arith.muli %58, %33 : tensor<256x1xi32, #blocked2>
    %60 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked2>
    %61 = tt.addptr %60, %59 : tensor<256x1x!tt.ptr<f16>, #blocked2>, tensor<256x1xi32, #blocked2>
    %62 = tt.broadcast %61 : tensor<256x1x!tt.ptr<f16>, #blocked2> -> tensor<256x256x!tt.ptr<f16>, #blocked2>
    %63 = tt.broadcast %48 : tensor<1x256xi32, #blocked2> -> tensor<256x256xi32, #blocked2>
    %64 = tt.addptr %62, %63 : tensor<256x256x!tt.ptr<f16>, #blocked2>, tensor<256x256xi32, #blocked2>
    %65 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #blocked2>
    %66 = arith.cmpi slt, %33, %65 : tensor<256x1xi32, #blocked2>
    %67 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
    %68 = arith.cmpi slt, %48, %67 : tensor<1x256xi32, #blocked2>
    %69 = tt.broadcast %66 : tensor<256x1xi1, #blocked2> -> tensor<256x256xi1, #blocked2>
    %70 = tt.broadcast %68 : tensor<1x256xi1, #blocked2> -> tensor<256x256xi1, #blocked2>
    %71 = arith.andi %69, %70 : tensor<256x256xi1, #blocked2>
    %72 = triton_gpu.convert_layout %64 : tensor<256x256x!tt.ptr<f16>, #blocked2> -> tensor<256x256x!tt.ptr<f16>, #mfma>
    %73 = triton_gpu.convert_layout %71 : tensor<256x256xi1, #blocked2> -> tensor<256x256xi1, #mfma>
    tt.store %72, %57, %73 : tensor<256x256x!tt.ptr<f16>, #mfma>
    tt.return
  }
}
