// RUN: triton-opt %s -split-input-file --tritonamdgpu-stream-pipeline | FileCheck %s

#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @matmul_kernel
  // CHECK-SAME: (%[[Aptr:.*]]: {{.*}}, %[[Bptr:.*]]: {{.*}}, %[[vecIdxA:.*]]: {{.*}}, %[[vecIdxB:.*]]: {{.*}}, %[[arg4:.*]]: i32)
  // tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32) attributes {noinline = false} {
  tt.func public @matmul_kernel(%Aptr: tensor<128x16x!tt.ptr<f32>, #blocked1>,
                               %Bptr : tensor<16x256x!tt.ptr<f32>, #blocked3>,
                               %vecIdxA : tensor<16x1xi32, #blocked3>,
                               %vecIdxB : tensor<1x16xi32, #blocked1>,
                               %arg4 : i32) {

    %cst_0 = arith.constant dense<16> : tensor<128x16xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #blocked3>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32

    // CHECK: %[[C1:.*]] = arith.constant 1 : i32
    // CHECK: %[[C15:.*]] = arith.constant 15 : i32
    // CHECK: %[[C16:.*]] = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32

    %31 = arith.addi %arg4, %c15_i32 : i32
    %32 = arith.divsi %31, %c16_i32 : i32
    %33 = arith.cmpi eq, %0, %c0_i32 : i32
    %35 = tt.splat %0 : i32 -> tensor<16x256xi32, #blocked3>
    // CHECK: %[[T0:.*]] = arith.addi %[[arg4]], %[[C15]]
    // CHECK: %[[UB:.*]] = arith.divsi %[[T0]], %[[C16]]
    // CHECK: %[[UB1:.*]] = arith.subi %[[UB]], %[[C1]] : i32
    %36:3 = scf.for %arg9 = %c0_i32 to %32 step %c1_i32 iter_args(%arg10 = %cst_5,
                                                                  %arg12 = %Aptr,
                                                                  %arg13 = %Bptr) -> (tensor<128x256xf32, #mma>,
                                                                                      tensor<128x16x!tt.ptr<f32>, #blocked1>,
                                                                                      tensor<16x256x!tt.ptr<f32>, #blocked3>)  : i32 {
      %61 = arith.muli %arg9, %c16_i32 : i32
      %62 = arith.subi %arg4, %61 : i32
      %63 = tt.splat %62 : i32 -> tensor<1x16xi32, #blocked1>
      %64 = arith.cmpi slt, %vecIdxB, %63 : tensor<1x16xi32, #blocked1>
      %65 = tt.broadcast %64 : tensor<1x16xi1, #blocked1> -> tensor<128x16xi1, #blocked1>
      %66 = tt.load %arg12, %65, %cst_2 : tensor<128x16x!tt.ptr<f32>, #blocked1>
      %67 = tt.splat %62 : i32 -> tensor<16x1xi32, #blocked3>
      %68 = arith.cmpi slt, %vecIdxA, %67 : tensor<16x1xi32, #blocked3>
      %69 = tt.broadcast %68 : tensor<16x1xi1, #blocked3> -> tensor<16x256xi1, #blocked3>
      %70 = tt.load %arg13, %69, %cst_3 : tensor<16x256x!tt.ptr<f32>, #blocked3>
      %71 = triton_gpu.convert_layout %66 : tensor<128x16xf32, #blocked1> -> tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = triton_gpu.convert_layout %70 : tensor<16x256xf32, #blocked3> -> tensor<16x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %73 = tt.dot %71, %72, %arg10 : tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x256xf32, #mma>
      // This scf.if will make load at %66 non-pipelineable
      %74 = scf.if %33 -> (tensor<128x16xf32, #blocked1>){
        scf.yield %66 : tensor<128x16xf32, #blocked1>
      } else {
        scf.yield %cst_2: tensor<128x16xf32, #blocked1>
      }
      %75 = tt.addptr %arg12, %cst_0 : tensor<128x16x!tt.ptr<f32>, #blocked1>, tensor<128x16xi32, #blocked1>
      %76 = tt.addptr %arg13, %35 : tensor<16x256x!tt.ptr<f32>, #blocked3>, tensor<16x256xi32, #blocked3>
      scf.yield %73, %75, %76 : tensor<128x256xf32, #mma>,  tensor<128x16x!tt.ptr<f32>, #blocked1>, tensor<16x256x!tt.ptr<f32>, #blocked3>
    }
    // CHECK: arith.muli %[[UB1]], %[[C16]]
    tt.return
  }
}
