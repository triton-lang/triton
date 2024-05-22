// RUN: triton-opt %s -split-input-file --tritonamdgpu-stream-pipeline | FileCheck %s

// CHECK-LABEL: @check_stream_pipeline_epilogue
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @check_stream_pipeline_epilogue(%Aptr: tensor<32x32x!tt.ptr<f32>, #blocked>, %Bptr : tensor<32x32x!tt.ptr<f32>, #blocked>, %arg4 : i32, %arg5 : i1) {
    %cst_0 = arith.constant dense<16> : tensor<32x32xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for {{.*}} = %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] iter_args({{.*}})
    %36:3 = scf.for %arg9 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg10 = %cst_5, %arg12 = %Aptr, %arg13 = %Bptr) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>) : i32 {
      %61 = arith.muli %arg9, %arg4 : i32
      %62 = arith.cmpi slt, %arg4, %61 : i32
      %63 = tt.splat %62 : i1 -> tensor<32x32xi1, #blocked>
      // This load will not be pipelined
      %66 = tt.load %arg12, %63 : tensor<32x32x!tt.ptr<f32>, #blocked>
      // This load will be pipelined
      %70 = tt.load %arg13 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %71 = triton_gpu.convert_layout %66 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = triton_gpu.convert_layout %70 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %73 = tt.dot %71, %72, %arg10 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      // This scf.if will make load at %66 non-pipelineable
      %74 = scf.if %arg5 -> (tensor<32x32xf32, #blocked>){
          scf.yield %66 : tensor<32x32xf32, #blocked>
      } else {
        scf.yield %cst_2: tensor<32x32xf32, #blocked>
      }
      %75 = tt.addptr %arg12, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %76 = tt.addptr %arg13, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %73, %75, %76 : tensor<32x32xf32, #mma>,  tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>
    }
    // CHECK: %[[C1:.*]] = arith.constant 1 : i32
    // CHECK: %[[t0:.*]] = arith.subi %[[UB:.*]], %[[C1]]
    // CHECK: %[[t1:.*]] = arith.subi %[[t0]], %[[LB]]
    // CHECK: %[[t2:.*]] = arith.divui %[[t1]], %[[STEP]]
    // CHECK: %[[t3:.*]] = arith.muli %[[t2]], %[[STEP]]
    // CHECK: %[[PPLUB:.*]] = arith.addi %[[LB]], %[[t3]]
    // CHECK: arith.muli %[[PPLUB]], {{.*}}
    tt.return
  }
}
