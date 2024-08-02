// RUN: triton-opt --split-input-file %s | FileCheck %s

// CHECK: #[[$WMMA_GEN1:.*]] = #triton_gpu.amd_wmma<{{.*}}version = 1{{.*}}>
// CHECK: #[[$WMMA_GEN2:.*]] = #triton_gpu.amd_wmma<{{.*}}version = 2{{.*}}>
#blocked = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>

module attributes {"triton_gpu.target" = "cuda:0", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_layout
  tt.func @wmma_layout(%0: tensor<16x16xf16, #blocked>) {
    %1 = triton_gpu.convert_layout %0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #triton_gpu.amd_wmma<{version = 1, warpsPerCTA = [1, 1]}>>
    // CHECK:  %{{.+}} = triton_gpu.convert_layout %{{.+}} : tensor<16x16xf16, #{{.+}}> -> tensor<16x16xf16, #[[$WMMA_GEN1]]>
    tt.return
  }

  // CHECK-LABEL: wmma_dot_op_layout
  tt.func @wmma_dot_op_layout(%0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) {
    %1 = triton_gpu.convert_layout %0 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.amd_wmma<{version = 1, warpsPerCTA = [1, 1]}>, kWidth = 16}>>
    // CHECK:  %{{.+}} = triton_gpu.convert_layout %{{.+}} : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #{{.+}}}>> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[$WMMA_GEN1]], kWidth = 16}>>
    tt.return
  }

  // CHECK-LABEL: wmma_gen2_layout
  tt.func @wmma_gen2_layout(%0: tensor<16x16xf16, #blocked>) {
    %1 = triton_gpu.convert_layout %0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #triton_gpu.amd_wmma<{version = 2, warpsPerCTA = [1, 1]}>>
    // CHECK:  %{{.+}} = triton_gpu.convert_layout %{{.+}} : tensor<16x16xf16, #{{.+}}> -> tensor<16x16xf16, #[[$WMMA_GEN2]]>
    tt.return
  }

  // CHECK-LABEL: wmma_gen2_dot_op_layout
  tt.func @wmma_gen2_dot_op_layout(%0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) {
    %1 = triton_gpu.convert_layout %0 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.amd_wmma<{version = 2, warpsPerCTA = [1, 1]}>, kWidth = 8}>>
    // CHECK:  %{{.+}} = triton_gpu.convert_layout %{{.+}} : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #{{.+}}}>> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[$WMMA_GEN2]], kWidth = 8}>>
    tt.return
  }
}
