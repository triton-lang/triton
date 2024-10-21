// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 --mlir-print-debuginfo --mlir-pretty-debuginfo| FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = false}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 544 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @local_load_offset
  tt.func @local_load_offset(%arg0: tensor<16x16xf16, #mma>) {
    %0 = triton_gpu.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #blocked> loc(#loc1)
    %1 = triton_gpu.local_alloc %0 {allocation.offset = 0 : i32} : (tensor<16x16xf16, #blocked>) -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory> loc(#loc2)
    // This catches base ptr calculation in the computeBasePtr, checks if the gep has correct element type.
    // CHECK: llvm.getelementptr {{.*}} (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16 local_load:3:0
    %2 = triton_gpu.local_load %1 : !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> loc(#loc3)
    tt.return
  }
}
#loc1 = loc("conert_layout":1:0)
#loc2 = loc("local_alloc":2:0)
#loc3 = loc("local_load":3:0)
