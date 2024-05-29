// RUN: triton-opt %s --split-input-file --decompose-unsupported-amd-conversions=arch=gfx942 | FileCheck %s

// CHECK: #[[BLOCKED:.+]] = #triton_gpu.blocked<{{.*}}>
// CHECK: #[[WMMA:.+]] = #triton_gpu.amd_wmma<{{.*}}>
// CHECK: #[[SHARED:.+]] = #triton_gpu.shared<{{.*}}>
#mma = #triton_gpu.amd_wmma<{warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @wmma_to_wmma_dot_op(%arg0: tensor<16x16xf16, #mma>) {
    // CHECK: %[[SRC_BLOCKED:.+]] = triton_gpu.convert_layout %{{.*}} : tensor<16x16xf16, #[[WMMA]]> -> tensor<16x16xf16, #[[BLOCKED]]>
    // CHECK-NEXT: %[[INT_SHARED:.+]] = triton_gpu.local_alloc %[[SRC_BLOCKED]] : {{.*}} -> !tt.memdesc<16x16xf16, #[[SHARED]], #triton_gpu.shared_memory>
    // CHECK-NEXT: %[[DST_DOT_OP:.+]] = triton_gpu.local_load %[[INT_SHARED]] : {{.*}} -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[WMMA]], kWidth = 16}>>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    tt.return
  }
}
