// RUN: triton-opt %s --split-input-file --decompose-unsupported-amd-conversions | FileCheck %s

// CHECK: #[[$BLOCKED:.+]] = #triton_gpu.blocked<{{.*}}>
// CHECK: #[[$WMMA:.+]] = #triton_gpu.amd_wmma<{{.*}}>
// CHECK: #[[$SHARED:.+]] = #triton_gpu.shared<{{.*}}>
// CHECK-LABEL: wmma_to_wmma_dot_op
#mma = #triton_gpu.amd_wmma<{version = 1, warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx1130", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @wmma_to_wmma_dot_op(%arg0: tensor<16x16xf16, #mma>) {
    // CHECK: %[[SRC_BLOCKED:.+]] = triton_gpu.convert_layout %{{.*}} : tensor<16x16xf16, #[[$WMMA]]> -> tensor<16x16xf16, #[[$BLOCKED]]>
    // CHECK-NEXT: %[[INT_SHARED:.+]] = triton_gpu.local_alloc %[[SRC_BLOCKED]] : {{.*}} -> !tt.memdesc<16x16xf16, #[[$SHARED]], #triton_gpu.shared_memory>
    // CHECK-NEXT: %[[DST_DOT_OP:.+]] = triton_gpu.local_load %[[INT_SHARED]] : {{.*}} -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[$WMMA]], kWidth = 16}>>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    tt.return
  }
}

// -----

// CHECK: #[[$BLOCKED:.+]] = #triton_gpu.blocked<{{.*}}>
// CHECK: #[[$WMMA:.+]] = #triton_gpu.amd_wmma<{{.*}}>
// CHECK: #[[$SHARED:.+]] = #triton_gpu.shared<{{.*}}>
// CHECK-LABEL: wmma_to_wmma_dot3d_op
#mma = #triton_gpu.amd_wmma<{version = 1, warpsPerCTA = [2, 2, 2]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @wmma_to_wmma_dot3d_op(%arg0: tensor<2x16x16xf16, #mma>) {
    // CHECK: %[[SRC_BLOCKED:.+]] = triton_gpu.convert_layout %{{.*}} : tensor<2x16x16xf16, #[[WMMA]]> -> tensor<2x16x16xf16, #[[$BLOCKED]]>
    // CHECK-NEXT: %[[INT_SHARED:.+]] = triton_gpu.local_alloc %[[SRC_BLOCKED]] : {{.*}} -> !tt.memdesc<2x16x16xf16, #[[$SHARED]], #triton_gpu.shared_memory>
    // CHECK-NEXT: %[[DST_DOT_OP:.+]] = triton_gpu.local_load %[[INT_SHARED]] : {{.*}} -> tensor<2x16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[$WMMA]], kWidth = 16}>>
    %0 = triton_gpu.convert_layout %arg0 : tensor<2x16x16xf16, #mma> -> tensor<2x16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot_op_shortcut_gfx1130
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx1130", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @blocked_to_dot_op_shortcut_gfx1130(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: triton_gpu.local_alloc
    // CHECK: triton_gpu.convert_layout
    // CHECK-NOT: triton_gpu.local_alloc
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot_op_shortcut_gfx940
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx940", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @blocked_to_dot_op_shortcut_gfx940(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: triton_gpu.local_alloc
    // CHECK: triton_gpu.convert_layout
    // CHECK-NOT: triton_gpu.local_alloc
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}
