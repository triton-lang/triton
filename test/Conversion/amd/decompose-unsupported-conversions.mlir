// RUN: triton-opt %s --split-input-file --decompose-unsupported-amd-conversions | FileCheck %s

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{{.*}}>
// CHECK: #[[$WMMA:.+]] = #ttg.amd_wmma<{{.*}}>
// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{{.*}}>
// CHECK-LABEL: wmma_to_wmma_dot_op
#mma = #ttg.amd_wmma<{version = 1, warpsPerCTA = [2, 2]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1130", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_to_wmma_dot_op(%arg0: tensor<16x16xf16, #mma>) {
    // CHECK: %[[SRC_BLOCKED:.+]] = ttg.convert_layout %{{.*}} : tensor<16x16xf16, #[[$WMMA]]> -> tensor<16x16xf16, #[[$BLOCKED]]>
    // CHECK-NEXT: %[[INT_SHARED:.+]] = ttg.local_alloc %[[SRC_BLOCKED]] : {{.*}} -> !ttg.memdesc<16x16xf16, #[[$SHARED]], #smem>
    // CHECK-NEXT: %[[DST_DOT_OP:.+]] = ttg.local_load %[[INT_SHARED]] : {{.*}} -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$WMMA]], kWidth = 16}>>
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    tt.return
  }
}

// -----

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked<{{.*}}>
// CHECK: #[[$WMMA:.+]] = #ttg.amd_wmma<{{.*}}>
// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{{.*}}>
// CHECK-LABEL: wmma_to_wmma_dot3d_op
#mma = #ttg.amd_wmma<{version = 1, warpsPerCTA = [2, 2, 2]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_to_wmma_dot3d_op(%arg0: tensor<2x16x16xf16, #mma>) {
    // CHECK: %[[SRC_BLOCKED:.+]] = ttg.convert_layout %{{.*}} : tensor<2x16x16xf16, #[[$WMMA]]> -> tensor<2x16x16xf16, #[[$BLOCKED]]>
    // CHECK-NEXT: %[[INT_SHARED:.+]] = ttg.local_alloc %[[SRC_BLOCKED]] : {{.*}} -> !ttg.memdesc<2x16x16xf16, #[[$SHARED]], #smem>
    // CHECK-NEXT: %[[DST_DOT_OP:.+]] = ttg.local_load %[[INT_SHARED]] : {{.*}} -> tensor<2x16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$WMMA]], kWidth = 16}>>
    %0 = ttg.convert_layout %arg0 : tensor<2x16x16xf16, #mma> -> tensor<2x16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot_op_shortcut_gfx1130
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1130", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @blocked_to_dot_op_shortcut_gfx1130(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: ttg.local_alloc
    // CHECK: ttg.convert_layout
    // CHECK-NOT: ttg.local_alloc
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot_op_shortcut_gfx940
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @blocked_to_dot_op_shortcut_gfx940(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: ttg.local_alloc
    // CHECK: ttg.convert_layout
    // CHECK-NOT: ttg.local_alloc
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: neg_blocked_to_dot_op_incompatible_elems_gfx940
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @neg_blocked_to_dot_op_incompatible_elems_gfx940(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: neg_blocked_to_dot_op_incompatible_threads_gfx940
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @neg_blocked_to_dot_op_incompatible_threads_gfx940(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: neg_blocked_to_dot_op_incompatible_warp_gfx940
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @neg_blocked_to_dot_op_incompatible_warp_gfx940(%arg0: tensor<128x128xf16, #blocked>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %0 = ttg.convert_layout %arg0 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    tt.return
  }
}
