// RUN: triton-opt %s -split-input-file -test-tritonamdgpu-split-memdescsubview="num-splits=2,4" | FileCheck %s


#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: memdesc_subview_spliting
  tt.func public @memdesc_subview_spliting() attributes {noinline = false} {
    // CHECK: [[ORIG:%.*]] = ttg.memdesc_subview
    // CHECK-SAME: -> !ttg.memdesc<[[ORIG_SHAPE:.*]]>
    // CHECK: [[OFF_0_0:%.*]] = arith.constant 0 : i32
    // CHECK: [[OFF_0_1:%.*]] = arith.constant 0 : i32
    // CHECK: ttg.memdesc_subview [[ORIG]][[[OFF_0_0]], [[OFF_0_1]]] : !ttg.memdesc<[[ORIG_SHAPE]]> -> !ttg.memdesc<[[SPLIT_SHAPE:.*]]>
    // CHECK: [[OFF_1_0:%.*]] = arith.constant 0 : i32
    // CHECK: [[OFF_1_1:%.*]] = arith.constant 32 : i32
    // CHECK: ttg.memdesc_subview [[ORIG]][[[OFF_1_0]], [[OFF_1_1]]] : !ttg.memdesc<[[ORIG_SHAPE]]> -> !ttg.memdesc<[[SPLIT_SHAPE]]>
    // CHECK: [[OFF_2_0:%.*]] = arith.constant 0 : i32
    // CHECK: [[OFF_2_1:%.*]] = arith.constant 64 : i32
    // CHECK: ttg.memdesc_subview [[ORIG]][[[OFF_2_0]], [[OFF_2_1]]] : !ttg.memdesc<[[ORIG_SHAPE]]> -> !ttg.memdesc<[[SPLIT_SHAPE]]>
    // CHECK: [[OFF_3_0:%.*]] = arith.constant 0 : i32
    // CHECK: [[OFF_3_1:%.*]] = arith.constant 96 : i32
    // CHECK: ttg.memdesc_subview [[ORIG]][[[OFF_3_0]], [[OFF_3_1]]] : !ttg.memdesc<[[ORIG_SHAPE]]> -> !ttg.memdesc<[[SPLIT_SHAPE]]>
    // CHECK: [[OFF_4_0:%.*]] = arith.constant 128 : i32
    // CHECK: [[OFF_4_1:%.*]] = arith.constant 0 : i32
    // CHECK: ttg.memdesc_subview [[ORIG]][[[OFF_4_0]], [[OFF_4_1]]] : !ttg.memdesc<[[ORIG_SHAPE]]> -> !ttg.memdesc<[[SPLIT_SHAPE]]>
    // CHECK: [[OFF_5_0:%.*]] = arith.constant 128 : i32
    // CHECK: [[OFF_5_1:%.*]] = arith.constant 32 : i32
    // CHECK: ttg.memdesc_subview [[ORIG]][[[OFF_5_0]], [[OFF_5_1]]] : !ttg.memdesc<[[ORIG_SHAPE]]> -> !ttg.memdesc<[[SPLIT_SHAPE]]>
    // CHECK: [[OFF_6_0:%.*]] = arith.constant 128 : i32
    // CHECK: [[OFF_6_1:%.*]] = arith.constant 64 : i32
    // CHECK: ttg.memdesc_subview [[ORIG]][[[OFF_6_0]], [[OFF_6_1]]] : !ttg.memdesc<[[ORIG_SHAPE]]> -> !ttg.memdesc<[[SPLIT_SHAPE]]>
    // CHECK: [[OFF_7_0:%.*]] = arith.constant 128 : i32
    // CHECK: [[OFF_7_1:%.*]] = arith.constant 96 : i32
    // CHECK: ttg.memdesc_subview [[ORIG]][[[OFF_7_0]], [[OFF_7_1]]] : !ttg.memdesc<[[ORIG_SHAPE]]> -> !ttg.memdesc<[[SPLIT_SHAPE]]>

    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    tt.return
  }
}
