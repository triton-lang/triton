// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar='print-intervals=true' 2>&1 | FileCheck %s


#shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: Function: simple_1d_memdesc_index
tt.func public @simple_1d_memdesc_index(%desc_k: !tt.ptr<f8E5M2>, %N_CTX: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c15_i32 = arith.constant 14 : i32
  %k_83 = ttg.local_alloc : () -> !ttg.memdesc<2x16x32xf8E5M2, #shared, #smem, mutable>
  %dummy_value = arith.constant dense<0.000000e+00> : tensor<32xf8E5M2>
  %k_83_slice0 = ttg.memdesc_index %k_83[%c0_i32] : !ttg.memdesc<2x16x32xf8E5M2, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf8E5M2, #shared, #smem, mutable>
  %k_83_slice1 = ttg.memdesc_index %k_83_slice0[%c15_i32] : !ttg.memdesc<16x32xf8E5M2, #shared, #smem, mutable> -> !ttg.memdesc<32xf8E5M2, #shared, #smem, mutable>
  // CHECK: Op: ttg.local_store
  // CHECK-NEXT: Block Interval:
  // CHECK-NEXT:   Read Intervals:
  // CHECK-NEXT:   Write Intervals:
  // CHECK-NEXT:     [448, 480) ttg.local_store
  // CHECK-NEXT: ---
  ttg.local_store %dummy_value, %k_83_slice1 : tensor<32xf8E5M2> -> !ttg.memdesc<32xf8E5M2, #shared, #smem, mutable>
  // CHECK: Op: {{.*}}ttg.local_load
  // CHECK-NEXT: Block Interval:
  // CHECK-NEXT:   Read Intervals:
  // CHECK-NEXT:     [448, 480) ttg.local_load
  // CHECK-NEXT:   Write Intervals:
  // CHECK-NEXT: ---
  %loaded = ttg.local_load %k_83_slice1 : !ttg.memdesc<32xf8E5M2, #shared, #smem, mutable> -> tensor<32xf8E5M2>
  tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 8, order = [0, 1, 2]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: Function: simple_3d
tt.func public @simple_3d(%desc_k: !tt.ptr<f8E5M2>, %N_CTX: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c15_i32 = arith.constant 15 : i32
  %k_83 = ttg.local_alloc : () -> !ttg.memdesc<2x16x32xf8E5M2, #shared, #smem, mutable>
  %dummy_value = arith.constant dense<0.000000e+00> : tensor<2x16x32xf8E5M2>
  // CHECK: Op: ttg.local_store
  // CHECK-NEXT: Block Interval:
  // CHECK-NEXT:   Read Intervals:
  // CHECK-NEXT:   Write Intervals:
  // CHECK-NEXT:     [0, 1024) ttg.local_store
  // CHECK-NEXT: ---
  ttg.local_store %dummy_value, %k_83 : tensor<2x16x32xf8E5M2> -> !ttg.memdesc<2x16x32xf8E5M2, #shared, #smem, mutable>
  // CHECK: Op: {{.*}}ttg.local_load
  // CHECK-NEXT: Block Interval:
  // CHECK-NEXT:   Read Intervals:
  // CHECK-NEXT:     [0, 1024) ttg.local_load
  // CHECK-NEXT:   Write Intervals:
  // CHECK-NEXT: ---
  %loaded = ttg.local_load %k_83 : !ttg.memdesc<2x16x32xf8E5M2, #shared, #smem, mutable> -> tensor<2x16x32xf8E5M2>
  tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 8, order = [0, 1, 2]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: Function: simple_3d_with_index
tt.func public @simple_3d_with_index(%desc_k: !tt.ptr<f8E5M2>, %N_CTX: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c15_i32 = arith.constant 15 : i32
  %k_83 = ttg.local_alloc : () -> !ttg.memdesc<1x2x16x32xf8E5M2, #shared, #smem, mutable>
  %view = ttg.memdesc_index %k_83[%c0_i32] : !ttg.memdesc<1x2x16x32xf8E5M2, #shared, #smem, mutable> -> !ttg.memdesc<2x16x32xf8E5M2, #shared, #smem, mutable>
  %dummy_value = arith.constant dense<0.000000e+00> : tensor<2x16x32xf8E5M2>
  // CHECK: Op: ttg.local_store
  // CHECK-NEXT: Block Interval:
  // CHECK-NEXT:   Read Intervals:
  // CHECK-NEXT:   Write Intervals:
  // CHECK-NEXT:     [0, 32) ttg.local_store
  // CHECK-NEXT:     [32, 64) ttg.local_store
  // CHECK-NEXT:     [64, 96) ttg.local_store
  // CHECK-NEXT:     [96, 128) ttg.local_store
  // CHECK-NEXT:     [128, 160) ttg.local_store
  // CHECK-NEXT:     [160, 192) ttg.local_store
  // CHECK-NEXT:     [192, 224) ttg.local_store
  // CHECK-NEXT:     [224, 256) ttg.local_store
  // CHECK-NEXT:     [256, 288) ttg.local_store
  // CHECK-NEXT:     [288, 320) ttg.local_store
  // CHECK-NEXT:     [320, 352) ttg.local_store
  // CHECK-NEXT:     [352, 384) ttg.local_store
  // CHECK-NEXT:     [384, 416) ttg.local_store
  // CHECK-NEXT:     [416, 448) ttg.local_store
  // CHECK-NEXT:     [448, 480) ttg.local_store
  // CHECK-NEXT:     [480, 512) ttg.local_store
  // CHECK-NEXT:     [512, 544) ttg.local_store
  // CHECK-NEXT:     [544, 576) ttg.local_store
  // CHECK-NEXT:     [576, 608) ttg.local_store
  // CHECK-NEXT:     [608, 640) ttg.local_store
  // CHECK-NEXT:     [640, 672) ttg.local_store
  // CHECK-NEXT:     [672, 704) ttg.local_store
  // CHECK-NEXT:     [704, 736) ttg.local_store
  // CHECK-NEXT:     [736, 768) ttg.local_store
  // CHECK-NEXT:     [768, 800) ttg.local_store
  // CHECK-NEXT:     [800, 832) ttg.local_store
  // CHECK-NEXT:     [832, 864) ttg.local_store
  // CHECK-NEXT:     [864, 896) ttg.local_store
  // CHECK-NEXT:     [896, 928) ttg.local_store
  // CHECK-NEXT:     [928, 960) ttg.local_store
  // CHECK-NEXT:     [960, 992) ttg.local_store
  // CHECK-NEXT:     [992, 1024) ttg.local_store
  // CHECK-NEXT: ---
  ttg.local_store %dummy_value, %view : tensor<2x16x32xf8E5M2> -> !ttg.memdesc<2x16x32xf8E5M2, #shared, #smem, mutable>
  tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: Function: simple_subslice
tt.func public @simple_subslice(%desc_k: !tt.ptr<f8E5M2>, %N_CTX: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c15_i32 = arith.constant 14 : i32
  %k_83 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %dummy_value = arith.constant dense<0.000000e+00> : tensor<32x16xf16>
  %view1 = ttg.memdesc_subslice %k_83[32, 64] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable, 128x128>
  // CHECK: Op: ttg.local_store
  // CHECK-NEXT: Block Interval:
  // CHECK-NEXT:   Read Intervals:
  // CHECK-NEXT:   Write Intervals:
  // CHECK-NEXT:     [8320, 8352) ttg.local_store
  // CHECK-NEXT:     [8576, 8608) ttg.local_store
  // CHECK-NEXT:     [8832, 8864) ttg.local_store
  // CHECK-NEXT:     [9088, 9120) ttg.local_store
  // CHECK-NEXT:     [9344, 9376) ttg.local_store
  // CHECK-NEXT:     [9600, 9632) ttg.local_store
  // CHECK-NEXT:     [9856, 9888) ttg.local_store
  // CHECK-NEXT:     [10112, 10144) ttg.local_store
  // CHECK-NEXT:     [10368, 10400) ttg.local_store
  // CHECK-NEXT:     [10624, 10656) ttg.local_store
  // CHECK-NEXT:     [10880, 10912) ttg.local_store
  // CHECK-NEXT:     [11136, 11168) ttg.local_store
  // CHECK-NEXT:     [11392, 11424) ttg.local_store
  // CHECK-NEXT:     [11648, 11680) ttg.local_store
  // CHECK-NEXT:     [11904, 11936) ttg.local_store
  // CHECK-NEXT:     [12160, 12192) ttg.local_store
  // CHECK-NEXT:     [12416, 12448) ttg.local_store
  // CHECK-NEXT:     [12672, 12704) ttg.local_store
  // CHECK-NEXT:     [12928, 12960) ttg.local_store
  // CHECK-NEXT:     [13184, 13216) ttg.local_store
  // CHECK-NEXT:     [13440, 13472) ttg.local_store
  // CHECK-NEXT:     [13696, 13728) ttg.local_store
  // CHECK-NEXT:     [13952, 13984) ttg.local_store
  // CHECK-NEXT:     [14208, 14240) ttg.local_store
  // CHECK-NEXT:     [14464, 14496) ttg.local_store
  // CHECK-NEXT:     [14720, 14752) ttg.local_store
  // CHECK-NEXT:     [14976, 15008) ttg.local_store
  // CHECK-NEXT:     [15232, 15264) ttg.local_store
  // CHECK-NEXT:     [15488, 15520) ttg.local_store
  // CHECK-NEXT:     [15744, 15776) ttg.local_store
  // CHECK-NEXT:     [16000, 16032) ttg.local_store
  // CHECK-NEXT:     [16256, 16288) ttg.local_store
  // CHECK-NEXT: ---
  ttg.local_store %dummy_value, %view1 : tensor<32x16xf16> -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable, 128x128>
  tt.return
}


// -----
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

// %c_dyn (view2) is dynamic offset so its uses will alias with all other index
// CHECK-LABEL: Function: memindex_aliasing
tt.func public @memindex_aliasing(%c_dyn : i32) {
    %c1 = arith.constant 1 : i32
    %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable>
    %viewA = ttg.memdesc_index %alloc[%c1] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %viewB = ttg.memdesc_index %alloc[%c_dyn] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %dummy_value = arith.constant dense<0.000000e+00> : tensor<4x32xf16>
    %sliceA = ttg.memdesc_subslice %viewA[32, 64] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<4x32xf16, #shared, #smem, mutable, 128x128>
    %sliceB = ttg.memdesc_subslice %viewB[32, 64] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<4x32xf16, #shared, #smem, mutable, 128x128>
    // CHECK: Op: ttg.local_store
    // CHECK-NEXT: Block Interval:
    // CHECK-NEXT:   Read Intervals:
    // CHECK-NEXT:   Write Intervals:
    // CHECK-NEXT:     [41088, 41152) ttg.local_store
    // CHECK-NEXT:     [41344, 41408) ttg.local_store
    // CHECK-NEXT:     [41600, 41664) ttg.local_store
    // CHECK-NEXT:     [41856, 41920) ttg.local_store
    // CHECK-NEXT: ---
    ttg.local_store %dummy_value, %sliceA : tensor<4x32xf16> -> !ttg.memdesc<4x32xf16, #shared, #smem, mutable, 128x128>
    // CHECK: Op: {{.*}}ttg.local_load
    // CHECK-NEXT: Block Interval:
    // CHECK-NEXT:   Read Intervals:
    // CHECK-NEXT:     [41088, 41152) ttg.local_load
    // CHECK-NEXT:     [41344, 41408) ttg.local_load
    // CHECK-NEXT:     [41600, 41664) ttg.local_load
    // CHECK-NEXT:     [41856, 41920) ttg.local_load
    // CHECK-NEXT:   Write Intervals:
    // CHECK-NEXT: ---
    %loadedA = ttg.local_load %sliceA : !ttg.memdesc<4x32xf16, #shared, #smem, mutable, 128x128> -> tensor<4x32xf16>
    // CHECK: Op: ttg.local_store
    // CHECK-NEXT: Block Interval:
    // CHECK-NEXT:   Read Intervals:
    // CHECK-NEXT:   Write Intervals:
    // CHECK-NEXT:     [8320, 8384) ttg.local_store
    // CHECK-NEXT:     [8576, 8640) ttg.local_store
    // CHECK-NEXT:     [8832, 8896) ttg.local_store
    // CHECK-NEXT:     [9088, 9152) ttg.local_store
    // CHECK-NEXT:     [41088, 41152) ttg.local_store
    // CHECK-NEXT:     [41344, 41408) ttg.local_store
    // CHECK-NEXT:     [41600, 41664) ttg.local_store
    // CHECK-NEXT:     [41856, 41920) ttg.local_store
    // CHECK-NEXT:     [73856, 73920) ttg.local_store
    // CHECK-NEXT:     [74112, 74176) ttg.local_store
    // CHECK-NEXT:     [74368, 74432) ttg.local_store
    // CHECK-NEXT:     [74624, 74688) ttg.local_store
    // CHECK-NEXT:     [106624, 106688) ttg.local_store
    // CHECK-NEXT:     [106880, 106944) ttg.local_store
    // CHECK-NEXT:     [107136, 107200) ttg.local_store
    // CHECK-NEXT:     [107392, 107456) ttg.local_store
    // CHECK-NEXT: ---
    ttg.local_store %dummy_value, %sliceB : tensor<4x32xf16> -> !ttg.memdesc<4x32xf16, #shared, #smem, mutable, 128x128>
    // CHECK: Op: {{.*}}ttg.local_load
    // CHECK-NEXT: Block Interval:
    // CHECK-NEXT:   Read Intervals:
    // CHECK-NEXT:     [8320, 8384) ttg.local_load
    // CHECK-NEXT:     [8576, 8640) ttg.local_load
    // CHECK-NEXT:     [8832, 8896) ttg.local_load
    // CHECK-NEXT:     [9088, 9152) ttg.local_load
    // CHECK-NEXT:     [41088, 41152) ttg.local_load
    // CHECK-NEXT:     [41344, 41408) ttg.local_load
    // CHECK-NEXT:     [41600, 41664) ttg.local_load
    // CHECK-NEXT:     [41856, 41920) ttg.local_load
    // CHECK-NEXT:     [73856, 73920) ttg.local_load
    // CHECK-NEXT:     [74112, 74176) ttg.local_load
    // CHECK-NEXT:     [74368, 74432) ttg.local_load
    // CHECK-NEXT:     [74624, 74688) ttg.local_load
    // CHECK-NEXT:     [106624, 106688) ttg.local_load
    // CHECK-NEXT:     [106880, 106944) ttg.local_load
    // CHECK-NEXT:     [107136, 107200) ttg.local_load
    // CHECK-NEXT:     [107392, 107456) ttg.local_load
    // CHECK-NEXT:   Write Intervals:
    // CHECK-NEXT: ---
    %loadedB = ttg.local_load %sliceB : !ttg.memdesc<4x32xf16, #shared, #smem, mutable, 128x128> -> tensor<4x32xf16>
    tt.return
}
