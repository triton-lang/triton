// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

// CHECK-LABEL: amd_in_thread_transpose
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 1]], lane = [[0, 2], [0, 4], [0, 8], [2, 0], [4, 0], [8, 0]], warp = [], block = []}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose(%arg0: tensor<16x16xf16, #blocked>) {
    // CHECK-DAG:  [[VEC_UNDEF:%.*]] = llvm.mlir.undef : vector<2xf16>
    // CHECK-DAG: [[CST_0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: [[CST_1:%.*]] = llvm.mlir.constant(1 : i32) : i32

    // CHECK-DAG: [[VAL0:%.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(f16, f16, f16, f16)>
    // CHECK-DAG: [[VAL1:%.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(f16, f16, f16, f16)>
    // CHECK-DAG: [[VAL2:%.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(f16, f16, f16, f16)>
    // CHECK-DAG: [[VAL3:%.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(f16, f16, f16, f16)>

    // CHECK-DAG: [[VEC1_TMP:%.*]] = llvm.insertelement [[VAL0]], [[VEC_UNDEF]]{{\[}}[[CST_0]] : i32] : vector<2xf16>
    // CHECK-DAG: [[VEC1:%.*]] = llvm.insertelement [[VAL2]], [[VEC1_TMP]]{{\[}}[[CST_1]] : i32] : vector<2xf16>
    // CHECK-DAG: llvm.store [[VEC1]], {{.*}} {alignment = 4 : i64} : vector<2xf16>, !llvm.ptr<3>

    // CHECK-DAG: [[VEC2_TMP:%.*]] = llvm.insertelement [[VAL1]], [[VEC_UNDEF]]{{\[}}[[CST_0]] : i32] : vector<2xf16>
    // CHECK-DAG: [[VEC2:%.*]] = llvm.insertelement [[VAL3]], [[VEC2_TMP]]{{\[}}[[CST_1]] : i32] : vector<2xf16>
    // CHECK-DAG: llvm.store [[VEC2]], {{.*}} {alignment = 4 : i64} : vector<2xf16>, !llvm.ptr<3>

    %0 = amdg.in_thread_transpose %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #linear>
    ttg.local_alloc %0 : (tensor<16x16xf16, #linear>) -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: amd_in_thread_transpose_with_reg_repeats
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 1], [0, 16], [16, 0]], lane = [[0, 2], [0, 4], [0, 8], [2, 0], [4, 0], [8, 0]], warp = [], block = []}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_with_reg_repeats(%arg0: tensor<32x32xf16, #blocked>) {
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #linear>
    ttg.local_alloc %0 : (tensor<32x32xf16, #linear>) -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Verify broadcasted registers in source layout are handled correctly
// CHECK-LABEL: amd_in_thread_transpose_skinny_shape
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 0], [0, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [], block = []}>
#linear2 = #ttg.linear<{register = [[1, 0], [0, 1], [0, 2], [0, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [], block = []}>
#linear3 = #ttg.linear<{register = [[1, 0], [0, 1], [0, 2], [0, 0], [0, 256]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [], block = []}>

#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [0, 1]}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [0, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [0, 0], [0, 256]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [], block = []}>

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_skinny_shape(
      %arg1: tensor<1x256xf16, #blocked1>,
      %arg2: tensor<2x256xf16, #blocked1>,
      %arg3: tensor<2x512xf16, #blocked1>,
      %arg4: tensor<1x256xf16, #blocked2>,
      %arg5: tensor<2x256xf16, #blocked2>,
      %arg6: tensor<2x512xf16, #blocked2>
      ) {
    %l1 = amdg.in_thread_transpose %arg1 : tensor<1x256xf16, #blocked1> -> tensor<1x256xf16, #linear1>
    %m1 = ttg.local_alloc %l1 : (tensor<1x256xf16, #linear1>) -> !ttg.memdesc<1x256xf16, #shared, #smem, mutable>

    %l2 = amdg.in_thread_transpose %arg2 : tensor<2x256xf16, #blocked1> -> tensor<2x256xf16, #linear2>
    %m2 = ttg.local_alloc %l2 : (tensor<2x256xf16, #linear2>) -> !ttg.memdesc<2x256xf16, #shared, #smem, mutable>

    %l3 = amdg.in_thread_transpose %arg3 : tensor<2x512xf16, #blocked1> -> tensor<2x512xf16, #linear3>
    %m3 = ttg.local_alloc %l3 : (tensor<2x512xf16, #linear3>) -> !ttg.memdesc<2x512xf16, #shared, #smem, mutable>

    %l4 = amdg.in_thread_transpose %arg4 : tensor<1x256xf16, #blocked2> -> tensor<1x256xf16, #linear1>
    %m4 = ttg.local_alloc %l4 : (tensor<1x256xf16, #linear1>) -> !ttg.memdesc<1x256xf16, #shared, #smem, mutable>

    %l5 = amdg.in_thread_transpose %arg5 : tensor<2x256xf16, #blocked2> -> tensor<2x256xf16, #linear4>
    %m5 = ttg.local_alloc %l5 : (tensor<2x256xf16, #linear4>) -> !ttg.memdesc<2x256xf16, #shared, #smem, mutable>

    %l6 = amdg.in_thread_transpose %arg6 : tensor<2x512xf16, #blocked2> -> tensor<2x512xf16, #linear5>
    %m6 = ttg.local_alloc %l6 : (tensor<2x512xf16, #linear5>) -> !ttg.memdesc<2x512xf16, #shared, #smem, mutable>
    tt.return
  }
}
