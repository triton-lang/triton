// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s

// COM: Tests reduction when threads_per_warp(=16) < num_warps(=64). A possible case for Intel XPUs

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [64], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 64 : i32} {
  // CHECK-LABEL: reduce_problem_size_64_threads_per_warp_32
  tt.func @reduce_problem_size_64_threads_per_warp_32(%f : tensor<2048xi32, #blocked>) {

  // 1st round intra-warp reduce
  // CHECK: %{{.*}} = nvvm.redux.sync  add %{{.*}}, %{{.*}} : i32 -> i32


  // 2nd round inter-warp reduce with problem size 64 with threads_per_warp 32
  // CHECK nvvm.barrier0
  // CHECK [[STRIDE_1:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT [[OFFSET_1:%.*]] = llvm.mul %{{.*}}, [[STRIDE_1]] : i32
  // CHECK %{{.*}} = llvm.icmp "slt" [[OFFSET_1]], %{{.*}} : i32
  // CHECK: %{{.*}} = nvvm.redux.sync  add %{{.*}}, %{{.*}} : i32 -> i32

  // 3rd round inter-warp reduce with problem size 2 with threads_per_warp 32
  // CHECK nvvm.barrier0
  // CHECK [[STRIDE_2:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT [[OFFSET_2:%.*]] = llvm.mul %{{.*}}, [[STRIDE_2]] : i32
  // CHECK %{{.*}} = llvm.icmp "slt" [[OFFSET_1]], %{{.*}} : i32
  // Because reduction size is 2, it performs single shuffle-and-add
  // CHECK: [[SHUFFLE_1:%.*]] = nvvm.shfl.sync  bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32 -> i32
  // CHECK-NEXT %{{.*}} = llvm.add %{{.*}}, [[SHUFFLE_1]] : i32

  // get final result
  // CHECK nvvm.barrier0
  // CHECK: [[FINAL_RESULT:%.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32

    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: i32, %arg1: i32):
      %add = arith.addi %arg0, %arg1 : i32
      tt.reduce.return %add : i32
    }) {axis = 0 : i32} : (tensor<2048xi32, #blocked>) -> i32
    tt.return
  }
}
