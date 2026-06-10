// RUN: triton-opt %s --triton-nvidia-gpu-cluster-barrier-mbar-allocator | FileCheck %s

#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedSplitN = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>
#blockedBroadcast = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0]]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blockedSplitM}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 5 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: module attributes {
  // CHECK-DAG: ttg.shared = 40 : i32
  // CHECK-DAG: ttg.ws_cluster_barrier_count = 2 : i32
  // CHECK-LABEL: @cluster_barrier_mbar_allocator
  tt.func @cluster_barrier_mbar_allocator(%ptr: !tt.ptr<i32>) {
    ttg.warp_specialize()
    default {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSplitM>
      // CHECK: ttg.convert_layout {{.*}} {ttg.mbar_offset = 8 : i32}
      %cvt = ttg.convert_layout %cst : tensor<256x128xf16, #blockedSplitM> -> tensor<256x128xf16, #blockedSplitN>
      // CHECK: "tt.reduce"
      // CHECK: }) {ttg.mbar_offset = 8 : i32}
      %red = "tt.reduce"(%cst) ({
      ^bb0(%lhs: f16, %rhs: f16):
        %add = arith.addf %lhs, %rhs : f16
        tt.reduce.return %add : f16
      }) {axis = 0 : i32} : (tensor<256x128xf16, #blockedSplitM>) -> tensor<128xf16, #slice0>
      // CHECK: tt.atomic_cas {{.*}} {ttg.mbar_offset = 8 : i32}
      %cas = tt.atomic_cas acq_rel, gpu, %ptr, %c0, %c1 : (!tt.ptr<i32>, i32, i32) -> i32
      tt.store %ptr, %cas : !tt.ptr<i32>
      %ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blockedBroadcast>
      %zeros = arith.constant dense<0> : tensor<128xi32, #blockedBroadcast>
      %ones = arith.constant dense<1> : tensor<128xi32, #blockedBroadcast>
      // CHECK: tt.atomic_cas {{.*}}ttg.mbar_offset = 8 : i32
      %tensor_cas = tt.atomic_cas acq_rel, gpu, %ptrs, %zeros, %ones {allocation.offset = 0 : i32} : (tensor<128x!tt.ptr<i32>, #blockedBroadcast>, tensor<128xi32, #blockedBroadcast>, tensor<128xi32, #blockedBroadcast>) -> tensor<128xi32, #blockedBroadcast>
      tt.store %ptrs, %tensor_cas : tensor<128x!tt.ptr<i32>, #blockedBroadcast>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      // CHECK: ttng.cluster_barrier {ttg.mbar_offset = 24 : i32}
      ttng.cluster_barrier
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
