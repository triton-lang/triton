// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s

#axis2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scan_axis_cta
  // CHECK: nvg.cluster_id
  // CHECK: st.shared::cta
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: nvvm.mapa
  tt.func @scan_axis_cta(%v: tensor<32x32xf32, #axis2>) {
    %r = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      "tt.scan.return"(%s) : (f32) -> ()
    }) : (tensor<32x32xf32, #axis2>) -> tensor<32x32xf32, #axis2>
    tt.return
  }
}

// -----

#parallel2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scan_parallel_cta
  // CHECK-NOT: nvg.cluster_id
  // CHECK-NOT: nvvm.cluster.arrive
  // CHECK-NOT: nvvm.mapa
  // CHECK: llvm.return
  tt.func @scan_parallel_cta(%v: tensor<32x32xf32, #parallel2>) {
    %r = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      "tt.scan.return"(%s) : (f32) -> ()
    }) : (tensor<32x32xf32, #parallel2>) -> tensor<32x32xf32, #parallel2>
    tt.return
  }
}

// -----

#axis4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scan_axis_cta_reverse
  // CHECK: nvg.cluster_id
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: nvvm.mapa
  // CHECK: nvvm.mapa
  // CHECK: nvvm.mapa
  tt.func @scan_axis_cta_reverse(%v: tensor<32x32xf32, #axis4>) {
    %r = "tt.scan"(%v) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      "tt.scan.return"(%s) : (f32) -> ()
    }) : (tensor<32x32xf32, #axis4>) -> tensor<32x32xf32, #axis4>
    tt.return
  }
}

// -----

#onewarp = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [1, 0], CGALayout = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scan_one_warp_local_blocks
  // CHECK: nvg.cluster_id
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: nvvm.mapa
  tt.func @scan_one_warp_local_blocks(%v: tensor<128x16xf32, #onewarp>) {
    %r = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      "tt.scan.return"(%s) : (f32) -> ()
    }) : (tensor<128x16xf32, #onewarp>) -> tensor<128x16xf32, #onewarp>
    tt.return
  }
}

// -----

#mixed4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [0, 1]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Preserve the non-axis CTA coordinate when selecting an axis predecessor.
  // CHECK-LABEL: llvm.func @scan_mixed_cta
  // CHECK: [[CTA:%.*]] = nvg.cluster_id
  // CHECK: [[DIV:%.*]] = llvm.udiv [[CTA]], %{{.*}} : i32
  // CHECK: [[PARALLEL:%.*]] = llvm.urem [[DIV]], %{{.*}} : i32
  // CHECK: nvvm.cluster.wait
  // CHECK: [[PARALLEL_BASE:%.*]] = llvm.mul [[PARALLEL]], %{{.*}} : i32
  // CHECK: [[TARGET:%.*]] = llvm.add [[PARALLEL_BASE]], %{{.*}} : i32
  // CHECK: nvvm.mapa %{{.*}}, [[TARGET]]
  tt.func @scan_mixed_cta(%v: tensor<32x32xf32, #mixed4>) {
    %r = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      "tt.scan.return"(%s) : (f32) -> ()
    }) : (tensor<32x32xf32, #mixed4>) -> tensor<32x32xf32, #mixed4>
    tt.return
  }
}
