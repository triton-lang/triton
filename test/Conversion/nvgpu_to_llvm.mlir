// RUN: triton-opt %s --convert-nv-gpu-to-llvm  -split-input-file | FileCheck %s

// CHECK-LABEL: @nvvm_syncs
llvm.func @nvvm_syncs() {
  // CHECK: wgmma.fence.sync.aligned;
  nvgpu.wgmma_fence

  // CHECK: wgmma.commit_group.sync.aligned;
  nvgpu.wgmma_commit_group

  // CHECK: barrier.cluster.wait.aligned;
  nvgpu.cluster_wait

  // CHECK: fence.proxy.async.shared::cta;
  nvgpu.fence_async_shared {bCluster = false}
  // CHECK: fence.proxy.async.shared::cluster;
  nvgpu.fence_async_shared {bCluster = true}

  // CHECK: barrier.cluster.arrive.aligned;
  nvgpu.cluster_arrive {relaxed = false}
  // CHECK: barrier.cluster.arrive.relaxed.aligned;
  nvgpu.cluster_arrive {relaxed = true}

  llvm.return
}

// CHECK-LABEL: @cluster_id
llvm.func @cluster_id() -> i32 {
  // CHECK:      %cluster_ctaid.x;
  // CHECK-SAME: %cluster_ctaid.y;
  // CHECK-SAME: %cluster_ctaid.z;
  // CHECK-SAME: %cluster_nctaid.x;
  // CHECK-SAME: %cluster_nctaid.y;
  %id = nvgpu.cluster_id
  llvm.return %id : i32
}

// -----

// CHECK-LABEL: @st_matrix
llvm.func @st_matrix(%i: i32, %ptr: !llvm.ptr<3>) {
  // CHECK: stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1, $2, $3, $4};
  nvgpu.stmatrix %ptr, %i, %i, %i, %i : !llvm.ptr<3>, i32, i32, i32, i32
  llvm.return
}

// -----

!struct_128xf32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
)>

!struct_64xf32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
)>

// CHECK-LABEL: @wgmma
llvm.func @wgmma(%desc: i64, %in: !struct_64xf32) {
// CHECK: wgmma.mma_async.sync.aligned.m64n256k32.f32.e5m2.e5m2
%acc0 = nvgpu.wgmma %desc, %desc {
  eltTypeA = 3 : i32,
  eltTypeB = 3 : i32,
  eltTypeC = 7 : i32,
  layoutA = 0 : i32,
  layoutB = 1 : i32,
  m = 64 : i32,
  n = 256 : i32,
  k = 32 : i32
} : (i64, i64) -> !struct_128xf32

  // CHECK: // wait for regs: $0,$1,$2,{{.*}},$127
  // CHECK: wgmma.wait_group.sync.aligned 0;
  %out = nvgpu.wgmma_wait_group %in {pendings = 0 : i32} : !struct_64xf32
  llvm.return
}
