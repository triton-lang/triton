// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

// 4 x f32 per lane, 8 lanes per 128-byte group: 8x16B.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @load_acquire_8x16B
  // CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_string<"agent">>
  // CHECK: %[[ORD:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %[[V:.*]] = llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.load.8x16B.p1"(%{{.*}}, %[[ORD]], %[[MD]]) : (!llvm.ptr<1>, i32, !llvm.metadata) -> vector<4xi32>
  // CHECK: llvm.bitcast %[[V]] : vector<4xi32> to vector<4xf32>
  // CHECK-NOT: llvm.amdgcn.cooperative.atomic
  // Program-level acquire: CTA barrier after the load.
  // CHECK: rocdl.s.barrier
  tt.func public @load_acquire_8x16B(%base: !tt.ptr<f32> {tt.divisibility = 128 : i32}, %out: !tt.ptr<f32> {tt.divisibility = 128 : i32}) {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %splat, %range : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    %v = amdg.cooperative_atomic_load acquire, gpu, %ptrs : tensor<512x!tt.ptr<f32>, #blocked>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    tt.store %out_ptrs, %v : tensor<512x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// 1 x i32 per lane, all 32 lanes form one group: 32x4B.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @store_release_32x4B
  // CHECK: llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_string<"agent">>
  // CHECK: %[[ORD:.*]] = llvm.mlir.constant(3 : i32) : i32
  // Program-level release: CTA barrier before the store.
  // CHECK: rocdl.s.barrier
  // CHECK-NOT: llvm.bitcast
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.store.32x4B.p1"(%{{.*}}, %{{.*}}, %[[ORD]], %{{.*}}) : (!llvm.ptr<1>, i32, i32, !llvm.metadata) -> ()
  tt.func public @store_release_32x4B(%base: !tt.ptr<i32> {tt.divisibility = 128 : i32}, %value: tensor<128xi32, #blocked>) {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %splat = tt.splat %base : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %splat, %range : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    amdg.cooperative_atomic_store release, gpu, %ptrs, %value : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// 4 x f16 per lane, 16 lanes per group: 16x8B.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @relaxed_cta_16x8B
  // CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_string<"workgroup">>
  // CHECK: %[[ORD:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[V:.*]] = llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.load.16x8B.p1"(%{{.*}}, %[[ORD]], %[[MD]]) : (!llvm.ptr<1>, i32, !llvm.metadata) -> vector<2xi32>
  // CHECK: llvm.bitcast %[[V]] : vector<2xi32> to vector<4xf16>
  // Relaxed: no barriers.
  // CHECK-NOT: rocdl.s.barrier
  // CHECK: %[[D:.*]] = llvm.bitcast %{{.*}} : vector<4xf16> to vector<2xi32>
  // CHECK-NOT: rocdl.s.barrier
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.store.16x8B.p1"(%{{.*}}, %[[D]], %{{.*}}, %{{.*}}) : (!llvm.ptr<1>, vector<2xi32>, i32, !llvm.metadata) -> ()
  tt.func public @relaxed_cta_16x8B(%base: !tt.ptr<f16> {tt.divisibility = 128 : i32}) {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %splat = tt.splat %base : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %splat, %range : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %v = amdg.cooperative_atomic_load relaxed, cta, %ptrs : tensor<128x!tt.ptr<f16>, #blocked>
    amdg.cooperative_atomic_store relaxed, cta, %ptrs, %v : tensor<128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// 2D: each row of 32 x f32 is one 128-byte group of 8 lanes; a wave covers
// four rows.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @rows_sys_8x16B
  // CHECK: %[[MD:.*]] = llvm.mlir.metadata_as_value #llvm.md_node<#llvm.md_string<"">>
  // CHECK: %[[D:.*]] = llvm.bitcast %{{.*}} : vector<4xf32> to vector<4xi32>
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.cooperative.atomic.store.8x16B.p1"(%{{.*}}, %[[D]], %{{.*}}, %[[MD]]) : (!llvm.ptr<1>, vector<4xi32>, i32, !llvm.metadata) -> ()
  // CHECK-NOT: llvm.amdgcn.cooperative.atomic
  tt.func public @rows_sys_8x16B(%base: !tt.ptr<f32> {tt.divisibility = 128 : i32}, %value: tensor<16x32xf32, #blocked>) {
    %c32 = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #row>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #col>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32, #row> -> tensor<16x1xi32, #blocked>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #col> -> tensor<1x32xi32, #blocked>
    %row_offs = arith.muli %rows2, %c32 : tensor<16x1xi32, #blocked>
    %row_b = tt.broadcast %row_offs : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %col_b = tt.broadcast %cols2 : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %offs = arith.addi %row_b, %col_b : tensor<16x32xi32, #blocked>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %splat, %offs : tensor<16x32x!tt.ptr<f32>, #blocked>, tensor<16x32xi32, #blocked>
    amdg.cooperative_atomic_store relaxed, sys, %ptrs, %value : tensor<16x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
