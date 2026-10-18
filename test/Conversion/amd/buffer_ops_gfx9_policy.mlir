// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s

// A buffer descriptor packs its base and num_records into the resource;
// soffset feeds the instruction's scalar offset, scaled from elements to bytes.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_desc_soffset
  tt.func @buffer_desc_soffset(%arg0: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>, %s: i32, %n: i32) {
    // CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: %[[NUM:.*]] = llvm.zext %{{.*}} : i32 to i64
    // CHECK: %[[RSRC:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %[[STRIDE]], %[[NUM]], %{{.*}}
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: %[[SOFF:.*]] = llvm.mul %[[C4]], %{{.*}} : i32
    // CHECK: rocdl.raw.ptr.buffer.load %[[RSRC]], %{{.*}}, %[[SOFF]], 0 : f32
    %0 = amdg.buffer_load %desc[%offsets] soffset = %s : i32 : !amdg.buffer_desc<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %[[RSRC]], %{{.*}}, %{{.*}}, 0 : f32
    amdg.buffer_store %0, %desc[%offsets] soffset = %s : i32 : !amdg.buffer_desc<f32> -> tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

// A tensor soffset that is uniform across each instruction's lanes is read from
// the first lane into the scalar offset.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_desc_tensor_soffset_uniform
  tt.func @buffer_desc_tensor_soffset_uniform(%arg0: !tt.ptr<f32>, %offsets: tensor<4x64xi32, #blocked>, %rows: tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, %n: i32) {
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    %0 = tt.expand_dims %rows {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
    %s = tt.broadcast %0 : tensor<4x1xi32, #blocked> -> tensor<4x64xi32, #blocked>
    // CHECK: rocdl.make.buffer.rsrc
    // CHECK: %[[FIRST:.*]] = rocdl.readfirstlane %{{.*}} : i32
    // CHECK: %[[SOFF:.*]] = llvm.mul %{{.*}}, %[[FIRST]] : i32
    // CHECK-NOT: llvm.add
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %[[SOFF]], 0 : f32
    %1 = amdg.buffer_load %desc[%offsets] soffset = %s : tensor<4x64xi32, #blocked> : !amdg.buffer_desc<f32> -> tensor<4x64xf32, #blocked>
    // CHECK: %[[FIRST2:.*]] = rocdl.readfirstlane %{{.*}} : i32
    // CHECK: %[[SOFF2:.*]] = llvm.mul %{{.*}}, %[[FIRST2]] : i32
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %[[SOFF2]], 0 : f32
    amdg.buffer_store %1, %desc[%offsets] soffset = %s : tensor<4x64xi32, #blocked> : !amdg.buffer_desc<f32> -> tensor<4x64xf32, #blocked>
    tt.return
  }
}

// -----

// scope selects SC0/SC1 and the gfx9 policy selects NT, independently.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_scope_policy
  tt.func @buffer_scope_policy(%arg0: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>) {
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 2 : f32
    %0 = amdg.buffer_load %arg0[%offsets] {cachePolicy = #amdg.cache_policy_gfx9<nontemporal = true>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 1 : f32
    %1 = amdg.buffer_load %arg0[%offsets] {scope = 2 : i32} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 18 : f32
    %2 = amdg.buffer_load %arg0[%offsets] {scope = 1 : i32, cachePolicy = #amdg.cache_policy_gfx9<nontemporal = true>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 17 : f32
    amdg.buffer_store %0, %arg0[%offsets] {scope = 3 : i32} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

// Atomic buffer loads use the atomic buffer-load intrinsic, coherent at their
// scope, with an acquire fence after them; atomic stores are scoped stores
// after a release fence.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_atomic_load_store
  tt.func @buffer_atomic_load_store(%arg0: !tt.ptr<i32>, %offsets: tensor<64xi32, #blocked>, %mask: tensor<64xi1, #blocked>, %s: i32, %n: i32) {
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<i32> -> !amdg.buffer_desc<i32>
    // CHECK: %[[AUX:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.atomic.buffer.load"(%{{.*}}, %{{.*}}, %{{.*}}, %[[AUX]]) : (!llvm.ptr<8>, i32, i32, i32) -> i32
    // CHECK: llvm.fence syncscope("agent") acquire
    %0 = amdg.buffer_atomic_load acquire, gpu, %desc[%offsets], %mask soffset = %s : !amdg.buffer_desc<i32> -> tensor<64xi32, #blocked>
    // CHECK: llvm.fence release
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 17 : i32
    amdg.buffer_atomic_store release, sys, %0, %desc[%offsets], %mask soffset = %s : !amdg.buffer_desc<i32> -> tensor<64xi32, #blocked>
    tt.return
  }
}
