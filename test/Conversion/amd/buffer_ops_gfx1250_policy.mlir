// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

// A buffer descriptor packs its base and num_records into the resource;
// soffset feeds the instruction's scalar offset, scaled from elements to bytes.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_desc_soffset
  tt.func @buffer_desc_soffset(%arg0: !tt.ptr<f32>, %offsets: tensor<32xi32, #blocked>, %s: i32, %n: i32) {
    // CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: %[[NUM:.*]] = llvm.zext %{{.*}} : i32 to i64
    // CHECK: %[[RSRC:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %[[STRIDE]], %[[NUM]], %{{.*}}
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: %[[SOFF:.*]] = llvm.mul %[[C4]], %{{.*}} : i32
    // CHECK: rocdl.raw.ptr.buffer.load %[[RSRC]], %{{.*}}, %[[SOFF]], 0 : f32
    %0 = amdg.buffer_load %desc[%offsets] soffset = %s : i32 : !amdg.buffer_desc<f32> -> tensor<32xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %[[RSRC]], %{{.*}}, %{{.*}}, 0 : f32
    amdg.buffer_store %0, %desc[%offsets] soffset = %s : i32 : !amdg.buffer_desc<f32> -> tensor<32xf32, #blocked>
    tt.return
  }
}

// -----

// scope selects SCOPE and the gfx1250 policy selects TH, independently.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_scope_policy
  tt.func @buffer_scope_policy(%arg0: !tt.ptr<f32>, %offsets: tensor<32xi32, #blocked>) {
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 1 : f32
    %0 = amdg.buffer_load %arg0[%offsets] {cachePolicy = #amdg.cache_policy_gfx1250<temporalHint = nontemporal>} : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 0 : f32
    %1 = amdg.buffer_load %arg0[%offsets] {scope = 2 : i32} : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 19 : f32
    %2 = amdg.buffer_load %arg0[%offsets] {scope = 1 : i32, cachePolicy = #amdg.cache_policy_gfx1250<temporalHint = last_use>} : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 26 : f32
    amdg.buffer_store %0, %arg0[%offsets] {scope = 3 : i32, cachePolicy = #amdg.cache_policy_gfx1250<temporalHint = high_temporal>} : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    tt.return
  }
}
