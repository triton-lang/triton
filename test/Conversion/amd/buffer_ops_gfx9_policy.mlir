// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s

// A buffer descriptor packs its base, num_records and (14-bit) stride into
// the resource; soffset feeds the instruction's scalar offset, scaled from
// elements to bytes.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_desc_soffset
  tt.func @buffer_desc_soffset(%arg0: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>, %s: i32, %n: i32, %stride: i32) {
    // CHECK: %[[STRIDE16:.*]] = llvm.trunc %{{.*}} : i32 to i16
    // CHECK: %[[MASK:.*]] = llvm.mlir.constant(16383 : i16) : i16
    // CHECK: %[[STRIDE:.*]] = llvm.and %[[STRIDE16]], %[[MASK]] : i16
    // CHECK: %[[NUM:.*]] = llvm.zext %{{.*}} : i32 to i64
    // CHECK: %[[RSRC:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %[[STRIDE]], %[[NUM]], %{{.*}}
    %desc = amdg.make_buffer_desc %arg0, %n, %stride : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: %[[SOFF:.*]] = llvm.mul %[[C4]], %{{.*}} : i32
    // CHECK: rocdl.raw.ptr.buffer.load %[[RSRC]], %{{.*}}, %[[SOFF]], 0 : f32
    %0 = amdg.buffer_load %desc[%offsets] soffset = %s : !amdg.buffer_desc<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %[[RSRC]], %{{.*}}, %{{.*}}, 0 : f32
    amdg.buffer_store %0, %desc[%offsets] soffset = %s : !amdg.buffer_desc<f32> -> tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

// Record indices make the access indexed (struct): masked-off lanes get an
// out-of-bounds index instead of an out-of-bounds offset.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_desc_indices
  tt.func @buffer_desc_indices(%arg0: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>, %indices: tensor<64xi32, #blocked>, %mask: tensor<64xi1, #blocked>, %n: i32, %stride: i32) {
    %desc = amdg.make_buffer_desc %arg0, %n, %stride : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // CHECK: %[[OOB:.*]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[INDEX:.*]] = llvm.select %{{.*}}, %{{.*}}, %[[OOB]] : i1, i32
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.struct.ptr.buffer.load"(%{{.*}}, %[[INDEX]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<8>, i32, i32, i32, i32) -> f32
    %0 = amdg.buffer_load %desc[%offsets], %mask indices = %indices : !amdg.buffer_desc<f32> -> tensor<64xf32, #blocked>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.struct.ptr.buffer.store"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32, i32) -> ()
    amdg.buffer_store %0, %desc[%offsets] indices = %indices : !amdg.buffer_desc<f32> -> tensor<64xf32, #blocked>
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
